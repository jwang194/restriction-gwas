import contextlib
import logging
import tempfile
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import polars as pl
import typer
from igwas.igwas import igwas_files
from rich.logging import RichHandler
from rich.progress import track

from restrict_gwas.cli.ldsc import app as ldsc_app
from restrict_gwas.cli.ldsc import ldsc_rg
from restrict_gwas.cli.utils import munge_parallel, remove_all_suffixes, rg_parallel
from restrict_gwas.estimators import N_SEVER, MaxGCP_R, fit_model

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("rich")


app = typer.Typer(
    add_completion=False, context_settings={"help_option_names": ["-h", "--help"]}
)
app.add_typer(ldsc_app, name="ldsc", help="LDSC commands")


@contextlib.contextmanager
def _maybe_tmpdir(path: Optional[Path]):
    """Yield path as-is (creating it if needed), or yield a fresh temporary directory."""
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        yield path
    else:
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)


@app.command(name="pcov")
def compute_phenotypic_covariance(
    phenotype_file: Annotated[
        Path,
        typer.Option(
            "--pheno", exists=True, help="Path to phenotype file", show_default=False
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option("--out", help="Path to output file", show_default=False),
    ],
    covariate_file: Annotated[
        Optional[Path],
        typer.Option("--covar", exists=True, help="Path to covariate file"),
    ] = None,
    person_id_col: Annotated[
        list[str], typer.Option("--person-id", help="Person ID column(s)")
    ] = ["#FID", "IID"],
    no_intercept: Annotated[
        bool, typer.Option("--no-intercept", help="Do not add intercept to covariates")
    ] = False,
) -> None:
    """Compute phenotypic covariance matrix from a phenotype file.

    Covariates are optional. When provided, the covariates will be residualized
    out and the resulting file will contain the partial covariance matrix.

    Person ID columns will be used to join the phenotype and covariate files and
    will be ignored in the computation of the covariance matrix. Multiple ID
    can be specified one or more times like this: --person-id FID --person-id IID
    """
    add_intercept = not no_intercept
    logger.info("Computing covariance")
    if covariate_file is None and add_intercept:
        logger.warning(
            "You have specified to add an intercept to the covariates, but no "
            "covariate file was provided. No intercept will be added."
        )
    sep = "," if phenotype_file.suffix == ".csv" else "\t"
    phenotype_df = pl.read_csv(phenotype_file, separator=sep)
    logger.info(
        f"Found {phenotype_df.shape[0]} samples, {phenotype_df.shape[1]} phenotypes"
    )
    has_person_ids = person_id_col is not None and len(person_id_col) > 0
    if has_person_ids:
        phenotype_names = phenotype_df.drop(person_id_col).columns
    else:
        phenotype_names = phenotype_df.columns
    phenotype_df = phenotype_df.with_columns(
        (pl.col(c) / pl.col(c).std()).alias(c) for c in phenotype_names
    )
    if covariate_file is not None:
        logger.info("Residualizing covariates")
        sep = "," if covariate_file.suffix == ".csv" else "\t"
        covariate_df = pl.read_csv(covariate_file, separator="\t")
        if add_intercept:
            covariate_df = covariate_df.with_columns(
                pl.lit(1.0).alias("const").cast(pl.Float64)
            )
        has_person_ids = person_id_col is not None and len(person_id_col) > 0
        if has_person_ids:
            covariate_names = covariate_df.drop(person_id_col).columns
        else:
            covariate_names = covariate_df.columns
        # Dummy-encode any categorical/string columns (drop first level to avoid collinearity)
        categorical_cols = [
            c for c in covariate_names
            if covariate_df[c].dtype in (pl.String, pl.Categorical, pl.Utf8)
        ]
        if categorical_cols:
            logger.info(f"Dummy-encoding categorical covariates: {categorical_cols}")
            covariate_df = covariate_df.to_dummies(columns=categorical_cols)
            # Drop the first dummy level per variable to avoid perfect collinearity
            for col in categorical_cols:
                first_dummy = next(c for c in covariate_df.columns if c.startswith(f"{col}_"))
                covariate_df = covariate_df.drop(first_dummy)
        # Prefix covariate columns to avoid name collisions with phenotype columns on join
        covar_prefix = "__covar_"
        covariate_df = covariate_df.rename(
            {c: f"{covar_prefix}{c}" for c in covariate_df.columns if c not in person_id_col}
        )
        if has_person_ids:
            covariate_names = covariate_df.drop(person_id_col).columns
        else:
            covariate_names = covariate_df.columns
        merged_df = phenotype_df.join(covariate_df, on=person_id_col)
        X = merged_df.select(covariate_names).cast(pl.Float64).to_numpy()
        Y = merged_df.select(phenotype_names).to_numpy()
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        Y_resid = Y - X @ beta
        covariance = np.cov(Y_resid.T)
    else:
        X = phenotype_df.select(phenotype_names).to_numpy()
        covariance = np.cov(X.T)

    index = pd.Index(phenotype_names, name="phenotype")
    covariance_df = pd.DataFrame(covariance, index=index, columns=index)
    logger.info(f"Writing covariance to {output_file}")
    covariance_df.to_csv(output_file, sep="\t")

def read_ldsc_cross_intercept_output(output_path: Path) -> pd.DataFrame:
    """Read cross-trait intercept estimates from an LDSC log file.

    Parses the summary table printed at the end of a multi-phenotype LDSC
    --rg run, extracting the gcov_int (cross-trait intercept) column for
    each pairwise comparison.

    Returns:
        DataFrame with columns p1, p2, gcov_int.
    """
    with open(output_path) as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("p1 ") or line.strip() == "p1":
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            f"Could not find summary table in LDSC output: {output_path}"
        )

    header = lines[header_idx].split()
    if "gcov_int" not in header:
        raise ValueError(
            f"'gcov_int' column not found in LDSC summary table at {output_path}. "
            f"Found columns: {header}"
        )

    p1_col = header.index("p1")
    p2_col = header.index("p2")
    gcov_int_col = header.index("gcov_int")

    records = []
    for line in lines[header_idx + 1 :]:
        parts = line.split()
        if len(parts) < len(header):
            break
        p1 = Path(parts[p1_col]).with_suffix("").with_suffix("").name
        p2 = Path(parts[p2_col]).with_suffix("").with_suffix("").name
        try:
            gcov_int = float(parts[gcov_int_col])
        except ValueError:
            gcov_int = 0.0
        records.append({"p1": p1, "p2": p2, "gcov_int": gcov_int})

    return pd.DataFrame(records)


def _estimate_pcov_from_sumstats(
    gwas_paths: list[Path],
    ldsc_reference: Path,
    ldsc_weights: Path,
    sample_size_col: str,
    std_error_col: str,
    maf_col: str,
    snp_col: str,
    a1_col: str,
    a2_col: str,
    p_col: str,
    signed_sumstat_col: str,
    signed_sumstat_null: float,
    use_stem: bool,
    n_threads: int,
    sample_overlap_file: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Estimate the phenotypic covariance matrix from GWAS summary statistics.

    Diagonal entries are estimated per phenotype as:
        Var(y) = median_SNPs( N * 2 * f * (1-f) * SE^2 )
    where N is sample size, f is allele frequency, and SE is the standard error.
    This follows from the asymptotic OLS variance formula Var(beta_hat) =
    Var(y) / (N * Var(g)) with Var(g) = 2*f*(1-f).

    Off-diagonal entries are estimated as:
        Cov(y_i, y_j) = gcov_int_ij * sqrt(N_i * N_j) / N_s_ij
                        * sqrt( Var(y_i) * Var(y_j) )
    where gcov_int_ij is the bivariate LDSC cross-trait intercept, N_i and N_j
    are the median sample sizes from the respective sumstats files, and N_s_ij
    is the number of shared samples (from sample_overlap_file if provided,
    otherwise assumed equal to sqrt(N_i * N_j), which simplifies to the
    intercept equalling the phenotypic correlation under full sample overlap).

    Args:
        gwas_paths: Paths to all GWAS summary statistics files.
        ldsc_reference: Path to LDSC reference LD scores directory/prefix.
        ldsc_weights: Path to LDSC regression weights directory/prefix.
        sample_size_col: Column name for sample size (N).
        std_error_col: Column name for standard error (SE).
        maf_col: Column name for allele frequency (f).
        snp_col, a1_col, a2_col, p_col: Column names for LDSC munging.
        signed_sumstat_col, signed_sumstat_null: Signed sumstat for munging.
        use_stem: Whether to use file stems as phenotype names.
        n_threads: Number of threads for parallel LDSC runs.
        sample_overlap_file: Optional path to a tab-separated pairwise sample
            overlap table (phenotype names as row/column headers, N_s values as
            entries). When omitted, sqrt(N_i * N_j) is used as N_s_ij.

    Returns:
        Square DataFrame indexed by phenotype name containing the estimated
        partial phenotypic variance-covariance matrix.
    """
    # Step 1: Load sample overlap table if provided
    if sample_overlap_file is not None:
        sep = "," if sample_overlap_file.suffix == ".csv" else "\t"
        overlap_df = pd.read_csv(sample_overlap_file, sep=sep, index_col=0)
    else:
        overlap_df = None

    # Step 2: Munge all sumstats (includes SE and FRQ), then estimate variances
    # from the munged files so everything uses the cache.
    munge_cache = cache_dir / "munged" if cache_dir is not None else None
    rg_pcov_cache = cache_dir / "rg_pcov" if cache_dir is not None else None
    with _maybe_tmpdir(munge_cache) as munge_dir, _maybe_tmpdir(rg_pcov_cache) as rg_directory:
        logger.info("Munging sumstats for phenotypic covariance estimation")
        munged_paths = munge_parallel(
            gwas_paths=gwas_paths,
            output_dir=munge_dir,
            snp_col=snp_col,
            a1_col=a1_col,
            a2_col=a2_col,
            sample_size_col=sample_size_col,
            p_col=p_col,
            signed_sumstat_col=signed_sumstat_col,
            signed_sumstat_null=signed_sumstat_null,
            n_threads=n_threads,
            std_error_col=std_error_col,
            maf_col=maf_col,
        )

        # Estimate diagonal variances and effective N from munged files
        phenotype_names = []
        variances = []
        effective_n = []
        for gwas_path, munged_path in zip(gwas_paths, munged_paths):
            df = pd.read_csv(munged_path, sep="\t", usecols=["N", "SE", "FRQ"])
            N = df["N"].to_numpy(dtype=float)
            se = df["SE"].to_numpy(dtype=float)
            f = df["FRQ"].to_numpy(dtype=float)
            var_est = float(np.median(N * 2.0 * f * (1.0 - f) * se**2))
            variances.append(var_est)
            effective_n.append(float(np.median(N)))
            name = remove_all_suffixes(gwas_path).name if use_stem else gwas_path.name
            phenotype_names.append(name)

        variances_arr = np.array(variances)
        effective_n_arr = np.array(effective_n)

        ref_path = ldsc_reference.as_posix() + "/" if ldsc_reference.is_dir() else ldsc_reference.as_posix()
        w_path = ldsc_weights.as_posix() + "/" if ldsc_weights.is_dir() else ldsc_weights.as_posix()
        logger.info("Running all-pairwise LDSC for cross-trait intercepts")
        rg_log_paths = rg_parallel(
            gwas_paths=munged_paths,
            targets=munged_paths,
            ldsc_reference=ref_path,
            ldsc_weights=w_path,
            directory=rg_directory,
            n_threads=n_threads,
        )

        # Step 4: Parse cross-trait intercepts from log files
        intercept_dict: dict[tuple[str, str], float] = {}
        for log_path in rg_log_paths:
            intercept_df = read_ldsc_cross_intercept_output(log_path)
            if use_stem:
                intercept_df["p1"] = intercept_df["p1"].apply(
                    lambda x: remove_all_suffixes(Path(x)).name
                )
                intercept_df["p2"] = intercept_df["p2"].apply(
                    lambda x: remove_all_suffixes(Path(x)).name
                )
            for _, row in intercept_df.iterrows():
                intercept_dict[(row["p1"], row["p2"])] = row["gcov_int"]
                intercept_dict[(row["p2"], row["p1"])] = row["gcov_int"]

    # Step 5: Build the pcov matrix
    n = len(phenotype_names)
    pcov = np.zeros((n, n))
    for i in range(n):
        pcov[i, i] = variances_arr[i]
    for i in range(n):
        for j in range(i + 1, n):
            intercept = intercept_dict.get(
                (phenotype_names[i], phenotype_names[j]), 0.0
            )
            # Scaling factor: sqrt(N_i * N_j) / N_s_ij converts the LDSC
            # cross-trait intercept to a phenotypic correlation.
            if overlap_df is not None:
                n_s = float(
                    overlap_df.loc[phenotype_names[i], phenotype_names[j]]
                )
            else:
                n_s = np.sqrt(effective_n_arr[i] * effective_n_arr[j])
            scale = np.sqrt(effective_n_arr[i] * effective_n_arr[j]) / n_s
            cov_ij = intercept * scale * np.sqrt(variances_arr[i] * variances_arr[j])
            pcov[i, j] = cov_ij
            pcov[j, i] = cov_ij

    index = pd.Index(phenotype_names, name="phenotype")
    return pd.DataFrame(pcov, index=index, columns=index)


def read_ldsc_gcov_output(output_path: Path) -> pd.DataFrame:
    """Read the genetic covariance estimates from the LDSC output file."""
    with open(output_path) as f:
        lines = f.readlines()

    files = None
    # One record per feature: (h2, gencov). We open a new record when we see
    # "Heritability of phenotype N/M" with N > 1, and close it when we see
    # gencov or when the next phenotype block starts. This handles the case
    # where LDSC fails to converge for a pair and omits the gencov line.
    records: list[dict] = []   # [{h2, gencov}] one per non-target phenotype
    current: dict | None = None

    def _flush(rec: dict | None) -> None:
        if rec is not None:
            rec.setdefault("h2", float("nan"))
            rec.setdefault("gencov", float("nan"))
            records.append(rec)

    for line in lines:
        if line.startswith("--rg "):
            files = line.replace("--rg ", "").replace(" \\", "").split(",")
            continue
        if line.startswith("Heritability of phenotype "):
            state_str = line.replace("Heritability of phenotype ", "").split("/")[0]
            pheno_idx = int(state_str)
            if pheno_idx == 1:
                # Target self-pair — flush any open record but don't open a new one
                _flush(current)
                current = None
            else:
                _flush(current)
                current = {}
            continue
        if current is None:
            continue
        if line.startswith("Total Observed scale h2: "):
            val = line.replace("Total Observed scale h2: ", "").split()[0]
            try:
                current["h2"] = float(val)
            except ValueError:
                current["h2"] = float("nan")
        elif line.startswith("Total Observed scale gencov: "):
            val = line.replace("Total Observed scale gencov: ", "").split()[0]
            try:
                current["gencov"] = float(val)
            except ValueError:
                current["gencov"] = float("nan")
            _flush(current)
            current = None

    _flush(current)  # close any trailing record

    if files is None:
        raise ValueError("Could not find files in LDSC output")

    files = [Path(f).with_suffix("").with_suffix("").name for f in files]
    target = files[0]
    features = files[1:]

    if len(records) != len(features):
        logger.warning(
            f"Expected {len(features)} feature records in {output_path.name}, "
            f"got {len(records)}. Some pairs may have failed to converge."
        )
        # Pad with NaN rows so downstream code still gets a full-size matrix
        while len(records) < len(features):
            records.append({"h2": float("nan"), "gencov": float("nan")})
        records = records[: len(features)]

    return (
        pd.DataFrame(
            {
                "target": [target] * len(features),
                "feature": features,
                "heritability": [r["h2"] for r in records],
                "genetic_covariance": [r["gencov"] for r in records],
            }
        )
        .pivot(index="feature", columns="target", values="genetic_covariance")
        .rename_axis(columns=None)
    )

def _rg_parallel_pairs(
    target_munged: Path,
    feature_munged_paths: list[Path],
    ldsc_reference: str,
    ldsc_weights: str,
    directory: Path,
    n_threads: int,
) -> list[Path]:
    """Run ldsc --rg target,feature_i in parallel, one call per feature."""
    import concurrent.futures

    def _run_pair(feat_path: Path) -> Path:
        target_name = remove_all_suffixes(target_munged).name
        feat_name = remove_all_suffixes(feat_path).name
        output_stem = directory / f"{target_name}_x_{feat_name}"
        output_log = Path(str(output_stem) + ".log")
        if output_log.exists():
            logger.info(f"Skipping LDSC rg (cached): {output_log.name}")
            return output_log
        ldsc_rg(
            [target_munged, feat_path],
            ldsc_reference,
            ldsc_weights,
            output_stem,
        )
        if not output_log.exists():
            raise ValueError(f"RG output file {output_log} not found")
        return output_log

    if n_threads <= 1:
        return list(
            track(
                (_run_pair(f) for f in feature_munged_paths),
                total=len(feature_munged_paths),
                description="Computing genetic covariances...",
            )
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {
            executor.submit(_run_pair, f): i
            for i, f in enumerate(feature_munged_paths)
        }
        results = [None] * len(feature_munged_paths)
        for f in track(
            concurrent.futures.as_completed(futures),
            total=len(feature_munged_paths),
            description="Computing genetic covariances...",
        ):
            results[futures[f]] = f.result()
    return results


def compute_genetic_covariance_vector(
    *,
    gwas_paths: Annotated[
        list[Path],
        typer.Argument(
            exists=True, help="Path to GWAS summary statistics", show_default=False
        ),
    ],
    target: Annotated[
        Path, typer.Option(exists=True, help="Target phenotype(s) for MaxGCP")
    ],
    ldsc_reference: Annotated[
        Path,
        typer.Option("--ldsc-reference", exists=True, help="Path to LDSC reference LD scores (--ref-ld-chr)"),
    ],
    ldsc_weights: Annotated[
        Path,
        typer.Option("--ldsc-weights", exists=True, help="Path to LDSC regression weights (--w-ld-chr)"),
    ],
    output_file: Annotated[
        Path,
        typer.Option("--out", help="Path to output file", show_default=False),
    ] = Path("maxgcp_genetic_covariance.tsv"),
    snp_col: Annotated[str, typer.Option("--snp", help="Name of SNP column")] = "ID",
    a1_col: Annotated[
        str, typer.Option("--a1", help="Name of effect allele column")
    ] = "A1",
    a2_col: Annotated[
        str, typer.Option("--a2", help="Name of non-effect allele column")
    ] = "A2",
    sample_size_col: Annotated[
        str, typer.Option("--sample-size", help="Name of sample size column")
    ] = "OBS_CT",
    p_col: Annotated[str, typer.Option("--p", help="Name of p-value column")] = "P",
    signed_sumstat_col: Annotated[
        str,
        typer.Option(
            "--signed-sumstat", help="Name of signed sumstat column (e.g. Z, OR)"
        ),
    ] = "T_STAT",
    signed_sumstat_null: Annotated[
        float,
        typer.Option(
            "--signed-sumstat-null", help="Null value for the signed sumstat column"
        ),
    ] = 0.0,
    use_stem: Annotated[
        bool, typer.Option(help="Use stem of GWAS file as phenotype name")
    ] = True,
    n_threads: Annotated[
        int, typer.Option("--n-threads", help="Number of threads for LDSC")
    ] = 1,
    cache_dir: Optional[Path] = None,
) -> None:
    """Compute a genetic covariance vector (features x target) using LDSC."""
    if target not in gwas_paths:
        raise ValueError(f"Target {target} not found in GWAS paths")

    gwas_paths = [target] + [p for p in gwas_paths if p != target]

    munge_cache = cache_dir / "munged" if cache_dir is not None else None
    rg_gcov_cache = cache_dir / "rg_gcov" if cache_dir is not None else None
    with _maybe_tmpdir(munge_cache) as fmt_dir, _maybe_tmpdir(rg_gcov_cache) as rg_dir:
        logger.info("Formatting sumstats for LDSC")
        output_paths = munge_parallel(
            gwas_paths=gwas_paths,
            output_dir=fmt_dir,
            snp_col=snp_col,
            a1_col=a1_col,
            a2_col=a2_col,
            sample_size_col=sample_size_col,
            p_col=p_col,
            signed_sumstat_col=signed_sumstat_col,
            signed_sumstat_null=signed_sumstat_null,
            n_threads=n_threads,
        )
        logger.info("Computing genetic covariances using LDSC")
        ref_path = ldsc_reference.as_posix() + "/" if ldsc_reference.is_dir() else ldsc_reference.as_posix()
        w_path = ldsc_weights.as_posix() + "/" if ldsc_weights.is_dir() else ldsc_weights.as_posix()

        # Run one ldsc --rg call per feature (target + feature_i) in parallel.
        # Each 2-file call produces one gencov estimate; much faster than a
        # single 29-file call that processes pairs sequentially.
        target_munged = output_paths[0]
        feature_munged = output_paths[1:]
        rg_log_paths = _rg_parallel_pairs(
            target_munged=target_munged,
            feature_munged_paths=feature_munged,
            ldsc_reference=ref_path,
            ldsc_weights=w_path,
            directory=rg_dir,
            n_threads=n_threads,
        )

        # Parse each per-feature log and combine
        gcov_records = []
        for log_path in rg_log_paths:
            try:
                pair_df = read_ldsc_gcov_output(log_path)
                gcov_records.append(pair_df)
            except Exception as e:
                logger.warning(f"Failed to parse {log_path.name}: {e}")
        if not gcov_records:
            raise RuntimeError("All LDSC rg calls failed")
        result_df = pd.concat(gcov_records)

    if use_stem:
        result_df.index = pd.Index(
            [remove_all_suffixes(Path(p)) for p in result_df.index], name="phenotype"
        )
        result_df.columns = [remove_all_suffixes(Path(p)) for p in result_df.columns]

    result_df.to_csv(output_file, sep="\t")


def fit_command(
    genetic_covariance_file: Annotated[
        Path,
        typer.Option("--gcov", exists=True, help="Path to genetic covariance file"),
    ],
    phenotypic_covariance_file: Annotated[
        Path,
        typer.Option("--pcov", exists=True, help="Path to phenotypic covariance file"),
    ],
    target: Annotated[str, typer.Option(help="Target phenotype for MaxGCP")],
    output_file: Annotated[
        Path,
        typer.Option("--out", help="Path to output file", show_default=False),
    ] = Path("maxgcp_weights.tsv"),
):
    """Fit MaxGCP_R using existing genetic and phenotypic covariances."""
    logger.info("Fitting MaxGCP_R phenotype")
    sep = "," if genetic_covariance_file.suffix == ".csv" else "\t"
    genetic_covariance_df = pd.read_csv(genetic_covariance_file, sep=sep, index_col=0)
    sep = "," if phenotypic_covariance_file.suffix == ".csv" else "\t"
    phenotypic_covariance_df = pd.read_csv(
        phenotypic_covariance_file, sep=sep, index_col=0
    )
    if (
        phenotypic_covariance_df.index.values.tolist()
        != phenotypic_covariance_df.columns.values.tolist()
    ):
        raise ValueError("Phenotypic covariance matrix must be symmetric")
    if target not in genetic_covariance_df.columns:
        raise ValueError(f"Target {target} not found in genetic covariance file")

    features = phenotypic_covariance_df.index.drop(target, errors="ignore").tolist()

    gcov_vec = genetic_covariance_df.loc[features, target].values
    nan_mask = np.isnan(gcov_vec)
    if nan_mask.any():
        nan_features = [f for f, m in zip(features, nan_mask) if m]
        logger.warning(
            f"NaN genetic covariance for {len(nan_features)} feature(s): "
            f"{nan_features}. Filling with 0."
        )
        gcov_vec = np.where(nan_mask, 0.0, gcov_vec)
    pcov_mat = phenotypic_covariance_df.loc[features, features].values
    logger.info(f"Using {len(features)} features")
    data = MaxGCP_R(
        endophenotype_names=features,
        target_name=target,
        cov_G_vec=gcov_vec,
        cov_P=pcov_mat,
    )
    maxgcp_weights = fit_model(data)
    maxgcp_weights_df = pd.DataFrame(
        maxgcp_weights,
        index=pd.Index(features, name="feature"),
        columns=pd.Index([target], name="target"),
    )
    logger.info(f"Writing weights to {output_file}")
    maxgcp_weights_df.to_csv(output_file, sep="\t")


def _run_indirect_gwas_in_memory(
    gwas_paths: list[Path],
    projection_coefficient_file: Path,
    phenotype_covariance_file: Path,
    n_covar: int,
    output_file: Path,
    snp_col: str,
    chrom_col: str,
    pos_col: str,
    a1_col: str,
    a2_col: str,
    beta_col: str,
    std_error_col: str,
    sample_size_col: str,
    n_threads: int,
) -> None:
    """In-memory indirect GWAS: reads all files at once and does a single matrix pass."""
    import concurrent.futures
    from scipy.stats import t as t_dist

    W_df = pd.read_csv(projection_coefficient_file, sep="\t", index_col=0)
    C_df = pd.read_csv(phenotype_covariance_file, sep="\t", index_col=0)

    features = W_df.index.tolist()
    C_df = C_df.loc[features, features]
    W_arr = W_df.values.astype(float)  # (P, K)
    C_arr = C_df.values.astype(float)  # (P, P)
    projection_names = W_df.columns.tolist()
    write_phenotype_id = len(projection_names) > 1

    name_to_path = {remove_all_suffixes(p).name: p for p in gwas_paths}
    try:
        ordered_paths = [name_to_path[f] for f in features]
    except KeyError as e:
        raise ValueError(
            f"Feature {e} from projection matrix not found among GWAS paths. "
            f"Available: {list(name_to_path)}"
        )

    read_cols = [snp_col, chrom_col, pos_col, a1_col, a2_col,
                 beta_col, std_error_col, sample_size_col]

    def _read(path: Path) -> pd.DataFrame:
        sep = "," if path.suffix == ".csv" else "\t"
        return pd.read_csv(path, sep=sep, usecols=read_cols)

    logger.info(f"Reading {len(ordered_paths)} GWAS files into memory")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
        dfs = list(ex.map(_read, ordered_paths))

    snp_ids    = dfs[0][snp_col].values
    chrom_vals = dfs[0][chrom_col].values
    pos_vals   = dfs[0][pos_col].values
    a1_vals    = dfs[0][a1_col].values
    a2_vals    = dfs[0][a2_col].values

    betas   = np.column_stack([df[beta_col].values        for df in dfs]).astype(float)
    ses     = np.column_stack([df[std_error_col].values   for df in dfs]).astype(float)
    obs_cts = np.column_stack([df[sample_size_col].values for df in dfs]).astype(float)
    del dfs

    # Projected betas: (M, K)
    beta_proj = betas @ W_arr

    # Per-SNP genetic variance estimate (averaged over phenotypes)
    # Matches multi_indirect_gwas.py: pcov_diag[p] / (SE[m,p]^2 * (N-k-1) + beta[m,p]^2)
    pcov_diag = np.diag(C_arr)  # (P,)
    denom = ses ** 2 * (obs_cts - n_covar - 1) + betas ** 2  # (M, P)
    with np.errstate(divide="ignore", invalid="ignore"):
        partial_G = np.where(denom > 0, pcov_diag[None, :] / denom, np.nan)
    partial_G = np.nanmean(partial_G, axis=1, keepdims=True)  # (M, 1)

    # diag(W^T C W): per-projection phenotypic variance of the projected phenotype
    WtCW_diag = np.diag(W_arr.T @ C_arr @ W_arr)  # (K,)

    N_eff = obs_cts.mean(axis=1, keepdims=True) - n_covar - 1  # (M, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        se_proj_sq = (WtCW_diag[None, :] / partial_G - beta_proj ** 2) / N_eff
    se_proj = np.sqrt(np.maximum(se_proj_sq, 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = beta_proj / np.where(se_proj > 0, se_proj, np.nan)
    p_value = 2 * t_dist.sf(np.abs(t_stat), df=N_eff)
    sample_size = obs_cts.mean(axis=1)

    base = {
        snp_col: snp_ids,
        chrom_col: chrom_vals,
        pos_col: pos_vals,
        a1_col: a1_vals,
        a2_col: a2_vals,
    }
    if write_phenotype_id:
        frames = [
            pd.DataFrame({
                "phenotype_id": name,
                **base,
                "BETA": beta_proj[:, k],
                "SE": se_proj[:, k],
                "T_STAT": t_stat[:, k],
                "P": p_value[:, k],
                "OBS_CT": sample_size,
            })
            for k, name in enumerate(projection_names)
        ]
        result = pd.concat(frames, ignore_index=True)
    else:
        result = pd.DataFrame({
            **base,
            "BETA": beta_proj[:, 0],
            "SE": se_proj[:, 0],
            "T_STAT": t_stat[:, 0],
            "P": p_value[:, 0],
            "OBS_CT": sample_size,
        })

    result.to_csv(output_file, sep="\t", index=False, compression="gzip")


def run_indirect_gwas(
    gwas_paths: Annotated[
        list[Path],
        typer.Argument(
            exists=True, help="Path to GWAS summary statistics", show_default=False
        ),
    ],
    projection_coefficient_file: Annotated[
        Path,
        typer.Option("--proj", exists=True, help="Path to projection coefficient file"),
    ],
    phenotype_covariance_file: Annotated[
        Path,
        typer.Option("--pcov", exists=True, help="Path to phenotypic covariance file"),
    ],
    n_covar: Annotated[
        int,
        typer.Option("--n-covar", help="Number of covariates to use"),
    ],
    output_file: Annotated[
        Path,
        typer.Option("--out", help="Path to output file"),
    ],
    snp_col: Annotated[str, typer.Option("--snp", help="Name of SNP column")] = "ID",
    chrom_col: Annotated[str, typer.Option("--chrom", help="Name of chromosome column")] = "#CHROM",
    pos_col: Annotated[str, typer.Option("--pos", help="Name of base pair position column")] = "POS",
    a1_col: Annotated[str, typer.Option("--a1", help="Name of effect allele column")] = "A1",
    a2_col: Annotated[str, typer.Option("--a2", help="Name of non-effect allele column")] = "A2",
    beta_col: Annotated[
        str, typer.Option("--beta", help="Name of beta column")
    ] = "BETA",
    std_error_col: Annotated[
        str, typer.Option("--std-error", help="Name of standard error column")
    ] = "SE",
    sample_size_col: Annotated[
        str, typer.Option("--sample-size", help="Name of sample size column")
    ] = "OBS_CT",
    compress: Annotated[
        bool, typer.Option("--compress", help="Compress output file (igwas path only)")
    ] = True,
    use_stem: Annotated[
        bool, typer.Option(help="Use stem of GWAS file as phenotype name")
    ] = True,
    chunksize: Annotated[
        int, typer.Option("--chunksize", help="Chunksize for IGWAS")
    ] = 100_000,
    n_threads: Annotated[
        int, typer.Option("--n-threads", help="Number of threads")
    ] = 1,
    in_memory: Annotated[
        bool,
        typer.Option(
            "--in-memory",
            help="Load all GWAS files into memory and compute in one matrix pass. "
            "Faster than the default igwas streaming approach for large feature sets. "
            "Output is gzipped TSV with A1/A2 columns.",
        ),
    ] = False,
):
    """Compute GWAS summary statistics for a projected phenotype."""
    if not output_file.name.endswith(".sumstats.gz"):
        output_file = Path(str(output_file) + ".sumstats.gz")

    if not use_stem:
        raise NotImplementedError(
            "Indirect GWAS only currently supports GWAS files where the file "
            "stem represents the phenotype"
        )

    if in_memory:
        _run_indirect_gwas_in_memory(
            gwas_paths=gwas_paths,
            projection_coefficient_file=projection_coefficient_file,
            phenotype_covariance_file=phenotype_covariance_file,
            n_covar=n_covar,
            output_file=output_file,
            snp_col=snp_col,
            chrom_col=chrom_col,
            pos_col=pos_col,
            a1_col=a1_col,
            a2_col=a2_col,
            beta_col=beta_col,
            std_error_col=std_error_col,
            sample_size_col=sample_size_col,
            n_threads=n_threads,
        )
        return

    projections = pd.read_csv(
        projection_coefficient_file, sep="\t", index_col=0, nrows=0
    ).columns.tolist()
    write_projection = len(projections) > 1

    # igwas strips ALL dot-separated suffixes from filenames to get phenotype names,
    # which collapses e.g. blood_biochem_pcs.PC1.glm.linear → blood_biochem_pcs.
    # Fix: create symlinks whose phenotype-identifying stem has no dots (replace "." with "_"),
    # preserving the known GWAS extensions so igwas can still strip them correctly.
    # E.g. blood_biochem_pcs.PC1.glm.linear → blood_biochem_pcs_PC1.glm.linear
    # igwas strips .glm.linear → blood_biochem_pcs_PC1 ✓
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        safe_gwas_paths = []
        rename_map: dict[str, str] = {}
        for p in gwas_paths:
            # stripped_name is the phenotype identifier (known extensions removed)
            stripped_name = remove_all_suffixes(p).name
            safe_name = stripped_name.replace(".", "_")
            # suffix is only the known GWAS extensions that were stripped
            suffix = p.name[len(stripped_name):]
            link = tmp / (safe_name + suffix)
            link.symlink_to(p.resolve())
            safe_gwas_paths.append(link)
            if stripped_name != safe_name:
                rename_map[stripped_name] = safe_name

        if rename_map:
            # Patch projection matrix index
            proj_df = pd.read_csv(
                projection_coefficient_file, sep="\t", index_col=0
            )
            proj_df.index = [rename_map.get(x, x) for x in proj_df.index]
            patched_proj = tmp / projection_coefficient_file.name
            proj_df.to_csv(patched_proj, sep="\t")
            proj_path = patched_proj

            # Patch covariance matrix index and columns
            pcov_df = pd.read_csv(
                phenotype_covariance_file, sep="\t", index_col=0
            )
            pcov_df.index = [rename_map.get(x, x) for x in pcov_df.index]
            pcov_df.columns = [rename_map.get(x, x) for x in pcov_df.columns]
            patched_pcov = tmp / phenotype_covariance_file.name
            pcov_df.to_csv(patched_pcov, sep="\t")
            pcov_path = patched_pcov
        else:
            proj_path = projection_coefficient_file
            pcov_path = phenotype_covariance_file

        igwas_files(
            projection_matrix_path=proj_path.as_posix(),
            covariance_matrix_path=pcov_path.as_posix(),
            gwas_result_paths=[p.as_posix() for p in safe_gwas_paths],
            output_file_path=output_file.as_posix(),
            num_covar=n_covar,
            chunksize=chunksize,
            variant_id=snp_col,
            beta=beta_col,
            std_error=std_error_col,
            sample_size=sample_size_col,
            num_threads=n_threads,
            capacity=n_threads,
            compress=compress,
            quiet=True,
            write_phenotype_id=write_projection,
        )


@app.command(name="maxgcp_r")
def run_command(
    gwas_paths: Annotated[
        list[Path],
        typer.Argument(
            exists=True, help="Path to GWAS summary statistics", show_default=False
        ),
    ],
    ldsc_reference: Annotated[
        Path,
        typer.Option("--ldsc-reference", exists=True, help="Path to LDSC reference LD scores (--ref-ld-chr)"),
    ],
    ldsc_weights: Annotated[
        Path,
        typer.Option("--ldsc-weights", exists=True, help="Path to LDSC regression weights (--w-ld-chr)"),
    ],
    output_file: Annotated[
        Path, typer.Option("--out", help="Path to output GWAS summary statistics file")
    ],
    target: Annotated[
        Path, typer.Option(exists=True, help="Target phenotype for MaxGCP_R")
    ],
    phenotype_covariance_file: Annotated[
        Optional[Path],
        typer.Option(
            "--pcov",
            help="Path to phenotypic covariance file. "
            "Mutually exclusive with --sumstats-only.",
        ),
    ] = None,
    snp_col: Annotated[str, typer.Option("--snp", help="Name of SNP column")] = "ID",
    chrom_col: Annotated[str, typer.Option("--chrom", help="Name of chromosome column")] = "#CHROM",
    pos_col: Annotated[str, typer.Option("--pos", help="Name of base pair position column")] = "POS",
    beta_col: Annotated[
        str, typer.Option("--beta", help="Name of beta column")
    ] = "BETA",
    std_error_col: Annotated[
        str, typer.Option("--std-error", help="Name of standard error column")
    ] = "SE",
    a1_col: Annotated[
        str, typer.Option("--a1", help="Name of effect allele column")
    ] = "A1",
    a2_col: Annotated[
        str, typer.Option("--a2", help="Name of non-effect allele column")
    ] = "A2",
    sample_size_col: Annotated[
        str, typer.Option("--sample-size", help="Name of sample size column")
    ] = "OBS_CT",
    p_col: Annotated[str, typer.Option("--p", help="Name of p-value column")] = "P",
    signed_sumstat_col: Annotated[
        str,
        typer.Option(
            "--signed-sumstat", help="Name of signed sumstat column (e.g. Z, OR)"
        ),
    ] = "T_STAT",
    signed_sumstat_null: Annotated[
        float,
        typer.Option(
            "--signed-sumstat-null", help="Null value for the signed sumstat column"
        ),
    ] = 0.0,
    use_stem: Annotated[
        bool, typer.Option(help="Use stem of GWAS file as phenotype name")
    ] = True,
    n_covar: Annotated[
        int, typer.Option("--n-covar", help="Number of covariates to use")
    ] = 0,
    compress_output: Annotated[bool, typer.Option(help="Compress output file")] = True,
    chunksize: Annotated[
        int, typer.Option("--chunksize", help="Chunksize for IGWAS")
    ] = 100_000,
    n_threads: Annotated[
        int, typer.Option("--n-threads", help="Number of threads for IGWAS")
    ] = 1,
    clean_up: Annotated[bool, typer.Option(help="Clean up intermediate files")] = True,
    sumstats_only: Annotated[
        bool,
        typer.Option(
            "--sumstats-only",
            help="Estimate phenotypic covariance matrix from summary statistics "
            "instead of a precomputed file. Mutually exclusive with --pcov.",
        ),
    ] = False,
    maf_col: Annotated[
        str,
        typer.Option(
            "--maf-col",
            help="Column name for allele frequency, used with --sumstats-only.",
        ),
    ] = "A1_FREQ",
    sample_overlap_file: Annotated[
        Optional[Path],
        typer.Option(
            "--sample-overlap",
            help="Path to a tab-separated pairwise sample overlap table "
            "(phenotype names as row/column headers, N_s values as entries). "
            "Used with --sumstats-only to scale cross-trait LDSC intercepts "
            "to phenotypic correlations. When omitted, sqrt(N_i * N_j) is used.",
        ),
    ] = None,
    cache_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--cache-dir",
            help="Directory to cache munged sumstats and LDSC rg outputs. "
            "Existing files are reused, skipping redundant computation.",
        ),
    ] = None,
    in_memory: Annotated[
        bool,
        typer.Option(
            "--in-memory",
            help="Load all GWAS files into memory and compute indirect GWAS in one "
            "matrix pass. Faster for large feature sets. Output is gzipped TSV.",
        ),
    ] = False,
):
    """Run MaxGCP on a set of GWAS summary statistics."""
    # Validate mutual exclusion between --pcov and --sumstats-only
    if sumstats_only and phenotype_covariance_file is not None:
        raise typer.BadParameter(
            "--sumstats-only and --pcov are mutually exclusive."
        )
    if not sumstats_only and phenotype_covariance_file is None:
        raise typer.BadParameter(
            "Either --pcov or --sumstats-only must be specified."
        )

    # When --sumstats-only is set, estimate the phenotypic covariance matrix
    # from summary statistics and write it to a temporary file so the rest of
    # the pipeline can proceed unchanged.
    pcov_tmp_path: Optional[Path] = None
    if sumstats_only:
        logger.info("Estimating phenotypic covariance matrix from summary statistics")
        import tempfile as _tmpfile

        pcov_df = _estimate_pcov_from_sumstats(
            gwas_paths=gwas_paths,
            ldsc_reference=ldsc_reference,
            ldsc_weights=ldsc_weights,
            sample_size_col=sample_size_col,
            std_error_col=std_error_col,
            maf_col=maf_col,
            snp_col=snp_col,
            a1_col=a1_col,
            a2_col=a2_col,
            p_col=p_col,
            signed_sumstat_col=signed_sumstat_col,
            signed_sumstat_null=signed_sumstat_null,
            use_stem=use_stem,
            n_threads=n_threads,
            sample_overlap_file=sample_overlap_file,
            cache_dir=cache_dir,
        )
        _tmp = _tmpfile.NamedTemporaryFile(suffix=".tsv", delete=False)
        pcov_tmp_path = Path(_tmp.name)
        pcov_df.to_csv(pcov_tmp_path, sep="\t")
        _tmp.close()
        phenotype_covariance_file = pcov_tmp_path

    try:
        logger.info("Computing genetic covariances using LDSC")
        target_name = remove_all_suffixes(target).name
        with (
            tempfile.NamedTemporaryFile(suffix=".tsv") as covariance_file,
            tempfile.NamedTemporaryFile(suffix=".tsv") as maxgcp_weights_file,
        ):
            covariance_path = Path(covariance_file.name)
            compute_genetic_covariance_vector(
                gwas_paths=gwas_paths,
                target=target,
                ldsc_reference=ldsc_reference,
                ldsc_weights=ldsc_weights,
                output_file=covariance_path,
                snp_col=snp_col,
                a1_col=a1_col,
                a2_col=a2_col,
                sample_size_col=sample_size_col,
                p_col=p_col,
                signed_sumstat_col=signed_sumstat_col,
                signed_sumstat_null=signed_sumstat_null,
                use_stem=use_stem,
                n_threads=n_threads,
                cache_dir=cache_dir,
            )
            maxgcp_weights_path = Path(maxgcp_weights_file.name)
            fit_command(
                genetic_covariance_file=covariance_path,
                phenotypic_covariance_file=phenotype_covariance_file,
                target=target_name,
                output_file=maxgcp_weights_path,
            )
            logger.info("Computing GWAS summary statistics for the MaxGCP_R phenotype")
            run_indirect_gwas(
                gwas_paths=gwas_paths,
                projection_coefficient_file=maxgcp_weights_path,
                phenotype_covariance_file=phenotype_covariance_file,
                n_covar=n_covar,
                output_file=output_file,
                snp_col=snp_col,
                chrom_col=chrom_col,
                pos_col=pos_col,
                a1_col=a1_col,
                a2_col=a2_col,
                beta_col=beta_col,
                std_error_col=std_error_col,
                sample_size_col=sample_size_col,
                compress=compress_output,
                use_stem=use_stem,
                chunksize=chunksize,
                n_threads=n_threads,
                in_memory=in_memory,
            )
            if not clean_up:
                logger.info(
                    "Keeping intermediate files ['maxgcp_genetic_covariance.tsv', "
                    "'maxgcp_weights.tsv']"
                )
                covariance_path.rename(
                    output_file.parent.joinpath("maxgcp_genetic_covariance.tsv")
                )
                maxgcp_weights_path.rename(
                    output_file.parent.joinpath("maxgcp_weights.tsv")
                )
                # Create empty files to avoid errors when cleaning up
                covariance_path.touch()
                maxgcp_weights_path.touch()
            else:
                logger.info("Cleaning up intermediate files")
        logger.info("Done")
    finally:
        if pcov_tmp_path is not None and pcov_tmp_path.exists():
            pcov_tmp_path.unlink()


def _compute_nsever_gcov_matrix(
    gwas_paths: list[Path],
    covariate_gwas_paths: list[Path],
    ldsc_reference: Path,
    ldsc_weights: Path,
    snp_col: str,
    a1_col: str,
    a2_col: str,
    sample_size_col: str,
    p_col: str,
    signed_sumstat_col: str,
    signed_sumstat_null: float,
    use_stem: bool,
    n_threads: int,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute genetic covariance matrix between endophenotypes and covariates.

    Runs LDSC with each covariate as the target and all endophenotype + covariate
    paths as features, then assembles the results into a matrix with endophenotype
    names as the index and covariate names as columns.

    The target's self-pair row is skipped by read_ldsc_gcov_output and dropped from results.
    for each covariate's self-pair row, which is then dropped so only the
    endophenotype-covariate cross-covariances are retained.

    Args:
        gwas_paths: GWAS summary statistics for the endophenotypes.
        covariate_gwas_paths: GWAS summary statistics for the covariates.
        ldsc_reference: Path to LDSC reference LD scores directory/prefix.
        ldsc_weights: Path to LDSC regression weights directory/prefix.
        snp_col, a1_col, a2_col, sample_size_col, p_col: Column name arguments.
        signed_sumstat_col, signed_sumstat_null: Signed sumstat for munging.
        use_stem: Whether to use file stems as phenotype names.
        n_threads: Number of threads for parallel LDSC.

    Returns:
        DataFrame of shape (n_endophenotypes, n_covariates) containing the
        estimated genetic covariances, indexed by endophenotype and covariate name.
    """
    endophenotype_names = [
        remove_all_suffixes(p).name if use_stem else p.name for p in gwas_paths
    ]
    covariate_names = [
        remove_all_suffixes(p).name if use_stem else p.name
        for p in covariate_gwas_paths
    ]

    # Deduplicate: any covariate path already in gwas_paths is not re-munged
    covar_paths_new = [p for p in covariate_gwas_paths if p not in gwas_paths]
    all_paths = gwas_paths + covar_paths_new

    munge_cache = cache_dir / "munged" if cache_dir is not None else None
    rg_nsever_cache = cache_dir / "rg_nsever" if cache_dir is not None else None
    with _maybe_tmpdir(munge_cache) as munge_dir, _maybe_tmpdir(rg_nsever_cache) as rg_directory:
        logger.info("Munging sumstats for endophenotype-covariate genetic covariance matrix")
        munged_all = munge_parallel(
            gwas_paths=all_paths,
            output_dir=munge_dir,
            snp_col=snp_col,
            a1_col=a1_col,
            a2_col=a2_col,
            sample_size_col=sample_size_col,
            p_col=p_col,
            signed_sumstat_col=signed_sumstat_col,
            signed_sumstat_null=signed_sumstat_null,
            n_threads=n_threads,
        )

        # Map each original path's stem to its munged path
        stem_to_munged = {
            remove_all_suffixes(orig).name: munged
            for orig, munged in zip(all_paths, munged_all)
        }
        munged_covariates = [stem_to_munged[n] for n in covariate_names]

        ref_path = ldsc_reference.as_posix() + "/" if ldsc_reference.is_dir() else ldsc_reference.as_posix()
        w_path = ldsc_weights.as_posix() + "/" if ldsc_weights.is_dir() else ldsc_weights.as_posix()
        logger.info("Computing endophenotype-covariate genetic covariance matrix using LDSC")
        # Run per-covariate: for each covariate, run parallel pairwise calls
        # against all endophenotypes (avoids a single slow sequential LDSC call).
        gcov_cols: list[pd.DataFrame] = []
        for covar_munged, covar_name in zip(munged_covariates, covariate_names):
            # All munged paths except this covariate are the features
            feature_paths = [p for p in munged_all if p != covar_munged]
            covar_rg_dir = rg_directory / covar_name.replace(".", "_")
            covar_rg_dir.mkdir(parents=True, exist_ok=True)
            rg_log_paths = _rg_parallel_pairs(
                target_munged=covar_munged,
                feature_munged_paths=feature_paths,
                ldsc_reference=ref_path,
                ldsc_weights=w_path,
                directory=covar_rg_dir,
                n_threads=n_threads,
            )
            # Parse each per-feature log and combine
            pair_dfs = []
            for log_path in rg_log_paths:
                try:
                    pair_df = read_ldsc_gcov_output(log_path)
                    pair_dfs.append(pair_df)
                except Exception as e:
                    logger.warning(f"Failed to parse {log_path.name}: {e}")
            if pair_dfs:
                gcov_cols.append(pd.concat(pair_dfs))

        for i, col_df in enumerate(gcov_cols):
            if use_stem:
                col_df.index = pd.Index(
                    [remove_all_suffixes(Path(p)).name for p in col_df.index],
                    name="phenotype",
                )
                col_df.columns = [
                    remove_all_suffixes(Path(p)).name for p in col_df.columns
                ]
            # Retain only rows that belong to endophenotypes (drop covariate self-rows)
            gcov_cols[i] = col_df.loc[col_df.index.isin(endophenotype_names)]

    gcov_matrix_df = pd.concat(gcov_cols, axis=1)
    # Reindex to canonical ordering matching gwas_paths / covariate_gwas_paths
    gcov_matrix_df = gcov_matrix_df.reindex(
        index=endophenotype_names, columns=covariate_names
    )
    return gcov_matrix_df


@app.command(name="nsever")
def nsever_command(
    gwas_paths: Annotated[
        list[Path],
        typer.Argument(
            exists=True, help="Path to GWAS summary statistics", show_default=False
        ),
    ],
    covariate_gwas_paths: Annotated[
        list[Path],
        typer.Option(
            "--covar-gwas",
            exists=True,
            help="Path to GWAS summary statistics for a covariate/confounder. "
            "May be repeated for multiple covariates. The genetic covariances "
            "between the endophenotypes and these covariates define the null "
            "space that N-SEVER projects out before fitting.",
            show_default=False,
        ),
    ],
    ldsc_reference: Annotated[
        Path,
        typer.Option("--ldsc-reference", exists=True, help="Path to LDSC reference LD scores (--ref-ld-chr)"),
    ],
    ldsc_weights: Annotated[
        Path,
        typer.Option("--ldsc-weights", exists=True, help="Path to LDSC regression weights (--w-ld-chr)"),
    ],
    output_file: Annotated[
        Path, typer.Option("--out", help="Path to output GWAS summary statistics file")
    ],
    target: Annotated[
        Path, typer.Option(exists=True, help="Target phenotype for MaxGCP")
    ],
    phenotype_covariance_file: Annotated[
        Optional[Path],
        typer.Option(
            "--pcov",
            help="Path to partial phenotypic variance-covariance file for the endophenotypes. "
            "Mutually exclusive with --sumstats-only.",
        ),
    ] = None,
    snp_col: Annotated[str, typer.Option("--snp", help="Name of SNP column")] = "ID",
    chrom_col: Annotated[str, typer.Option("--chrom", help="Name of chromosome column")] = "#CHROM",
    pos_col: Annotated[str, typer.Option("--pos", help="Name of base pair position column")] = "POS",
    beta_col: Annotated[
        str, typer.Option("--beta", help="Name of beta column")
    ] = "BETA",
    std_error_col: Annotated[
        str, typer.Option("--std-error", help="Name of standard error column")
    ] = "SE",
    a1_col: Annotated[
        str, typer.Option("--a1", help="Name of effect allele column")
    ] = "A1",
    a2_col: Annotated[
        str, typer.Option("--a2", help="Name of non-effect allele column")
    ] = "A2",
    sample_size_col: Annotated[
        str, typer.Option("--sample-size", help="Name of sample size column")
    ] = "OBS_CT",
    p_col: Annotated[str, typer.Option("--p", help="Name of p-value column")] = "P",
    signed_sumstat_col: Annotated[
        str,
        typer.Option(
            "--signed-sumstat", help="Name of signed sumstat column (e.g. Z, OR)"
        ),
    ] = "T_STAT",
    signed_sumstat_null: Annotated[
        float,
        typer.Option(
            "--signed-sumstat-null", help="Null value for the signed sumstat column"
        ),
    ] = 0.0,
    use_stem: Annotated[
        bool, typer.Option(help="Use stem of GWAS file as phenotype name")
    ] = True,
    n_covar: Annotated[
        int, typer.Option("--n-covar", help="Number of covariates that were used in the endophenotype GWAS")
    ] = 0,
    compress_output: Annotated[bool, typer.Option(help="Compress output file")] = True,
    chunksize: Annotated[
        int, typer.Option("--chunksize", help="Chunksize for IGWAS")
    ] = 100_000,
    n_threads: Annotated[
        int, typer.Option("--n-threads", help="Number of threads for IGWAS and LDSC")
    ] = 1,
    clean_up: Annotated[bool, typer.Option(help="Clean up intermediate files")] = True,
    sumstats_only: Annotated[
        bool,
        typer.Option(
            "--sumstats-only",
            help="Estimate phenotypic covariance matrix from summary statistics "
            "instead of a precomputed file. Mutually exclusive with --pcov.",
        ),
    ] = False,
    maf_col: Annotated[
        str,
        typer.Option(
            "--maf-col",
            help="Column name for allele frequency, used with --sumstats-only.",
        ),
    ] = "A1_FREQ",
    sample_overlap_file: Annotated[
        Optional[Path],
        typer.Option(
            "--sample-overlap",
            help="Path to a tab-separated pairwise sample overlap table "
            "(phenotype names as row/column headers, N_s values as entries). "
            "Used with --sumstats-only to scale cross-trait LDSC intercepts "
            "to phenotypic correlations. When omitted, sqrt(N_i * N_j) is used.",
        ),
    ] = None,
    cache_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--cache-dir",
            help="Directory to cache munged sumstats and LDSC rg outputs. "
            "Existing files are reused, skipping redundant computation.",
        ),
    ] = None,
    in_memory: Annotated[
        bool,
        typer.Option(
            "--in-memory",
            help="Load all GWAS files into memory and compute indirect GWAS in one "
            "matrix pass. Faster for large feature sets. Output is gzipped TSV.",
        ),
    ] = False,
):
    """Run N-SEVER on a set of GWAS summary statistics.

    Like the `maxgcp_r` command, but additionally accepts covariate GWAS summary
    statistics via --covar-gwas. The genetic covariances between the
    endophenotypes and the covariates define a null space; the MaxGCP fit is
    restricted to that null space so that the resulting phenotype is orthogonal
    to the genetic signal of each covariate (N-SEVER projection).
    """
    if sumstats_only and phenotype_covariance_file is not None:
        raise typer.BadParameter(
            "--sumstats-only and --pcov are mutually exclusive."
        )
    if not sumstats_only and phenotype_covariance_file is None:
        raise typer.BadParameter(
            "Either --pcov or --sumstats-only must be specified."
        )
    if not covariate_gwas_paths:
        raise typer.BadParameter(
            "At least one --covar-gwas path must be provided."
        )

    pcov_tmp_path: Optional[Path] = None
    if sumstats_only:
        logger.info("Estimating partial phenotypic variance-covariance matrix from summary statistics")
        import tempfile as _tmpfile

        pcov_df = _estimate_pcov_from_sumstats(
            gwas_paths=gwas_paths,
            ldsc_reference=ldsc_reference,
            ldsc_weights=ldsc_weights,
            sample_size_col=sample_size_col,
            std_error_col=std_error_col,
            maf_col=maf_col,
            snp_col=snp_col,
            a1_col=a1_col,
            a2_col=a2_col,
            p_col=p_col,
            signed_sumstat_col=signed_sumstat_col,
            signed_sumstat_null=signed_sumstat_null,
            use_stem=use_stem,
            n_threads=n_threads,
            sample_overlap_file=sample_overlap_file,
            cache_dir=cache_dir,
        )
        _tmp = _tmpfile.NamedTemporaryFile(suffix=".tsv", delete=False)
        pcov_tmp_path = Path(_tmp.name)
        pcov_df.to_csv(pcov_tmp_path, sep="\t")
        _tmp.close()
        phenotype_covariance_file = pcov_tmp_path

    try:
        logger.info("Loading partial phenotypic variance-covariance matrix")
        sep = "," if phenotype_covariance_file.suffix == ".csv" else "\t"  # type: ignore[union-attr]
        phenotypic_covariance_df = pd.read_csv(
            phenotype_covariance_file, sep=sep, index_col=0
        )
        target_name = remove_all_suffixes(target).name
        features = phenotypic_covariance_df.index.drop(target_name, errors="ignore").tolist()

        with (
            tempfile.NamedTemporaryFile(suffix=".tsv") as gcov_vec_file,
            tempfile.NamedTemporaryFile(suffix=".tsv") as weights_file,
        ):
            gcov_vec_path = Path(gcov_vec_file.name)
            gcov_matrix_path = Path(weights_file.name)

            # Step 1: Compute genetic covariance vector (endophenotypes → target)
            logger.info("Computing endophenotype-target genetic covariance vector using LDSC")
            compute_genetic_covariance_vector(
                gwas_paths=gwas_paths,
                target=target,
                ldsc_reference=ldsc_reference,
                ldsc_weights=ldsc_weights,
                output_file=gcov_vec_path,
                snp_col=snp_col,
                a1_col=a1_col,
                a2_col=a2_col,
                sample_size_col=sample_size_col,
                p_col=p_col,
                signed_sumstat_col=signed_sumstat_col,
                signed_sumstat_null=signed_sumstat_null,
                use_stem=use_stem,
                n_threads=n_threads,
                cache_dir=cache_dir,
            )
            gcov_vec_df = pd.read_csv(gcov_vec_path, sep="\t", index_col=0)
            gcov_vec = gcov_vec_df.loc[features, target_name].values

            # Step 2: Compute genetic covariance matrix (endophenotypes → covariates)
            logger.info("Computing endophenotype-covariate genetic covariance matrix using LDSC")
            gcov_matrix_df = _compute_nsever_gcov_matrix(
                gwas_paths=gwas_paths,
                covariate_gwas_paths=covariate_gwas_paths,
                ldsc_reference=ldsc_reference,
                ldsc_weights=ldsc_weights,
                snp_col=snp_col,
                a1_col=a1_col,
                a2_col=a2_col,
                sample_size_col=sample_size_col,
                p_col=p_col,
                signed_sumstat_col=signed_sumstat_col,
                signed_sumstat_null=signed_sumstat_null,
                use_stem=use_stem,
                n_threads=n_threads,
                cache_dir=cache_dir,
            )

            # Step 3: Build MaxGCP_R and N_SEVER, then fit
            pcov_mat = phenotypic_covariance_df.loc[features, features].values
            maxgcp_r_data = MaxGCP_R(
                endophenotype_names=features,
                target_name=target_name,
                cov_G_vec=gcov_vec,
                cov_P=pcov_mat,
            )
            nsever = N_SEVER(
                endophenotype_names=gcov_matrix_df.index.tolist(),
                covariate_names=gcov_matrix_df.columns.tolist(),
                gcov_matrix=gcov_matrix_df.values,
            )
            logger.info("Fitting N-SEVER phenotype")
            nsever_weights = fit_model(maxgcp_r_data, nsever=nsever)

            nsever_weights_df = pd.DataFrame(
                nsever_weights[:, None],
                index=pd.Index(features, name="feature"),
                columns=pd.Index([target_name], name="target"),
            )

            # Step 4: Run indirect GWAS with the N-SEVER weights
            nsever_weights_df.to_csv(gcov_matrix_path, sep="\t")
            logger.info("Computing GWAS summary statistics for the N-SEVER phenotype")
            run_indirect_gwas(
                gwas_paths=gwas_paths,
                projection_coefficient_file=gcov_matrix_path,
                phenotype_covariance_file=phenotype_covariance_file,
                n_covar=n_covar,
                output_file=output_file,
                snp_col=snp_col,
                chrom_col=chrom_col,
                pos_col=pos_col,
                a1_col=a1_col,
                a2_col=a2_col,
                beta_col=beta_col,
                std_error_col=std_error_col,
                sample_size_col=sample_size_col,
                compress=compress_output,
                use_stem=use_stem,
                chunksize=chunksize,
                n_threads=n_threads,
                in_memory=in_memory,
            )
            if not clean_up:
                logger.info(
                    "Keeping intermediate files ['nsever_genetic_covariance.tsv', "
                    "'nsever_gcov_matrix.tsv', 'nsever_weights.tsv']"
                )
                gcov_vec_path.rename(
                    output_file.parent.joinpath("nsever_genetic_covariance.tsv")
                )
                gcov_matrix_df.to_csv(
                    output_file.parent.joinpath("nsever_gcov_matrix.tsv"), sep="\t"
                )
                gcov_matrix_path.rename(
                    output_file.parent.joinpath("nsever_weights.tsv")
                )
                gcov_vec_path.touch()
                gcov_matrix_path.touch()
            else:
                logger.info("Cleaning up intermediate files")
        logger.info("Done")
    finally:
        if pcov_tmp_path is not None and pcov_tmp_path.exists():
            pcov_tmp_path.unlink()
