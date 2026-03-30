import concurrent.futures
import logging
from pathlib import Path

from rich.progress import track

from restrict_gwas.cli.ldsc import ldsc_munge, ldsc_rg

logger = logging.getLogger("rich")


_GWAS_EXTENSIONS = {
    ".gz", ".sumstats", ".tsv", ".csv", ".txt",
    ".glm", ".linear", ".logistic", ".hybrid", ".log",
}


def remove_all_suffixes(path: Path) -> Path:
    while path.suffix.lower() in _GWAS_EXTENSIONS:
        path = path.with_suffix("")
    return path


def run_munge(args: tuple[Path, Path, str, str, str, str, str, str, float, str, str]) -> Path:
    (
        gwas_path,
        output_dir,
        snp_col,
        a1_col,
        a2_col,
        sample_size_col,
        p_col,
        signed_sumstat_col,
        signed_sumstat_null,
        std_error_col,
        maf_col,
    ) = args
    output_root = output_dir.joinpath(gwas_path.name)
    output_path = output_dir.joinpath(gwas_path.name + ".sumstats.gz")
    if output_path.exists():
        logger.info(f"Skipping munge (cached): {output_path.name}")
        return output_path
    ldsc_munge(
        gwas_path,
        output_root,
        snp_col=snp_col,
        a1_col=a1_col,
        a2_col=a2_col,
        sample_size_col=sample_size_col,
        p_col=p_col,
        signed_sumstat_col=signed_sumstat_col,
        signed_sumstat_null=signed_sumstat_null,
        std_error_col=std_error_col,
        maf_col=maf_col,
    )
    return output_path


def munge_parallel(
    gwas_paths: list[Path],
    output_dir: Path,
    snp_col: str,
    a1_col: str,
    a2_col: str,
    sample_size_col: str,
    p_col: str,
    signed_sumstat_col: str,
    signed_sumstat_null: float,
    n_threads: int,
    std_error_col: str = "SE",
    maf_col: str = "FREQ",
) -> list[Path]:
    args = [
        (
            gwas_path,
            output_dir,
            snp_col,
            a1_col,
            a2_col,
            sample_size_col,
            p_col,
            signed_sumstat_col,
            signed_sumstat_null,
            std_error_col,
            maf_col,
        )
        for gwas_path in gwas_paths
    ]
    if n_threads == 1:
        return list(
            track(
                (run_munge(a) for a in args),
                total=len(args),
                description="Formatting sumstats...",
            )
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(run_munge, a): i for i, a in enumerate(args)}
        output_paths = [None] * len(args)
        for f in track(
            concurrent.futures.as_completed(futures),
            total=len(args),
            description="Formatting sumstats...",
        ):
            output_paths[futures[f]] = f.result()
    return output_paths


def run_rg(args: tuple[Path, list[Path], str, str, Path]) -> Path:
    (
        target,
        gwas_paths,
        ldsc_reference,
        ldsc_weights,
        directory,
    ) = args
    sorted_paths = [target] + [p for p in gwas_paths if p != target]
    output_stem = directory / remove_all_suffixes(target).name
    output_path = Path(str(output_stem) + ".log")
    if output_path.exists():
        logger.info(f"Skipping LDSC rg (cached): {output_path.name}")
        return output_path
    ldsc_rg(
        gwas_paths=sorted_paths,
        ldsc_reference=ldsc_reference,
        ldsc_weights=ldsc_weights,
        output_stem=output_stem,
    )
    if not output_path.exists():
        raise ValueError(f"RG output file {output_path} not found")
    return output_path


def rg_parallel(
    gwas_paths: list[Path],
    targets: list[Path],
    ldsc_reference: str,
    ldsc_weights: str,
    directory: Path,
    n_threads: int,
) -> list[Path]:
    args = [(target, gwas_paths, ldsc_reference, ldsc_weights, directory) for target in targets]
    if n_threads == 1:
        return list(
            track(
                (run_rg(a) for a in args),
                description="Computing RG...",
                total=len(args),
            )
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(run_rg, a): i for i, a in enumerate(args)}
        output_paths = [None] * len(args)
        for f in track(
            concurrent.futures.as_completed(futures),
            total=len(args),
            description="Computing genetic covariances...",
        ):
            output_paths[futures[f]] = f.result()
    return output_paths
