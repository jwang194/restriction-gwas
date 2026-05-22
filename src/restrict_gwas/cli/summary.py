"""Run-end summary table for `maxgcp_r` and `nsever`.

Writes `<output_file_stem>.summary.tsv` containing:

  * a one-record synthetic-trait block (as `# key: value` comment lines)
  * a per-feature TSV body with LDSC h2 / gcov(target) / rg(target) and
    MaxGCP_R / N-SEVER weights.

All h2 / rg values are intercept-h2-1-constrained (LDSC `--intercept-h2 1`,
hardcoded in `cli/ldsc.py`). Columns carrying that constraint are suffixed
`_ic1` to make this explicit. The gcov values and pheno_var values are not
affected by that constraint.

Inputs come from files already produced by a single run; no new LDSC calls.
The caller must pass `cache_dir`; if `None`, the writer logs a warning and
returns without writing.
"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from restrict_gwas.cli.utils import remove_all_suffixes

logger = logging.getLogger("rich")


_RE_H2 = re.compile(r"Total Observed scale h2:\s*(\S+)\s*\(\s*([^)]+)\s*\)")
_RE_GCOV = re.compile(r"Total Observed scale gencov:\s*(\S+)\s*\(\s*([^)]+)\s*\)")
_RE_RG = re.compile(r"Genetic Correlation:\s*(\S+)\s*\(\s*([^)]+)\s*\)")
_RE_PHENO = re.compile(r"Heritability of phenotype (\d+)(?:/(\d+))?")
_RE_RG_LINE = re.compile(r"^--rg\s+(.+?)(?:\s*\\)?$")


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return math.nan


def parse_rg_log(log_path: Path) -> Optional[dict]:
    """Parse one LDSC `--rg` log. Returns dict with:
        anchor: stem of phenotype 1 (str)
        anchor_h2: (h2, h2_se) for phenotype 1
        pairs: list of dicts, one per phenotype 2..N, each with keys
            stem, h2, h2_se, gencov, gencov_se, rg, rg_se
    Returns None if the log is malformed (no --rg line found).
    """
    text = Path(log_path).read_text().splitlines()

    files: Optional[list[str]] = None
    for line in text:
        m = _RE_RG_LINE.match(line.strip())
        if m:
            files = [f.strip() for f in m.group(1).split(",")]
            break
    if not files:
        return None

    stems = [remove_all_suffixes(Path(f)).name for f in files]
    n_phenos = len(stems)

    # Per-phenotype state
    h2: dict[int, tuple[float, float]] = {}
    gencov: dict[int, tuple[float, float]] = {}
    rg: dict[int, tuple[float, float]] = {}
    cur_pheno: Optional[int] = None
    in_h2 = in_gencov = in_rg = False

    for line in text:
        m = _RE_PHENO.match(line)
        if m:
            cur_pheno = int(m.group(1))
            in_h2, in_gencov, in_rg = True, False, False
            continue
        # Note: must check value lines BEFORE header lines because the
        # "Genetic Correlation: X (Y)" value starts with the same string
        # as the "Genetic Correlation" section header.
        if in_h2 and cur_pheno is not None and "Total Observed scale h2:" in line:
            m = _RE_H2.search(line)
            if m:
                h2[cur_pheno] = (_safe_float(m.group(1)), _safe_float(m.group(2)))
            in_h2 = False
            continue
        if in_gencov and cur_pheno is not None and "Total Observed scale gencov:" in line:
            m = _RE_GCOV.search(line)
            if m:
                gencov[cur_pheno] = (_safe_float(m.group(1)), _safe_float(m.group(2)))
            in_gencov = False
            continue
        if in_rg and cur_pheno is not None and "Genetic Correlation:" in line:
            m = _RE_RG.search(line)
            if m:
                rg[cur_pheno] = (_safe_float(m.group(1)), _safe_float(m.group(2)))
            in_rg = False
            continue
        # Section header detection (only after value-match attempts so that
        # the colon-bearing "Genetic Correlation:" value line above wins).
        if line.rstrip() == "Genetic Covariance":
            in_h2, in_gencov, in_rg = False, True, False
            continue
        if line.rstrip() == "Genetic Correlation":
            in_h2, in_gencov, in_rg = False, False, True
            continue

    anchor_h2 = h2.get(1, (math.nan, math.nan))
    pairs = []
    for k in range(2, n_phenos + 1):
        h, hse = h2.get(k, (math.nan, math.nan))
        gc, gcse = gencov.get(k, (math.nan, math.nan))
        r, rse = rg.get(k, (math.nan, math.nan))
        pairs.append(dict(stem=stems[k - 1], h2=h, h2_se=hse,
                          gencov=gc, gencov_se=gcse, rg=r, rg_se=rse))
    return dict(anchor=stems[0], anchor_h2=anchor_h2, pairs=pairs)


def _build_vg_matrix(rg_pcov_dir: Path, feature_stems: list[str]) -> pd.DataFrame:
    """Assemble the genetic covariance matrix among features from
    cache/rg_pcov/*.log. Diagonal = h2 of each feature; off-diagonal = pairwise
    gencov (averaged when both directions are present)."""
    sums: dict[tuple[str, str], list[float]] = {}
    diag_sums: dict[str, list[float]] = {}
    for log in rg_pcov_dir.glob("*.log"):
        parsed = parse_rg_log(log)
        if parsed is None:
            continue
        anchor = parsed["anchor"]
        if anchor in feature_stems and not math.isnan(parsed["anchor_h2"][0]):
            diag_sums.setdefault(anchor, []).append(parsed["anchor_h2"][0])
        for p in parsed["pairs"]:
            other = p["stem"]
            if anchor not in feature_stems or other not in feature_stems:
                continue
            if not math.isnan(p["gencov"]):
                sums.setdefault((anchor, other), []).append(p["gencov"])
            if not math.isnan(p["h2"]):
                diag_sums.setdefault(other, []).append(p["h2"])

    Vg = pd.DataFrame(np.nan, index=feature_stems, columns=feature_stems, dtype=float)
    for (a, b), vals in sums.items():
        Vg.at[a, b] = float(np.mean(vals))
    # Symmetrize using means of both directions when available
    Vg = Vg.where(~Vg.isna(), Vg.T)
    for a, vals in diag_sums.items():
        Vg.at[a, a] = float(np.mean(vals))
    return Vg


def write_run_summary(
    *,
    method: str,
    output_file: Path,
    target: Path,
    feature_paths: list[Path],
    weights_path: Path,
    pcov_path: Path,
    gcov_vec_path: Path,
    cache_dir: Optional[Path],
    use_stem: bool = True,
) -> None:
    """Emit `<output_file_stem>.summary.tsv` (see module docstring)."""
    if cache_dir is None:
        logger.warning("Skipping run summary: --cache-dir not provided")
        return
    rg_gcov_dir = cache_dir / "rg_gcov"
    rg_pcov_dir = cache_dir / "rg_pcov"
    if not rg_gcov_dir.exists() or not rg_pcov_dir.exists():
        logger.warning(
            f"Skipping run summary: missing {rg_gcov_dir} or {rg_pcov_dir}"
        )
        return

    target_stem = remove_all_suffixes(target).name if use_stem else target.name
    feature_stems = [
        remove_all_suffixes(p).name if use_stem else p.name for p in feature_paths
    ]

    # Per-feature stats from rg_gcov logs (target as anchor in each)
    feature_records: dict[str, dict] = {}
    target_h2 = (math.nan, math.nan)
    for log in rg_gcov_dir.glob("*.log"):
        parsed = parse_rg_log(log)
        if parsed is None or parsed["anchor"] != target_stem:
            continue
        if not math.isnan(parsed["anchor_h2"][0]):
            target_h2 = parsed["anchor_h2"]
        for p in parsed["pairs"]:
            feature_records[p["stem"]] = p

    # V_g matrix among features from rg_pcov logs
    Vg = _build_vg_matrix(rg_pcov_dir, feature_stems)

    # Per-feature N from munged sumstats
    n_median: dict[str, float] = {}
    for p, stem in zip(feature_paths, feature_stems):
        munged = cache_dir / "munged" / (p.name + ".sumstats.gz")
        if munged.exists():
            try:
                df = pd.read_csv(munged, sep="\t", usecols=["N"])
                n_median[stem] = float(df["N"].median())
            except Exception:
                n_median[stem] = math.nan
        else:
            n_median[stem] = math.nan

    # Load pcov, gcov_vec, weights
    pcov_df = pd.read_csv(pcov_path, sep=("\t" if pcov_path.suffix != ".csv" else ","), index_col=0)
    gcov_vec_df = pd.read_csv(gcov_vec_path, sep="\t", index_col=0)
    weights_df = pd.read_csv(weights_path, sep="\t", index_col=0)

    # Kept = those that survived --min-h2-z (== rows in weights file)
    kept = [s for s in weights_df.index if s in feature_stems]
    weights_col = weights_df.columns[0]
    w = weights_df.loc[kept, weights_col].to_numpy(dtype=float)

    # Synthetic block — w' M w over kept features
    pcov_kept = pcov_df.loc[kept, kept].to_numpy(dtype=float)
    pheno_var_synth = float(w @ pcov_kept @ w)
    gcov_vec_kept = gcov_vec_df.loc[kept].iloc[:, 0].to_numpy(dtype=float)
    gcov_synth_target = float(w @ gcov_vec_kept)
    Vg_kept = Vg.loc[kept, kept].to_numpy(dtype=float)
    h2_synth_ic1 = float(w @ np.nan_to_num(Vg_kept) @ w)
    h2_target_ic1 = target_h2[0]
    rg_synth_target_ic1 = (
        gcov_synth_target / math.sqrt(h2_synth_ic1 * h2_target_ic1)
        if h2_synth_ic1 > 0 and h2_target_ic1 > 0
        else math.nan
    )

    # Build path as <output_stem>.summary.tsv to avoid the maxgcp_r summary
    # being overwritten by the nsever summary when both share the parent
    # stem (e.g. out/psoriasis.maxgcp_r and out/psoriasis.nsever both
    # collapsing to out/psoriasis.summary.tsv via Path.with_suffix).
    base = output_file.name.removesuffix(".gz")
    summary_path = output_file.parent / f"{base}.summary.tsv"
    with open(summary_path, "w") as f:
        f.write(f"# method: {method}\n")
        f.write(f"# target: {target_stem}\n")
        f.write(f"# n_features_input: {len(feature_paths)}\n")
        f.write(f"# n_features_kept: {len(kept)}\n")
        f.write(f"# pheno_var_synth: {pheno_var_synth:.6g}\n")
        f.write(f"# gcov_synth_target: {gcov_synth_target:.6g}\n")
        f.write(f"# h2_synth_ic1: {h2_synth_ic1:.6g}\n")
        f.write(f"# h2_target_ic1: {h2_target_ic1:.6g}\n")
        f.write(f"# rg_synth_target_ic1: {rg_synth_target_ic1:.6g}\n")
        f.write(
            "feature\tn_median\tpheno_var\th2_ic1\th2_se_ic1\th2_z_ic1"
            "\tgcov_target\trg_target_ic1\trg_target_se_ic1\tweight\tincluded\n"
        )
        kept_set = set(kept)
        for stem in feature_stems:
            n = n_median.get(stem, math.nan)
            pv = float(pcov_df.at[stem, stem]) if stem in pcov_df.index else math.nan
            rec = feature_records.get(stem, {})
            h2 = rec.get("h2", math.nan)
            h2_se = rec.get("h2_se", math.nan)
            h2_z = h2 / h2_se if h2_se and not math.isnan(h2_se) and h2_se > 0 else math.nan
            gcov_t = rec.get("gencov", math.nan)
            rg_t = rec.get("rg", math.nan)
            rg_t_se = rec.get("rg_se", math.nan)
            weight = (
                float(weights_df.at[stem, weights_col]) if stem in weights_df.index
                else math.nan
            )
            included = stem in kept_set
            f.write(
                f"{stem}\t{n}\t{pv}\t{h2}\t{h2_se}\t{h2_z}"
                f"\t{gcov_t}\t{rg_t}\t{rg_t_se}\t{weight}\t{included}\n"
            )

    logger.info(f"Wrote run summary to {summary_path}")
