import logging
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

logger = logging.getLogger("rich")

_LDSC_DIR = Path(__file__).parent.parent / "ldsc" / "scripts"

app = typer.Typer(
    add_completion=False, context_settings={"help_option_names": ["-h", "--help"]}
)


@app.command(name="munge")
def ldsc_munge(
    gwas_path: Annotated[
        Path,
        typer.Argument(
            exists=True, help="Path to GWAS summary statistics", show_default=False
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option("--out", help="Path to output file", show_default=False),
    ],
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
    std_error_col: Annotated[
        str, typer.Option("--se", help="Name of standard error column")
    ] = "SE",
    maf_col: Annotated[
        str, typer.Option("--maf", help="Name of allele frequency column")
    ] = "FREQ",
) -> None:
    """Process a GWAS summary statistics file using LDSC."""
    cmd = [
        sys.executable,
        str(_LDSC_DIR / "munge_sumstats.py"),
        "--sumstats", gwas_path.as_posix(),
        "--out", output_file.as_posix(),
        "--snp", snp_col,
        "--a1", a1_col,
        "--a2", a2_col,
        "--N-col", sample_size_col,
        "--p", p_col,
        "--signed-sumstats", f"{signed_sumstat_col},{signed_sumstat_null}",
        "--se", std_error_col,
        "--frq", maf_col,
        "--keep-maf",
        "--keep-se",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f"munge_sumstats failed for {gwas_path.name}:\n{result.stderr}")


@app.command(name="rg")
def ldsc_rg(
    gwas_paths: Annotated[
        list[Path],
        typer.Argument(
            exists=True,
            help="Path to munged GWAS summary statistics",
        ),
    ],
    ldsc_reference: Annotated[
        str,
        typer.Option("--ldsc-reference", help="Path to LDSC reference LD scores (--ref-ld-chr)"),
    ],
    ldsc_weights: Annotated[
        str,
        typer.Option("--ldsc-weights", help="Path to LDSC regression weights (--w-ld-chr)"),
    ],
    output_stem: Annotated[
        Path,
        typer.Option("--out", help="Path to output file"),
    ],
) -> None:
    """Compute genetic covariances using LDSC."""
    cmd = [
        sys.executable,
        str(_LDSC_DIR / "ldsc.py"),
        "--rg", ",".join(p.as_posix() for p in gwas_paths),
        "--ref-ld-chr", ldsc_reference,
        "--w-ld-chr", ldsc_weights,
        "--out", output_stem.as_posix(),
        "--intercept-h2", ",".join(["1"] * len(gwas_paths)),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f"ldsc --rg failed for {output_stem.name}:\n{result.stderr}")
