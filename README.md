# restrict_gwas

Restricted GWAS. Optimized phenotype definitions boost GWAS power.

`restrict_gwas` is a Python package that implements maximum genetic component phenotyping (MaxGCP) with optional N-SEVER null-space projection to remove contaminant genetic signal from covariates.
It optimizes a linear phenotype definition to maximize its heritability and genetic correlation with a trait of interest, while optionally severing shared genetic components with specified covariates.
The resulting phenotype definition can be used to enhance statistical power in genome-wide association studies (GWAS).
`restrict_gwas` requires only estimates of genetic and phenotypic covariances, which can be obtained from GWAS summary statistics.

## Usage

```python
from restrict_gwas.estimators import MaxGCP_R, N_SEVER, fit_coheritability
import numpy as np

# Define the MaxGCP_R data object
data = MaxGCP_R(
    endophenotype_names=["pheno1", "pheno2", "pheno3"],
    target_name="target",
    cov_G_vec=np.array([0.5, 0.75, 0.25]),
    cov_P=np.array([
        [0.5, 0.25, 0.1],
        [0.25, 0.3, 0.05],
        [0.1, 0.05, 0.6],
    ]),
)

# Compute the MaxGCP_R weights
w = fit_coheritability(data)

# Optionally, sever shared genetic signal with covariates using N-SEVER
sever = N_SEVER(
    endophenotype_names=["pheno1", "pheno2", "pheno3"],
    covariate_names=["covar1"],
    gcov_matrix=np.array([[0.3], [0.1], [0.2]]),  # (n_endophenotypes, n_covariates)
)
w_severed = fit_coheritability(data, nsever=sever)
```

## Command-line usage

`restrict_gwas` exposes the `restrict_gwas` command in the command line.
Once installed, run `restrict_gwas --help` to see all available commands.

### MaxGCP_R (end-to-end run)

Runs MaxGCP with optional N-SEVER using GWAS summary statistics:

```bash
restrict_gwas maxgcp_r \
  --pcov phenotypic_covariance_matrix.csv \  # Can be computed using restrict_gwas pcov
  --tagfile ld_ref_panel/eur_w_ld_chr \      # LDSC tagfiles
  --target E11 \                             # Target phenotype (stem of GWAS file)
  --n-covar 12 \                             # Covariates used in GWAS (e.g. age+sex+10 PCs)
  --no-compress-output \
  --out E11.maxgcp_r.tsv \
  E11.glm.linear I10.glm.linear gwas/*.glm.linear
```

Alternatively, estimate the phenotypic covariance matrix from summary statistics (no individual-level data needed):

```bash
restrict_gwas maxgcp_r \
  --sumstats-only \
  --sample-overlap sample_overlap.csv \      # Pairwise sample overlap table
  --tagfile ld_ref_panel/eur_w_ld_chr \
  --target E11 \
  --n-covar 12 \
  --out E11.maxgcp_r.tsv \
  E11.glm.linear I10.glm.linear gwas/*.glm.linear
```

### N-SEVER (covariate-restricted run)

Extends MaxGCP_R by severing shared genetic signal with covariate phenotypes:

```bash
restrict_gwas nsever \
  --pcov phenotypic_covariance_matrix.csv \
  --tagfile ld_ref_panel/eur_w_ld_chr \
  --target E11 \
  --n-covar 12 \
  --covar-gwas covar1.glm.linear \           # Repeatable: one per covariate
  --covar-gwas covar2.glm.linear \
  --out E11.nsever.tsv \
  E11.glm.linear I10.glm.linear gwas/*.glm.linear
```

### Other commands

```bash
restrict_gwas pcov --help       # Compute phenotypic covariance from individual-level data
restrict_gwas gcov-vec --help   # Compute genetic covariance vector via LDSC
restrict_gwas igwas --help      # Run indirect GWAS given weights
restrict_gwas ldsc --help       # LDSC subcommands (munge, rg)
restrict_gwas compare --help    # Compare alternative methods (MaxH, PCA)
```

To run end-to-end, you'll need LDSC tagfiles.
These can be obtained from [the MTAG repository](https://github.com/JonJala/mtag/tree/9e17f3cf1fbcf57b6bc466daefdc51fd0de3c5dc/ld_ref_panel).
For more information about LD score regression, see [the LDSC repository](https://github.com/bulik/ldsc) and [its wiki](https://github.com/bulik/ldsc/wiki).

## Installation

```bash
pip install restrict_gwas
```

Please see this repository's [pyproject.toml](pyproject.toml) for a full list of dependencies.
