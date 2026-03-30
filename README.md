# restrict_gwas

Restriction GWAS. Targeted imputation of particular pathways underlying complex traits concentrates power and improves interpretability.

`restriction_gwas` is a Python package that implements two methods, $\text{MaxGCP}\_\text{R}$ and $\text{N-SEVER}$, for defining new phenotypes that closely mimic pathways underlying complex traits. 

$\text{MaxGCP}\_\text{R}$ imputes target traits using a number of endophenotypes; selecting the endophenotypes according to biological priors can restrict the imputation to particular pathways e.g. using neuroimaging phenotypes to impute type 2 diabetes (T2D) status concentrates downstream T2D genetic associations in behavioral risk pathways. 

$\text{N-SEVER}$ extends $\mathrm{MaxGCP}\_\mathrm{R}$, further concentrating association power by controlling for heritable covariates without collider bias e.g. controlling for BMI in a blood biochemistry imputation of T2D depletes for insulin resistance associations and enriches for insulin deficiency associations. 

`restrict_gwas` requires only estimates of genetic and phenotypic covariances, which can be obtained from GWAS summary statistics.

## Usage

```python
from restrict_gwas.estimators import MaxGCP_R, N_SEVER, fit_coheritability
import numpy as np

# Define the MaxGCP_R data object
mgcp = MaxGCP_R(
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
w = fit_model(mgcp)

# Sever shared genetic signal with covariates using N-SEVER
nsvr = N_SEVER(
    endophenotype_names=["pheno1", "pheno2", "pheno3"],
    covariate_names=["covar1"],
    gcov_matrix=np.array([[0.3], [0.1], [0.2]]),  # (n_endophenotypes, n_covariates)
)
w_severed = fit_model(mgcp, nsever=nsvr)
```

## Command-line usage

`restriction_gwas` exposes the `restrict_gwas` command in the command line.
Once installed, run `restrict_gwas --help` to see all available commands.

### $\text{MaxGCP}\_\text{R}$ (end-to-end run)

Runs $\text{MaxGCP}\_\text{R}$ using GWAS summary statistics and a precomputed partial (e.g. conditioned on endophenotypes' GWAS covariates) phenotypic variance-covariance matrix for the endophenotypes:

```bash
restrict_gwas maxgcp_r \
  --pcov phenotypic_covariance_matrix.csv \  # Can be computed using restrict_gwas pcov
  --ldsc-reference ld_ref_panel/eur_w_ld_chr \
  --ldsc-weights ld_ref_panel/eur_w_ld_chr \      # LDSC tagfiles
  --target E11 \                             # Target phenotype (stem of GWAS file)
  --n-covar 12 \                             # Covariates used in endophenotype GWAS (e.g. age+sex+10 PCs)
  --no-compress-output \
  --out E11.maxgcp_r.tsv \
  E11.glm.linear I10.glm.linear gwas/*.glm.linear # GWAS summary statistics for all endophenotypes and the target
```

Alternatively, estimate the endophenotypes' phenotypic variance-covariance matrix from summary statistics (no individual-level data needed):

```bash
restrict_gwas maxgcp_r \
  --sumstats-only \
  --sample-overlap sample_overlap.csv \      # Pairwise sample overlap table, exclude only if there is complete sample overlap between all input endophenotypes
  --ldsc-reference ld_ref_panel/eur_w_ld_chr \
  --ldsc-weights ld_ref_panel/eur_w_ld_chr \
  --target E11 \
  --n-covar 12 \
  --out E11.maxgcp_r.tsv \
  E11.glm.linear I10.glm.linear gwas/*.glm.linear
```

### $\text{N-SEVER}$ (covariate-restricted run)

Runs $\text{N-SEVER}$ using GWAS summary statistics:

```bash
restrict_gwas nsever \
  --pcov phenotypic_covariance_matrix.csv \
  --ldsc-reference ld_ref_panel/eur_w_ld_chr \
  --ldsc-weights ld_ref_panel/eur_w_ld_chr \
  --target E11 \
  --n-covar 12 \
  --covar-gwas covar1.glm.linear \           # Repeatable: one per heritable covariate for which N-SEVER should control
  --covar-gwas covar2.glm.linear \
  --out E11.nsever.tsv \
  E11.glm.linear I10.glm.linear gwas/*.glm.linear
```
$\text{N-SEVER}$ can be run without a precomputed $\texttt{pcov}$ matrix just as $\text{MaxGCP}\_\text{R}$ can.

### Other commands

```bash
restrict_gwas pcov --help       # Compute phenotypic covariance from individual-level data
restrict_gwas ldsc --help       # LDSC subcommands (munge, rg)
```

To run end-to-end, you'll need LDSC tagfiles.
These can be obtained from [the MTAG repository](https://github.com/JonJala/mtag/tree/9e17f3cf1fbcf57b6bc466daefdc51fd0de3c5dc/ld_ref_panel).
For more information about LD score regression, see [the LDSC repository](https://github.com/bulik/ldsc) and [its wiki](https://github.com/bulik/ldsc/wiki).

## Installation

```bash
pip install restrict_gwas
```

Please see this repository's [pyproject.toml](pyproject.toml) for a full list of dependencies.
