from dataclasses import dataclass, field
from typing import Optional, TypeAlias

import numpy as np  # type: ignore
import scipy.linalg  # type: ignore
from numpy.typing import NDArray  # type: ignore

Vector: TypeAlias = NDArray
Matrix: TypeAlias = NDArray


@dataclass
class N_SEVER:
    """Null-space projection for removing contaminant genetic signal.

    Computes and stores the null space basis from genetic covariances between
    endophenotypes and covariates, following the approach in maxc_minc.py: the
    null space of gcov_matrix.T defines the subspace of endophenotype
    combinations orthogonal to all covariates.

    Args:
        endophenotype_names: Names of the endophenotypes (features).
        covariate_names: Names of the covariates/contaminants.
        gcov_matrix: Genetic covariance matrix between endophenotypes and
            covariates, shape (n_endophenotypes, n_covariates).

    Attributes:
        null_basis: Orthonormal basis Z for the null space of gcov_matrix.T,
            shape (n_endophenotypes, n_endophenotypes - n_covariates).
    """

    endophenotype_names: list[str]
    covariate_names: list[str]
    gcov_matrix: NDArray  # (n_endophenotypes, n_covariates)
    null_basis: NDArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.gcov_matrix = np.asarray(self.gcov_matrix)
        n_endo = len(self.endophenotype_names)
        n_cov = len(self.covariate_names)
        if self.gcov_matrix.shape != (n_endo, n_cov):
            raise ValueError(
                f"gcov_matrix shape {self.gcov_matrix.shape} does not match "
                f"(n_endophenotypes={n_endo}, n_covariates={n_cov})"
            )
        self.null_basis = scipy.linalg.null_space(self.gcov_matrix.T)


@dataclass
class MaxGCP_R:
    """Data container for MaxGCP regression inputs.

    Stores the names and covariance data needed to fit a MaxGCP phenotype
    towards a given target. Intended to be passed directly to fit_coheritability,
    enabling name-based alignment with N_SEVER.

    Args:
        endophenotype_names: Names of the endophenotypes (features).
        target_name: Name of the target trait.
        cov_G_vec: Genetic covariance vector between endophenotypes and the
            target, shape (n_endophenotypes,).
        cov_P: Partial phenotypic variance-covariance matrix among
            endophenotypes, shape (n_endophenotypes, n_endophenotypes).
    """

    endophenotype_names: list[str]
    target_name: str
    cov_G_vec: NDArray  # (n_endophenotypes,)
    cov_P: NDArray  # (n_endophenotypes, n_endophenotypes)

    def __post_init__(self) -> None:
        self.cov_G_vec = np.asarray(self.cov_G_vec)
        self.cov_P = np.asarray(self.cov_P)
        n = len(self.endophenotype_names)
        if self.cov_G_vec.shape != (n,):
            raise ValueError(
                f"cov_G_vec shape {self.cov_G_vec.shape} does not match ({n},)"
            )
        if self.cov_P.shape != (n, n):
            raise ValueError(
                f"cov_P shape {self.cov_P.shape} does not match ({n}, {n})"
            )


def check_matrix(mat: Matrix):
    """Check that the input is a valid covariance matrix"""
    if mat.ndim != 2:
        raise ValueError("Input must be 2D")

    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Input must be square")

    if not np.allclose(mat, mat.T, atol=1e-5):
        raise ValueError("Input must be symmetric")


def check_matrix_inputs(cov_G: Matrix, cov_P: Matrix) -> None:
    """Check that the inputs are valid covariance matrices"""
    check_matrix(cov_G)
    check_matrix(cov_P)

    if (cov_G.shape[0] != cov_G.shape[1]) or (cov_P.shape[0] != cov_P.shape[1]):
        raise ValueError("Covariance matrices must be square")


def check_matrix_vector_inputs(cov_G: Vector, cov_P: Matrix) -> None:
    """Check that the inputs are a valid coheritability optimization problem"""
    check_matrix(cov_P)
    if cov_G.ndim != 1:
        raise ValueError("Input must be 1D")

    if cov_G.shape[0] != cov_P.shape[0]:
        raise ValueError("Vector must have the same length as the matrix")


def fit_heritability(cov_G: Matrix, cov_P: Matrix) -> Matrix:
    """Fit phenotypes to maximize heritability.

    Given n input phenotypes, this fits n linearly independent phenotypes that
    maximize heritability. This is akin to fitting principal components, but
    using heritability instead of overall variance as the criterion.

    Args:
        cov_G: Genetic covariance matrix (features x features)
        cov_P : Phenotypic covariance matrix (features x features)

    Returns:
        Weights that defines the phenotypes (features x features)

    Raises:
        ValueError: If the input matrices are not square or symmetric.
    """
    cov_G = np.asarray(cov_G)
    cov_P = np.asarray(cov_P)
    check_matrix_inputs(cov_G, cov_P)

    cov_G_sqrt: NDArray = scipy.linalg.sqrtm(cov_G)  # type: ignore
    if np.iscomplexobj(cov_G_sqrt):
        raise ValueError("Input covariance matrix must be real")
    lhs = cov_G_sqrt @ np.linalg.inv(cov_P) @ cov_G_sqrt
    if np.iscomplexobj(lhs):
        raise ValueError("LHS matrix must be real")
    if not np.allclose(lhs, lhs.T, atol=1e-5):
        raise ValueError("LHS matrix must be symmetric")
    _, evecs = np.linalg.eigh(lhs)
    if np.iscomplexobj(evecs):
        raise ValueError("Eigenvectors must be real")
    weights = np.linalg.inv(cov_G_sqrt) @ evecs
    if np.iscomplexobj(weights):
        raise ValueError("Eigenvectors must be real")
    weights = np.asarray(weights)

    # Normalize weights so that projections have unit variance
    weights = weights / np.sqrt(np.diag(weights.T @ cov_P @ weights))
    return weights


def fit_coheritability(
    data: MaxGCP_R,
    nsever: Optional[N_SEVER] = None,
) -> Vector:
    """Fit a MaxGCP phenotype to the genetic and phenotypic covariances.

    When an N_SEVER object is additionally provided, the fit is performed in
    the null space of the contaminant genetic covariances, restricted to the
    shared endophenotypes between the two objects (matched by name). The null
    space basis is recomputed from the subsetted gcov_matrix so that the
    projection correctly reflects the available features. The returned weights
    span the full endophenotype space of MaxGCP_R (zeros for endophenotypes
    not shared with N_SEVER).

    Args:
        data: MaxGCP_R object containing endophenotype names, target name,
            genetic covariance vector, and partial phenotypic covariance matrix.
        nsever: Optional N_SEVER object for null-space confounder projection.
            Shared endophenotypes are identified by name.

    Returns:
        A weight vector that defines the MaxGCP phenotype, shape
        (n_endophenotypes,) in data.endophenotype_names order.

    Raises:
        ValueError: If no shared endophenotypes exist between data and nsever,
            or if the projected variance is non-positive.
    """
    if nsever is not None:
        # Name-based subsetting to shared endophenotypes
        shared = [
            n for n in data.endophenotype_names if n in nsever.endophenotype_names
        ]
        if not shared:
            raise ValueError("No shared endophenotypes between MaxGCP_R and N_SEVER")
        r_idx = [data.endophenotype_names.index(n) for n in shared]
        s_idx = [nsever.endophenotype_names.index(n) for n in shared]

        cov_G_sub = data.cov_G_vec[r_idx]
        cov_P_sub = data.cov_P[np.ix_(r_idx, r_idx)]
        gcov_contam_sub = nsever.gcov_matrix[s_idx, :]

        # Recompute null basis for the subsetted endophenotype set
        Z = scipy.linalg.null_space(gcov_contam_sub.T)
        cov_G_proj = Z.T @ cov_G_sub
        cov_P_proj = Z.T @ cov_P_sub @ Z

        inner_weights, _, _, _ = np.linalg.lstsq(cov_P_proj, cov_G_proj, rcond=None)
        inner_weights = np.asarray(inner_weights)

        # Map back to shared feature space and normalize
        weights_shared = Z @ inner_weights
        var = weights_shared.T @ cov_P_sub @ weights_shared
        if var <= 0:
            raise ValueError(
                "Projection has zero or negative variance after N-SEVER projection"
            )
        weights_shared = weights_shared / np.sqrt(var)

        # Embed into full MaxGCP_R endophenotype space (zeros for non-shared)
        weights_full = np.zeros(len(data.endophenotype_names))
        for i, r_i in enumerate(r_idx):
            weights_full[r_i] = weights_shared[i]
        return weights_full

    weights, _, _, _ = np.linalg.lstsq(data.cov_P, data.cov_G_vec, rcond=None)
    weights = np.asarray(weights)
    var = weights.T @ data.cov_P @ weights
    return weights / np.sqrt(var)


def fit_all_coheritability(cov_G: Matrix, cov_P: Matrix) -> Matrix:
    """Fit a MaxGCP phenotype to every input phenotype.

    This is equivalent to treating each phenotype as the target in a MaxGCP
    regression, and fitting the corresponding weights.

    Args:
        cov_G: Matrix of genetic covariances (features x features)
        cov_P: Matrix of phenotypic covariances (features x features)

    Returns:
        A weight matrix that define the MaxGCP phenotypes (features x features).

    Raises:
        ValueError: If the input matrices are not square or symmetric.
    """
    cov_G = np.asarray(cov_G)
    cov_P = np.asarray(cov_P)
    check_matrix_inputs(cov_G, cov_P)

    weights, _, _, _ = np.linalg.lstsq(cov_P, cov_G, rcond=None)
    weights = np.asarray(weights)

    # Normalize weights so that projections have unit variance
    weights = weights / np.sqrt(np.diag(weights.T @ cov_P @ weights))
    return weights


def fit_genetic_correlation(phenotype_idx: int, cov_G: Matrix) -> Vector:
    """Fit a phenotype that is maximally genetically correlated with the
    specified feature phenotype.

    Unlike the other methods, this does not let the target be included as a
    feature in the resulting phenotype.

    Args:
        phenotype_idx: Index of the target phenotype
        cov_G: Genetic covariance matrix (features x features)

    Returns:
        A weight vector that defines the optimized phenotype (features x 1)

    Raises:
        ValueError: If the input matrix is not square or symmetric.
    """
    cov_G = np.asarray(cov_G)
    check_matrix(cov_G)

    if cov_G.ndim != 2:
        raise ValueError("Covariance matrix must be 2D")

    if cov_G.shape[0] != cov_G.shape[1]:
        raise ValueError("Covariance matrix must be square")

    v = np.delete(cov_G[phenotype_idx], phenotype_idx, 0)
    G = np.delete(np.delete(cov_G, phenotype_idx, 0), phenotype_idx, 1)

    weights, _, _, _ = np.linalg.lstsq(G, v)

    # Normalize weights so that projection has unit variance
    var = weights.T @ G @ weights
    weights = weights / np.sqrt(var)
    return np.asarray(weights)
