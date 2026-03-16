"""
NerVE: Core Eigenspectrum Metrics for FFN Latent Space Analysis.

Implements the four complementary metrics from the NerVE framework (Section 2.2):
  - Spectral Entropy (SE): variance uniformity / dispersion
  - Participation Ratio (PR): effective latent dimensionality
  - Eigenvalue Early Enrichment (EEE): top-heaviness of the spectrum
  - Jensen-Shannon Divergence (JS): distributional shift between pre/post activation

Also provides helper functions for covariance computation (Eq. 1) and
eigendecomposition used throughout the framework.

Reference: NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks (ICLR 2026)
"""

import torch


# ============================================================
#  Covariance & Eigendecomposition  (Section 2.1, Appendix A)
# ============================================================

def compute_covariance(x2d: torch.Tensor) -> torch.Tensor:
    """
    Compute unbiased sample covariance matrix from flattened activations.
    Implements Eq. 1: Sigma = (X - mu)^T (X - mu) / (N - 1)

    Args:
        x2d: [N, D] float tensor of flattened token embeddings
             where N = batch_size * seq_len, D = FFN hidden dimension

    Returns:
        [D, D] covariance matrix
    """
    # Convert dtype if needed but keep on original device
    if x2d.dtype == torch.float16 or x2d.dtype == torch.bfloat16:
        x2d = x2d.float()

    mu = x2d.mean(dim=0, keepdim=True)
    x_centered = x2d - mu
    N = x2d.size(0)
    cov = (x_centered.t() @ x_centered) / (N - 1 + 1e-12)  # Unbiased estimator: N-1
    return cov


def compute_sorted_eigs(cov: torch.Tensor) -> torch.Tensor:
    """
    Compute sorted eigenvalues from covariance matrix.
    Uses torch.linalg.eigvalsh (eigenvalues-only, no eigenvectors) for
    memory efficiency as described in Appendix H.2.

    Args:
        cov: [D, D] covariance matrix

    Returns:
        eigenvalues [D], sorted in descending order
    """
    # Convert dtype if needed
    if cov.dtype == torch.float16 or cov.dtype == torch.bfloat16:
        cov = cov.float()

    # Use torch.linalg.eigvalsh for symmetric matrices (covariance is symmetric)
    # This is more efficient than eigh when we don't need eigenvectors
    vals = torch.linalg.eigvalsh(cov)  # ascending order
    vals_sorted = torch.flip(vals, [0])  # Descending order
    vals_sorted = torch.clamp(vals_sorted, min=1e-12)  # Ensure positive eigenvalues with non-zero minimum

    return vals_sorted


def normalize_eigs(eigs: torch.Tensor) -> torch.Tensor:
    """
    Turn eigenvalues into a probability distribution: lambda_hat_i = lambda_i / Lambda
    where Lambda = sum(lambda_i).

    Args:
        eigs: eigenvalue tensor

    Returns:
        normalized eigenvalues (sum to 1)
    """
    lam_sum = torch.sum(eigs) + 1e-12
    return eigs / lam_sum


# ============================================================
#  Spectral Metrics  (Section 2.2)
# ============================================================

def compute_spectral_entropy(lam_norm: torch.Tensor) -> torch.Tensor:
    """
    Spectral Entropy (SE): Shannon entropy of normalized eigenvalues.
    SE = -sum( lambda_hat_i * log(lambda_hat_i) )

    Measures variance uniformity/dispersion. Range: [0, ln(D)].
    SE -> 0 indicates collapsed/low-rank representation.
    SE -> ln(D) indicates uniform variance across all dimensions.

    Args:
        lam_norm: normalized eigenvalues

    Returns:
        spectral entropy value
    """
    eps = 1e-12
    return -torch.sum(lam_norm * torch.log(lam_norm + eps))


def compute_participation_ratio(lam: torch.Tensor) -> torch.Tensor:
    """
    Participation Ratio (PR): effective dimensionality of the eigenspectrum.
    PR = (sum(lambda_i))^2 / sum(lambda_i^2)    (Eq. 2)

    Measures how many dimensions significantly hold variance. Range: [1, D].
    PR ~ 1 indicates maximal anisotropy (variance in single direction).
    PR ~ D indicates uniform variance across all dimensions.

    Args:
        lam: eigenvalues (raw, not normalized)

    Returns:
        participation ratio value
    """
    sum_ = torch.sum(lam) + 1e-12
    sum_sq = torch.sum(lam ** 2) + 1e-12
    return (sum_**2) / sum_sq


def compute_eee(lam: torch.Tensor) -> torch.Tensor:
    """
    Eigenvalue Early Enrichment (EEE): top-heaviness of eigenspectrum.
    Compares cumulative variance curve against uniform reference.  (Eq. 3)

    Measures how front-loaded variance is among top eigenvalues. Range: [0, 1).
    EEE ~ 1 indicates variance concentrated in top few directions.
    EEE ~ 0 indicates nearly uniform spectrum.

    Args:
        lam: eigenvalues (assumed to be sorted in descending order)

    Returns:
        scalar in [0, 1] as tensor
    """
    sorted_vals = lam  # assume descending
    lam_sum = torch.sum(sorted_vals) + 1e-12
    cumsums = torch.cumsum(sorted_vals, dim=0) / lam_sum

    d = len(sorted_vals)
    idx = torch.arange(1, d+1, device=sorted_vals.device, dtype=sorted_vals.dtype)
    uniform_line = idx / d
    differences = cumsums - uniform_line

    # Keep everything as tensors for efficiency
    area = torch.sum(differences)
    max_area = torch.tensor(0.5 * d, device=sorted_vals.device, dtype=sorted_vals.dtype)
    eee_value = area / (max_area + 1e-12)
    eee_value = torch.clamp(eee_value, min=0.0, max=1.0)

    return eee_value


def compute_js(lam_norm_a: torch.Tensor, lam_norm_b: torch.Tensor) -> torch.Tensor:
    """
    Jensen-Shannon Divergence (JS): distributional shift between two eigenspectra.
    JS(P_pre || P_post) = 0.5 * KL(P_pre || M) + 0.5 * KL(P_post || M)  (Eq. 4-6)
    where M = (P_pre + P_post) / 2.

    Unlike SE, PR, EEE which describe a single spectrum, JS quantifies the
    extent of distributional shift caused by FFN nonlinearity. Range: [0, ln(2)].

    Args:
        lam_norm_a: first normalized eigenvalue distribution (e.g., pre-activation)
        lam_norm_b: second normalized eigenvalue distribution (e.g., post-activation)

    Returns:
        JS divergence value
    """
    def kl_div(p, q):
        eps = 1e-12
        return torch.sum(p * torch.log((p+eps)/(q+eps)))

    # Ensure they're on the same device and have same shape
    device = lam_norm_a.device
    lam_norm_b = lam_norm_b.to(device)

    # In case the distributions have different dimensions
    d_a = lam_norm_a.shape[0]
    d_b = lam_norm_b.shape[0]

    if d_a != d_b:
        max_d = max(d_a, d_b)
        if d_a < max_d:
            pad_size = max_d - d_a
            lam_norm_a = torch.nn.functional.pad(lam_norm_a, (0, pad_size), value=0)
        if d_b < max_d:
            pad_size = max_d - d_b
            lam_norm_b = torch.nn.functional.pad(lam_norm_b, (0, pad_size), value=0)

    m = 0.5 * (lam_norm_a + lam_norm_b)
    js = 0.5 * kl_div(lam_norm_a, m) + 0.5 * kl_div(lam_norm_b, m)

    return js
