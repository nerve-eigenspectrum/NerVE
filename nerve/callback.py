"""
NerVE: Training-time FFN Eigenspectrum Monitoring Callback.

Implements the online, memory-efficient eigenspectrum tracking described in
Section 2.1, Appendix A, and Appendix H of the paper:
  - Hook registration on FFN up-projection (pre-activation) and 
    down-projection input (post-activation)
  - Sequential layer processing with memory cleanup (Algorithm 1)
  - Paired measurement: same tokens for pre/post activations
  - Hybrid CPU/GPU storage strategy
  - Optional token sub-sampling (Appendix G)

Usage with HuggingFace Trainer:
    from nerve.callback import FFNEigenMetricsCallback, register_ffn_hooks

    callback = FFNEigenMetricsCallback(log_steps=200, num_layers=12)
    trainer.add_callback(callback)
    register_ffn_hooks(model, callback)
    trainer.train()

Reference: NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks (ICLR 2026)
"""

import torch
import os
import gc
from transformers import TrainerCallback
from torch.utils.hooks import RemovableHandle
from typing import Dict, List, Tuple, Optional

from .metrics import (
    compute_covariance, compute_sorted_eigs, normalize_eigs,
    compute_spectral_entropy, compute_participation_ratio,
    compute_eee, compute_js
)


# ============================================================
#  Activation Accumulator (from eigen_metrics.py)
# ============================================================

class FFNEigenMetricsHelper:
    """
    Helper to store pre/post activation across batches.
    Optimized for memory efficiency using CPU storage.
    """
    def __init__(self, device: str = "cpu"):
        """
        Initialize the helper.

        Args:
            device: device to store tensors on (default: cpu for memory efficiency)
        """
        self.device = device
        self.layer_acts: Dict[int, List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]] = {}

    def add_activations(self,
                        layer_idx: int,
                        pre_act: Optional[torch.Tensor] = None,
                        post_act: Optional[torch.Tensor] = None):
        """
        Add pre/post activations for a layer.

        Args:
            layer_idx: index of the layer
            pre_act: pre-activation tensor or None
            post_act: post-activation tensor or None
        """
        if layer_idx not in self.layer_acts:
            self.layer_acts[layer_idx] = []

        if pre_act is not None:
            pre_act = pre_act.detach().to(self.device)
        if post_act is not None:
            post_act = post_act.detach().to(self.device)

        self.layer_acts[layer_idx].append((pre_act, post_act))

    def get_accumulated(self) -> Dict[int, List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]]:
        """
        Return the dictionary of layer -> list of (pre, post), and then clear it.

        Returns:
            Dictionary mapping layer indices to lists of activation pairs
        """
        data = self.layer_acts
        self.layer_acts = {}
        return data

    def clear(self):
        """Explicitly clear all stored activations."""
        self.layer_acts.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
#  Hook Registration (Appendix A.2)
# ============================================================

def register_ffn_hooks(model, callback_obj):
    """
    Attach forward hooks on each GPT2Block's MLP to capture pre/post activation.

    Hook placement (Appendix A.2):
      - Forward hook on c_fc (up-projection): captures PreAct = W_up * x + b
      - Pre-forward hook on c_proj (down-projection): captures PostAct = sigma(W_up * x + b)

    For gated architectures (SwiGLU/GeGLU):
      - Forward hook on c_fc captures PreAct = W_gate * x
      - Pre-forward hook on c_proj captures PostAct = sigma(W_gate * x) * (W_up * x)

    Uses the model's training flag for efficient conditional processing:
    hooks are no-ops on non-logging steps.

    Args:
        model: GPT-2 model with transformer.h blocks
        callback_obj: FFNEigenMetricsCallback instance to receive activations
    """
    num_layers = len(model.transformer.h)
    callback_obj.hook_handles = []

    for layer_idx in range(num_layers):
        block = model.transformer.h[layer_idx]
        fc_module = block.mlp.c_fc
        proj_module = block.mlp.c_proj

        def get_fc_forward_hook(idx):
            def fc_forward_hook(mod, inp, out):
                # Only collect during training at logging steps - fast path for most steps
                if not mod.training or not callback_obj.collecting_activations:
                    return
                callback_obj.capture_pre_acts(idx, out)
            return fc_forward_hook

        def get_proj_pre_forward_hook(idx):
            def proj_pre_forward_hook(mod, inp):
                # Only collect during training at logging steps - fast path for most steps
                if not mod.training or not callback_obj.collecting_activations:
                    return
                # inp is a tuple with the first element being the tensor
                callback_obj.capture_post_acts(idx, inp[0])
            return proj_pre_forward_hook

        h1 = fc_module.register_forward_hook(get_fc_forward_hook(layer_idx))
        h2 = proj_module.register_forward_pre_hook(get_proj_pre_forward_hook(layer_idx))
        callback_obj.hook_handles.extend([h1, h2])


# ============================================================
#  Training Callback (Algorithm 1, Appendix H)
# ============================================================

class FFNEigenMetricsCallback(TrainerCallback):
    """
    HuggingFace Trainer callback for online eigenspectrum monitoring of FFN activations.

    Features (as described in Appendix A.2 and H):
      - Memory-efficient: CPU storage with GPU computation (hybrid strategy)
      - Sequential layer processing with cleanup (Algorithm 1)
      - Paired measurement: same token indices for pre/post activations
      - Optional token sub-sampling (Appendix G)
      - Logging to per-layer text files

    Args:
        log_steps: compute metrics every N training steps
        device: device for eigendecomposition ('cuda' or 'cpu')
        output_dir: directory for metric log files
        num_layers: number of transformer layers
        do_sampling: whether to sub-sample tokens (Appendix G)
        sample_ratio: fraction of tokens to use if sampling
        batch_processing: whether to process activations in batches
        max_batch_size: maximum tokens per batch during processing
    """

    def __init__(self,
                 log_steps=100,
                 device=None,
                 output_dir="eigen_metrics_logs",
                 num_layers=12,
                 do_sampling=True,
                 sample_ratio=0.05,
                 batch_processing=True,
                 max_batch_size=10000):
        super().__init__()
        self.log_steps = log_steps

        # Device configuration
        self.compute_device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.storage_device = 'cpu'  # Always store on CPU to save GPU memory

        # State tracking
        self.current_step = 0
        self.last_logged_step = -1
        self.num_layers = num_layers
        self.collecting_activations = False

        # Sampling configuration
        self.do_sampling = do_sampling
        self.sample_ratio = sample_ratio
        self.batch_processing = batch_processing
        self.max_batch_size = max_batch_size

        # Storage for activations
        self.layer_pre_acts: Dict[int, List[torch.Tensor]] = {}
        self.layer_post_acts: Dict[int, List[torch.Tensor]] = {}
        self.sample_indices: Dict[int, torch.Tensor] = {}
        self.hook_handles: List[RemovableHandle] = []

        # Setup logging
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.log_file_path = f"{output_dir}/eigen_metrics.log"
        self.log_file = open(self.log_file_path, "a")

        # Layer metric files
        self.layer_files = [f"{output_dir}/layer_{i}_eigen.txt" for i in range(num_layers)]

        # Initialization logging
        self.log_file.write(f"NerVE callback initialized\n")
        self.log_file.write(f"Log frequency: {log_steps}, Compute device: {self.compute_device}\n")
        self.log_file.write(f"Sampling: {do_sampling}, Ratio: {sample_ratio}, Batch processing: {batch_processing}\n")
        self.log_file.flush()

    def should_log_step(self, step):
        """Fast check if current step should be logged."""
        return step == 1 or step % self.log_steps == 0

    # ----------------------------------------------------------
    #  Trainer lifecycle hooks
    # ----------------------------------------------------------

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Register hooks once at the start of training."""
        if model is not None and len(self.hook_handles) == 0:
            register_ffn_hooks(model, self)
            self.collecting_activations = self.should_log_step(1)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        """Flip activation collection on/off based on logging schedule."""
        self.current_step = state.global_step

        should_log = self.should_log_step(self.current_step)
        if should_log != self.collecting_activations:
            self.collecting_activations = should_log

            if should_log:
                # Clear storage when we start collecting
                self.layer_pre_acts = {}
                self.layer_post_acts = {}
                self.sample_indices = {}

        return control

    def on_step_end(self, args, state, control, **kwargs):
        """Process metrics at logging steps only."""
        step = state.global_step

        # Skip if not a logging step or already processed
        if not self.should_log_step(step) or step == self.last_logged_step:
            return control

        # Process metrics if we have data
        if self.layer_pre_acts and self.layer_post_acts:
            self._compute_metrics(step)

        # Stop collecting and update state
        self.collecting_activations = False
        self.last_logged_step = step

        # Clear memory
        self.layer_pre_acts = {}
        self.layer_post_acts = {}
        self.sample_indices = {}

        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Clean up at the end of training."""
        step = state.global_step
        if self.should_log_step(step) and step != self.last_logged_step and self.layer_pre_acts:
            self._compute_metrics(step)

        # Clean up resources
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

        self.log_file.close()
        return control

    # ----------------------------------------------------------
    #  Activation capture (paired measurement, Appendix A.2)
    # ----------------------------------------------------------

    def capture_pre_acts(self, layer_idx: int, tensor: torch.Tensor):
        """
        Capture pre-activations. Generates sampling indices once per layer
        per step; these same indices are reused in capture_post_acts to
        guarantee paired measurement.
        """
        if layer_idx not in self.layer_pre_acts:
            self.layer_pre_acts[layer_idx] = []

            # Generate sampling indices once per layer per step
            if self.do_sampling:
                batch_size = tensor.shape[0]
                if self.sample_ratio < 1.0:
                    num_samples = max(1, int(batch_size * self.sample_ratio))
                    self.sample_indices[layer_idx] = torch.randperm(batch_size)[:num_samples]
                else:
                    self.sample_indices[layer_idx] = None  # Flag to use entire tensor

        # Apply sampling if configured
        if self.do_sampling and layer_idx in self.sample_indices:
            indices = self.sample_indices[layer_idx]
            if indices is not None:
                tensor = tensor[indices]

        # Precision control and CPU storage
        if tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
            tensor = tensor.float()

        self.layer_pre_acts[layer_idx].append(tensor.detach().to(self.storage_device))

    def capture_post_acts(self, layer_idx: int, tensor: torch.Tensor):
        """
        Capture post-activations using the SAME sampling indices as
        pre-activations (paired measurement guarantee).
        """
        if layer_idx not in self.layer_post_acts:
            self.layer_post_acts[layer_idx] = []

        # Apply the SAME sampling as pre-activations for consistency
        if self.do_sampling and layer_idx in self.sample_indices:
            indices = self.sample_indices[layer_idx]
            if indices is not None:
                tensor = tensor[indices]

        # Precision control and CPU storage
        if tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
            tensor = tensor.float()

        self.layer_post_acts[layer_idx].append(tensor.detach().to(self.storage_device))

    # ----------------------------------------------------------
    #  Metric computation (Algorithm 1)
    # ----------------------------------------------------------

    def _compute_metrics(self, step):
        """
        Compute eigenvalue metrics for all layers.
        Processes layers sequentially with memory cleanup between each
        (Algorithm 1, Appendix H.2).
        """
        try:
            for layer_idx in sorted(self.layer_pre_acts.keys()):
                if layer_idx not in self.layer_post_acts:
                    continue

                if not self.layer_pre_acts[layer_idx] or not self.layer_post_acts[layer_idx]:
                    continue

                try:
                    # Process pre-activations
                    pre_acts = self.layer_pre_acts[layer_idx]
                    pre_metrics = self._process_activations(pre_acts)

                    # Process post-activations
                    post_acts = self.layer_post_acts[layer_idx]
                    post_metrics = self._process_activations(post_acts)

                    # Compute JS divergence between pre and post spectra
                    js_val = compute_js(pre_metrics['lam_norm'], post_metrics['lam_norm']).item()

                    # Write to layer-specific file
                    with open(self.layer_files[layer_idx], 'a') as f:
                        f.write(f"Step {step}: "
                                f"SE_pre={pre_metrics['se']:.3f}, SE_post={post_metrics['se']:.3f}, "
                                f"PR_pre={pre_metrics['pr']:.2f}, PR_post={post_metrics['pr']:.2f}, "
                                f"EEE_pre={pre_metrics['eee']:.3f}, EEE_post={post_metrics['eee']:.3f}, "
                                f"JS={js_val:.4f}\n")

                    del pre_metrics, post_metrics

                except Exception as e:
                    self.log_file.write(f"Error processing layer {layer_idx}: {str(e)}\n")

                # Clear activations for this layer and clean up (Algorithm 1: line 5)
                self.layer_pre_acts[layer_idx].clear()
                self.layer_post_acts[layer_idx].clear()

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.log_file.write(f"Completed metrics computation for step {step}\n")
            self.log_file.flush()

        except Exception as e:
            self.log_file.write(f"Error in metrics computation at step {step}: {str(e)}\n")
            self.log_file.flush()

    def _process_activations(self, acts_list):
        """
        Process a list of activation tensors to compute eigenspectrum metrics.

        Handles two paths:
          1. Fast path: single tensor, compute covariance directly
          2. Batch path: multiple tensors, combine using parallel covariance formula

        In both cases, tensors are flattened from [B, S, D] -> [N, D] as
        described in Section 2.1 (discarding sequence order).

        Args:
            acts_list: list of activation tensors

        Returns:
            dict with keys: lam, lam_norm, se, pr, eee
        """
        sample = acts_list[0]
        dim = sample.shape[-1]  # Last dimension is always feature dimension

        # Fast path for small data: process directly
        if len(acts_list) == 1 and not self.batch_processing:
            tensor = acts_list[0]
            if tensor.dim() == 3:  # [B, S, D]
                tensor = tensor.reshape(-1, dim)

            tensor = tensor.to(self.compute_device)

            # Optional additional sampling for large tensors
            if self.do_sampling and tensor.size(0) > self.max_batch_size:
                idx = torch.randperm(tensor.size(0), device=tensor.device)[:self.max_batch_size]
                tensor = tensor[idx]

            cov = compute_covariance(tensor)

            # Log covariance matrix size
            cov_size = cov.shape
            cov_mem_mb = (cov.element_size() * cov.nelement()) / (1024 * 1024)
            self.log_file.write(f"Single-batch covariance matrix size: {cov_size}, Memory: {cov_mem_mb:.2f} MB\n")
            self.log_file.flush()

            del tensor

        # Batch processing path: combine covariances using parallel formula
        else:
            batch_covs = []
            batch_means = []
            batch_sizes = []

            for i in range(0, len(acts_list), max(1, len(acts_list) // 4)):
                batch = acts_list[i:i+max(1, len(acts_list) // 4)]

                if batch[0].dim() == 3:  # [B, S, D]
                    tensor = torch.cat([x.reshape(-1, dim) for x in batch], dim=0)
                else:
                    tensor = torch.cat(batch, dim=0)

                tensor = tensor.to(self.compute_device)

                if self.do_sampling and tensor.size(0) > self.max_batch_size:
                    idx = torch.randperm(tensor.size(0), device=tensor.device)[:self.max_batch_size]
                    tensor = tensor[idx]

                N = tensor.size(0)
                batch_mean = tensor.mean(dim=0)
                centered = tensor - batch_mean.unsqueeze(0)
                batch_cov = (centered.t() @ centered) / max(1, N-1)

                batch_covs.append(batch_cov)
                batch_means.append(batch_mean)
                batch_sizes.append(N)

                del tensor, centered, batch_cov
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Combine batch covariances using parallel covariance formula
            total_samples = sum(batch_sizes)
            global_mean = torch.zeros_like(batch_means[0])
            for mean, size in zip(batch_means, batch_sizes):
                global_mean += mean * (size / total_samples)

            # First term: sum of weighted covariances
            cov_term1 = torch.zeros_like(batch_covs[0])
            for cov, size in zip(batch_covs, batch_sizes):
                cov_term1 += cov * (size - 1)

            # Second term: correction for mean differences
            cov_term2 = torch.zeros_like(batch_covs[0])
            for mean, size in zip(batch_means, batch_sizes):
                mean_diff = mean - global_mean
                correction = torch.outer(mean_diff, mean_diff) * size
                cov_term2 += correction

            cov = (cov_term1 + cov_term2) / (total_samples - 1)

            cov_size = cov.shape
            cov_mem_mb = (cov.element_size() * cov.nelement()) / (1024 * 1024)
            self.log_file.write(f"Covariance matrix size: {cov_size}, Memory: {cov_mem_mb:.2f} MB\n")
            self.log_file.flush()

            del batch_covs, batch_means, global_mean, cov_term1, cov_term2

        # Compute eigenvalue metrics
        lam = compute_sorted_eigs(cov)
        lam_norm = normalize_eigs(lam)

        se = compute_spectral_entropy(lam_norm).item()
        pr = compute_participation_ratio(lam).item()
        eee = compute_eee(lam).item()

        del cov

        return {
            'lam': lam,
            'lam_norm': lam_norm,
            'se': se,
            'pr': pr,
            'eee': eee
        }
