"""
Custom GPT-2 architectural variants for NerVE experiments.

Implements the configurable transformer blocks used across the paper's experimental axes:
  - Normalization placement: PreLN, PostLN, MixLN, norm-free (Section 3.4)
  - Normalization type: LayerNorm, RMSNorm, Identity (Appendix K)
  - FFN weight geometry: weight norm, spectral norm, hyperspherical, nGPT (Section 3.3)
  - Activation functions: GELU, ReLU, LeakyReLU, learnable LeakyReLU (Section 3.1, 3.2)
  - Gated activations: SwiGLU, GeGLU (Section 3)

Reference: NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks (ICLR 2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn.utils import weight_norm, spectral_norm

from typing import Optional, Tuple, Union
from transformers.activations import ACT2FN
import math


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        """
        Root Mean Square Layer Normalization, from https://github.com/bzhangGo/rmsnorm
        :param d: model size
        :param eps:  epsilon value, default 1e-8
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d

        self.scale = nn.Parameter(torch.ones(d))
        # Removed redundant parameter registration
        
    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * self.d ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        return self.scale * x_normed


def convertGPT2model(gpt2_model, new_cfg):
    """
    Convert the decoder blocks in a gpt2 model to customisable myGPT2BLock.
    """

    gpt2_model.config.norm_position = new_cfg.norm_position
    gpt2_model.config.post_ln_layers = new_cfg.post_ln_layers
    gpt2_model.config.ffn_norm_type = new_cfg.ffn_norm_type 

    new_blocks = []
    for i, _ in enumerate(gpt2_model.transformer.h):
        
        new_block = myGPT2Block(new_cfg, layer_idx=i)
        # print(f"Block {i} has norm_position={new_block.norm_position}")       
        new_blocks.append(new_block)
    gpt2_model.transformer.h = nn.ModuleList(new_blocks)
    return gpt2_model


class myGPT2Block(nn.Module):
    """
    A customisable GPT2Block that implements baseline (Pre-LN), post-normalization (Post-LN),
    mixed normalization, and normalization-free configurations.
    """
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.layer_idx = layer_idx

            
        if config.norm_type == "ln":
            self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        elif config.norm_type == "rmsnorm":
            self.ln_1 = RMSNorm(hidden_size, eps=config.layer_norm_epsilon)
            self.ln_2 = RMSNorm(hidden_size, eps=config.layer_norm_epsilon)
        elif config.norm_type == "free":
            self.ln_1 = nn.Identity()
            self.ln_2 = nn.Identity()
        else:
            raise NotImplementedError

        self.norm_position = config.norm_position
        self.post_ln_layers = config.post_ln_layers

             
        # Log normalization position information once during initialization
        self._log_norm_position()
        
        self.attn = myGPT2Attention(config, layer_idx=layer_idx)
        self.mlp = myGPT2MLP(inner_dim, config, layer_idx=layer_idx)
    
    def _log_norm_position(self):
        """
        Log the normalization position used for this layer.
        Called once during initialization.
        """
        if self.norm_position == "mixed":
            if self.layer_idx is None:
                print(f"Warning: layer_idx not specified for mixed normalization mode")
                return
                
            position = "post" if self.layer_idx < self.post_ln_layers else "pre"
            print(f"Layer {self.layer_idx}: Using {position}-normalization (mixed mode with {self.post_ln_layers} post-LN layers)")
        elif self.norm_position == "pre":
            print(f"Layer {self.layer_idx}: Using pre-normalization")
        elif self.norm_position == "post":
            print(f"Layer {self.layer_idx}: Using post-normalization")
        elif self.norm_position == "free":
            print(f"Layer {self.layer_idx}: Using normalization-free architecture")
        else:
            print(f"Layer {self.layer_idx}: Using {self.norm_position}-normalization (unknown type)")
    
    def get_norm_position(self):
        """
        Determine the normalization position based on configuration and layer index.
        For 'mixed' mode, use Post-LN for the first N layers, then Pre-LN for the rest.
        """
        if self.norm_position == "mixed":
            # Use Post-LN for first post_ln_layers layers, Pre-LN for the rest
            if self.layer_idx is None:
                raise ValueError("layer_idx must be specified when using mixed normalization")
            
            return "post" if self.layer_idx < self.post_ln_layers else "pre"
        else:
            # Use the specified norm position
            return self.norm_position


    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,  
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        # Determine effective normalization position for this layer
        effective_norm_position = self.get_norm_position()
        
        # Apply first normalization based on the effective position
        if effective_norm_position == "post":
            # For Post-LN, don't normalize yet
            pass
        else:  # Pre-LN
            hidden_states = self.ln_1(hidden_states)
            
        skip_branch = hidden_states
               
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        
        hidden_states = attn_output
        hidden_states += skip_branch
        
        # Apply normalization after attention based on effective position
        if effective_norm_position == "post":
            hidden_states = self.ln_1(hidden_states)

        skip_branch = hidden_states

        # Apply second normalization based on the effective position
        if effective_norm_position == "post":
            # For Post-LN, don't normalize yet
            pass
        else:  # Pre-LN
            hidden_states = self.ln_2(hidden_states)
        
        feed_forward_hidden_states = self.mlp(hidden_states)      
        hidden_states = feed_forward_hidden_states
        hidden_states += skip_branch

        # Apply final normalization for Post-LN mode
        if effective_norm_position == "post":
            hidden_states = self.ln_2(hidden_states)

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class myGPT2Attention(nn.Module):
    """
    Attn sub-block.
    """
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        assert is_cross_attention == False
        max_positions = config.max_position_embeddings

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.layer_idx = layer_idx

        self.qk_attn = MyConv1D(2 * self.embed_dim, self.embed_dim)
                
        self.v_attn = MyConv1D(self.embed_dim, self.embed_dim, bias=False)
        self.c_proj = MyConv1D(self.embed_dim, self.embed_dim, bias=False)
        
        self.split_size = self.embed_dim
        # Store the split weights but not used elsewhere - can be useful for analysis
        query_weight, key_weight = self.qk_attn.weight.data.split(self.split_size, dim=1)
                
                
        uniform_causal_attn_mat = torch.ones(
            (max_positions, max_positions), dtype=torch.float32
        ) / torch.arange(1, max_positions + 1).view(-1, 1)
        self.register_buffer(
            "uniform_causal_attn_mat",
            torch.tril(
                uniform_causal_attn_mat,
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )
        
        self.register_buffer(
            "diag",
            torch.eye(max_positions).view(1, 1, max_positions, max_positions),
            persistent=False,
        )
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )

    def _attn(self, query, key, value, attention_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [],
                value.size(-1) ** 0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )

                
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device
        )
        attn_weights = torch.where(
            causal_mask, attn_weights.to(attn_weights.dtype), mask_value
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        new_attn_weights = attn_weights.type(value.dtype)
        attn_output = torch.matmul(new_attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,  
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        assert encoder_hidden_states is None
        (query, key) = self.qk_attn(hidden_states).split(self.split_size, dim=2)
        value = self.v_attn(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask
        )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        proj_output = self.c_proj(attn_output)
        
        outputs = (proj_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  


def hyperspherical_norm(x, dim=-1, eps=1e-8):
    """
    Normalize tensor to unit norm along specified dimension with numerical stability.
    Used for hyperspherical normalization.
    """
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


# Add this class to support hyperspherical normalization
class HypersphericalConv1D(nn.Module):
    """
    A version of MyConv1D with hyperspherical normalization.
    Normalizes weights to unit norm along input dimension axis.
    """
    def __init__(self, nf, nx, bias=True):
        super().__init__()
        self.nx = nx
        self.nf = nf

        if bias:
            self.bias = nn.Parameter(torch.zeros(nf))
        else:
            self.bias = nn.Parameter(torch.zeros(nf), requires_grad=False)

        self.weight = nn.Parameter(torch.zeros(nx, nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        # Normalize weights to unit hypersphere along input dimension
        normalized_weight = F.normalize(self.weight, p=2, dim=0)
        
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), normalized_weight)
        x = x.view(size_out)
        return x

    def extra_repr(self):
        return f"in_dim={self.nx}, out_dim={self.nf}, hyperspherical"


class myGPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config, layer_idx=None):
        """
        MLP implementation with support for various normalization techniques including nGPT-style
        hyperspherical normalization with learnable scaling parameters.
        
        Parameters:
        - intermediate_size: (int) Dimension of the intermediate layer.
        - config: Model configuration object.
        - layer_idx: (int, optional) Index of the current layer.
        """
        super().__init__()
        
        embed_dim = config.hidden_size  
        self.layer_idx = layer_idx
        self.activation_function = config.activation_function
        self.ffn_norm_type = getattr(config, "ffn_norm_type", "none")
        
        
        self.n_layers = config.n_layer
        
        # Log the normalization type being used
        print(f"Layer {layer_idx} FFN using normalization: {self.ffn_norm_type}")
        
        # Check if we're using a gated activation (GeGLU or SwiGLU)
        self.is_gated = self.activation_function in ["geglu", "swiglu"]
        
        # For nGPT style, we need to track if we're using hyperspherical normalization of hidden states
        self.use_ngpt_residual = (self.ffn_norm_type == "ngpt")
        
        # Create the appropriate Conv1D layers based on the normalization type
        if self.ffn_norm_type in ["hyperspherical", "ngpt"]:
            # For both hyperspherical normalization types, use our custom implementation
            self.c_fc = HypersphericalConv1D(intermediate_size, embed_dim, bias=False)
            
            if self.is_gated:
                self.up_proj = HypersphericalConv1D(intermediate_size, embed_dim, bias=False)
            
            self.c_proj = HypersphericalConv1D(embed_dim, intermediate_size, bias=False)
            
        else:
            # For other normalizations, start with regular Conv1D and apply normalization if needed
            self.c_fc = MyConv1D(intermediate_size, embed_dim, bias=False)
            
            if self.is_gated:
                self.up_proj = MyConv1D(intermediate_size, embed_dim, bias=False)
                
            self.c_proj = MyConv1D(embed_dim, intermediate_size, bias=False)
            
            # Apply weight normalization if specified
            if self.ffn_norm_type == "weight":
                self.c_fc = weight_norm(self.c_fc, name='weight')
                if self.is_gated:
                    self.up_proj = weight_norm(self.up_proj, name='weight')
                self.c_proj = weight_norm(self.c_proj, name='weight')
                
            # Apply spectral normalization if specified
            elif self.ffn_norm_type == "spectral":
                self.c_fc = spectral_norm(self.c_fc)
                if self.is_gated:
                    self.up_proj = spectral_norm(self.up_proj)
                self.c_proj = spectral_norm(self.c_proj)
        
        # Set up activation functions
        if self.is_gated:
            # Use the Hugging Face Transformers version of activation functions from ACT2FN
            if self.activation_function == "swiglu":
                self.act_fn = ACT2FN["silu"]
            elif self.activation_function == "geglu":
                self.act_fn = ACT2FN["gelu"]
                
            # Create a simple wrapper function for applying the gating mechanism
            self.act = lambda x: self.apply_gating(x)
        else:
            # Standard activation functions
            if config.activation_function == "leaky_relu":
                self.act = LeakyReLU(negative_slope=config.lrelu_neg_slope)   
            elif config.activation_function == "learnable_lrelu":           
                self.act = LearnableLeakyReLU(
                    config=config, 
                    initial_slope=config.lrelu_neg_slope, 
                    layer_idx=layer_idx
                )
            else: 
                self.act = ACT2FN[config.activation_function]
        
        # Create nGPT-specific learnable parameters (only for nGPT mode)
        if self.ffn_norm_type == "ngpt":
            # === nGPT Scaling Parameters for MLP ===
            # Initialize u scaling parameter (equation 20 in paper)
            self.su_init_value = 1.0
            self.su_init_scaling = 1.0
            self.su = nn.Parameter(self.su_init_scaling * torch.ones(intermediate_size, dtype=torch.float32))
            
            # Initialize v scaling parameter (equation 21 in paper)
            self.sv_init_value = 1.0
            self.sv_init_scaling = 1.0
            self.sv = nn.Parameter(self.sv_init_scaling * torch.ones(intermediate_size, dtype=torch.float32))
            
            # === nGPT Residual Connection Parameters ===
            # Initialize alpha_M parameter (eigen learning rate for MLP block)
            # Using αM,init = 0.05 (in order of 1/nlayers) as in Section 2.6
            self.alpha_init_value = 1.0 / (self.n_layers)
            self.alpha_init_scaling = 1.0 / (embed_dim ** 0.5)  # Section 2.6 specifies 1/√dmodel
            self.alpha = nn.Parameter(self.alpha_init_scaling * torch.ones(embed_dim, dtype=torch.float32))
    
    def apply_gating(self, hidden_states):
        """Apply gating mechanism for gated activation functions."""
        gate_output = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        up_output = hidden_states[1] if isinstance(hidden_states, tuple) else None
        
        activated_gate = self.act_fn(gate_output)
        
        if up_output is not None:
            return activated_gate * up_output
        else:
            # This should not happen in normal operation
            return activated_gate
            
    def forward(self, hidden_states):
        # Store original input for nGPT residual connection if needed
        residual = hidden_states if self.use_ngpt_residual else None
        
        if not self.is_gated:
            # Standard non-gated implementation
            hidden_states = self.c_fc(hidden_states)
            
            # Apply nGPT scaling if using that mode
            if self.ffn_norm_type == "ngpt":
                # Get scale with proper initialization adjustment
                su = self.su * (self.su_init_value / self.su_init_scaling)
                hidden_states = hidden_states * su
                
            hidden_states = self.act(hidden_states)
            hidden_states = self.c_proj(hidden_states)
            
            # Apply nGPT-style residual connection if needed
            if self.use_ngpt_residual:
                # Normalize both representations to the hypersphere (Section 2.2.2 of paper)
                orig_norm = hyperspherical_norm(residual)
                output_norm = hyperspherical_norm(hidden_states)
                
                # Compute eigen learning rate with proper initialization adjustment
                # abs() ensures it's positive as in the paper
                lr = torch.abs(self.alpha * (self.alpha_init_value / self.alpha_init_scaling))
                
                # Equation 11 from paper: linear interpolation between original and new states
                result = orig_norm + lr * (output_norm - orig_norm)
                
                # Final normalization to return to hypersphere
                hidden_states = hyperspherical_norm(result)
                
            return hidden_states
        else:
            # Gated implementation (SwiGLU or GeGLU)
            gate_output = self.c_fc(hidden_states)
            up_output = self.up_proj(hidden_states)
            
            # Apply nGPT scaling if using that mode
            if self.ffn_norm_type == "ngpt":
                # Get scales with proper initialization adjustment
                su = self.su * (self.su_init_value / self.su_init_scaling)
                sv = self.sv * (self.sv_init_value / self.sv_init_scaling) 
                
                gate_output = gate_output * su
                up_output  =  up_output * (self.c_proj.in_features ** 0.5)
                up_output = up_output * sv
            
            # Apply gated activation
            activated_gate = self.act_fn(gate_output)
            post_act_output = activated_gate * up_output
            output = self.c_proj(post_act_output)
            
            # Apply nGPT-style residual connection if needed
            if self.use_ngpt_residual:
                # Normalize both representations
                orig_norm = hyperspherical_norm(residual)
                output_norm = hyperspherical_norm(output)
                
                # Calculate eigen learning rate
                lr = torch.abs(self.alpha * (self.alpha_init_value / self.alpha_init_scaling))
                
                # Equation 11 from paper: linear interpolation between original and new states
                result = orig_norm + lr * (output_norm - orig_norm)
                
                # Final normalization to return to hypersphere
                output = hyperspherical_norm(result)
            
            return output
            
    def post_update_step(self):
        """
        Re-normalize weights after optimizer updates if using hyperspherical modes.
        This should be called after each optimization step.
        """
        if self.ffn_norm_type in ["hyperspherical", "ngpt"]:
            with torch.no_grad():
                # Re-normalize weight matrices to ensure they stay on the unit hypersphere
                self.c_fc.weight.data = F.normalize(self.c_fc.weight.data, p=2, dim=0)
                self.c_proj.weight.data = F.normalize(self.c_proj.weight.data, p=2, dim=0)
                
                if self.is_gated:
                    self.up_proj.weight.data = F.normalize(self.up_proj.weight.data, p=2, dim=0)


class MyConv1D(nn.Module):
    """
    (Linear) 1D-convolutional layer that can be reparameterised into skip (see Eq. 6 of paper).

    Args:
        nf (int): The number of output features.
        nx (int): The number of input features.
        bias (bool): Whether or not to use bias parameters.
    """
    def __init__(self, nf, nx, bias=True):
        super().__init__()
        self.nx = nx
        self.nf = nf

        if bias:
            self.bias = nn.Parameter(torch.zeros(nf))
        else:
            self.bias = nn.Parameter(torch.zeros(nf), requires_grad=False)

        self.weight = nn.Parameter(torch.zeros(nx, nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

    def extra_repr(self):
        return f"in_dim={self.nx}, out_dim={self.nf}"  

class LeakyReLU(nn.Module):
    # LeakyReLU nonlinearity.
    __constants__ = ["inplace", "negative_slope"]
    inplace: bool
    negative_slope: float

    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.where(input >= 0.0, input, input * self.negative_slope)

    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""
        return "negative_slope={}{}".format(self.negative_slope, inplace_str)


class LearnableLeakyReLU(nn.Module):
    def __init__(self, config, initial_slope=0.01, layer_idx=None):
        """
        Initializes the Learnable Leaky ReLU activation function with learnable negative slope.

        Parameters:
        - initial_slope: (float, optional) Initial slope value for the negative part of Leaky ReLU. Defaults to 0.01.
        - layer_idx: (int, optional) Specifies the current layer index for per-layer mode. Defaults to None.
             
        """
        super().__init__()
        
        self.layer_idx = layer_idx       
        self.mode = config.learnable_lrelu_mode
        self.n_layers = config.n_layer

        # Initialize the slope parameter based on the mode
        if self.mode == 'global':            
            self.slopes = nn.Parameter(torch.tensor([initial_slope], dtype=torch.float32)) # Single learnable slope shared across all layers
        elif self.mode == 'per_layer':
            self.slopes = nn.Parameter(torch.full((self.n_layers,), initial_slope, dtype=torch.float32))  # Individual learnable slope for each layer
        else:
            raise ValueError("Invalid mode for LearnableLeakyReLU: must be 'global' or 'per_layer'")

    def forward(self, input):       
        if self.mode == 'global':
            slopes = self.slopes # Use a single slope for the entire model
        elif self.mode == 'per_layer':
            if self.layer_idx is None:
                raise ValueError("layer_idx must be specified in 'per_layer' mode")
            slopes = self.slopes[self.layer_idx].unsqueeze(0).unsqueeze(0).unsqueeze(-1) # Retrieve the slope for the current layer index
        else:
            raise ValueError("Unsupported mode")
        
        return torch.where(input >= 0, input, input * slopes)

    def extra_repr(self):
        return f"mode={self.mode}"
