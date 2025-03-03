"""
GPT Language Model Implementation
This file provides a comprehensive implementation of a GPT-style language model
with Mixture of Experts (MoE) and Multi-head Linear Attention (MLA) mechanisms.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

def rotate_half(x):
    """Split tensor in half along last dimension and rotate values."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    """Apply Rotary Position Embedding to query and key tensors."""
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

def apply_rope_x(x, cos, sin):
    """Apply Rotary Position Embedding to a single tensor."""
    return (x * cos) + (rotate_half(x) * sin)

class MLA(torch.nn.Module):
    """
    Multi-head Linear Attention with Rotary Position Embedding.
    Uses a decoupled approach for position-aware query and key projections.
    """
    def __init__(self, d_model, n_heads, max_len=2048, rope_theta=10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.q_proj_dim = d_model // 2
        self.kv_proj_dim = (2*d_model) // 3

        self.qk_nope_dim = self.dh // 2
        self.qk_rope_dim = self.dh // 2
        
        # Q projections
        self.W_dq = torch.nn.Parameter(0.01*torch.randn((d_model, self.q_proj_dim)))
        self.W_uq = torch.nn.Parameter(0.01*torch.randn((self.q_proj_dim, self.d_model)))
        self.q_layernorm = torch.nn.LayerNorm(self.q_proj_dim)
        
        # KV projections
        self.W_dkv = torch.nn.Parameter(0.01*torch.randn((d_model, self.kv_proj_dim + self.qk_rope_dim)))
        self.W_ukv = torch.nn.Parameter(0.01*torch.randn((self.kv_proj_dim,
                                                          self.d_model + (self.n_heads * self.qk_nope_dim))))
        self.kv_layernorm = torch.nn.LayerNorm(self.kv_proj_dim)
        
        # Output projection
        self.W_o = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))

        # RoPE parameters
        self.max_seq_len = max_len
        self.rope_theta = rope_theta

        # Precompute position embeddings
        freqs = 1.0 / (rope_theta ** (torch.arange(0, self.dh, 2).float() / self.dh))
        emb = torch.outer(torch.arange(self.max_seq_len).float(), freqs)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]

        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def forward(self, x, kv_cache=None, past_length=0):
        """
        Forward pass with optional KV caching for efficient inference.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, d_model]
            kv_cache: Previous key-value cache if available
            past_length: Length of past context if using cache
            
        Returns:
            output: Output tensor of shape [batch_size, sequence_length, d_model]
            new_kv_cache: Updated key-value cache
        """
        B, S, D = x.size()

        # Q Projections
        compressed_q = x @ self.W_dq
        compressed_q = self.q_layernorm(compressed_q)
        Q = compressed_q @ self.W_uq
        Q = Q.view(B, -1, self.n_heads, self.dh).transpose(1,2)
        Q, Q_for_rope = torch.split(Q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        # Q Decoupled RoPE
        cos_q = self.cos_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
        sin_q = self.sin_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
        Q_for_rope = apply_rope_x(Q_for_rope, cos_q, sin_q)

        # KV Projections with optional caching
        if kv_cache is None:
            compressed_kv = x @ self.W_dkv
            KV_for_lora, K_for_rope = torch.split(compressed_kv,
                                                  [self.kv_proj_dim, self.qk_rope_dim],
                                                  dim=-1)
            KV_for_lora = self.kv_layernorm(KV_for_lora)
        else:
            new_kv = x @ self.W_dkv
            compressed_kv = torch.cat([kv_cache, new_kv], dim=1)
            new_kv, new_K_for_rope = torch.split(new_kv,
                                                 [self.kv_proj_dim, self.qk_rope_dim],
                                                 dim=-1)
            old_kv, old_K_for_rope = torch.split(kv_cache,
                                                 [self.kv_proj_dim, self.qk_rope_dim],
                                                 dim=-1)
            new_kv = self.kv_layernorm(new_kv)
            old_kv = self.kv_layernorm(old_kv)
            KV_for_lora = torch.cat([old_kv, new_kv], dim=1)
            K_for_rope = torch.cat([old_K_for_rope, new_K_for_rope], dim=1)
            
        KV = KV_for_lora @ self.W_ukv
        KV = KV.view(B, -1, self.n_heads, self.dh+self.qk_nope_dim).transpose(1,2)
        K, V = torch.split(KV, [self.qk_nope_dim, self.dh], dim=-1)
        S_full = K.size(2)        

        # K Rope
        K_for_rope = K_for_rope.view(B, -1, 1, self.qk_rope_dim).transpose(1,2)
        cos_k = self.cos_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
        sin_k = self.sin_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
        K_for_rope = apply_rope_x(K_for_rope, cos_k, sin_k)

        # Apply position encoding to each head
        K_for_rope = K_for_rope.repeat(1, self.n_heads, 1, 1)

        # Split into multiple heads
        q_heads = torch.cat([Q, Q_for_rope], dim=-1)
        k_heads = torch.cat([K, K_for_rope], dim=-1)
        v_heads = V

        # Causal attention mask
        mask = torch.ones((S,S_full), device=x.device)
        mask = torch.tril(mask, diagonal=past_length)
        mask = mask[None, None, :, :]
        sq_mask = mask == 1

        # Attention mechanism
        x = torch.nn.functional.scaled_dot_product_attention(
            q_heads, k_heads, v_heads,
            attn_mask=sq_mask,
            is_causal=True
        )

        x = x.transpose(1, 2).reshape(B, S, D)

        # Output projection
        x = x @ self.W_o.T

        return x, compressed_kv


class SharedExpert(nn.Module):
    """
    Shared expert module that is always active in the Mixture of Experts layer.
    Implements a standard MLP with GELU activation.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class RoutedExpert(nn.Module):
    """
    Routed expert with bias term for load balancing.
    Each expert has a learnable centroid vector used for routing decisions.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        # Expert centroid vector for routing
        self.centroid = nn.Parameter(torch.randn(config.n_embd))
        
        # Bias term for load balancing (non-trainable)
        self.routing_bias = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class MoE(nn.Module):
    """
    Mixture of Experts implementation with:
    - Auxiliary-loss-free load balancing
    - Sequence-wise balance loss
    - No token dropping
    
    Combines both shared experts (always active) and routed experts 
    (conditionally active based on input).
    """
    def __init__(self, config):
        super().__init__()
        self.n_shared = getattr(config, 'n_shared_experts', 2)
        self.n_routed = getattr(config, 'n_routed_experts', 8)
        self.top_k = getattr(config, 'top_k_experts', 2)
        self.bias_update_speed = getattr(config, 'bias_update_speed', 0.01)
        self.balance_factor = getattr(config, 'balance_factor', 0.01)
        
        # Create shared experts (always active)
        self.shared_experts = nn.ModuleList([
            SharedExpert(config) for _ in range(self.n_shared)
        ])
        
        # Create routed experts (conditionally active)
        self.routed_experts = nn.ModuleList([
            RoutedExpert(config) for _ in range(self.n_routed)
        ])

        # Expert load tracking
        self.register_buffer('expert_load', torch.zeros(self.n_routed))
        self.register_buffer('total_tokens_processed', torch.zeros(1))
        self.balance_loss = 0.0

    def _compute_expert_scores(self, x):
        """Compute raw expert scores without bias."""
        scores = []
        for expert in self.routed_experts:
            # Compute affinity score
            score = F.linear(x, expert.centroid.unsqueeze(0))
            scores.append(score)
        scores = torch.cat(scores, dim=-1)  # (batch, seq_len, n_routed_experts)
        return scores

    def _apply_load_balancing(self, scores):
        """Apply load balancing bias terms to scores."""
        biases = torch.stack([expert.routing_bias for expert in self.routed_experts])
        # Expand biases to match scores shape: [n_routed] -> [1, 1, n_routed]
        biases = biases.view(1, 1, -1)
        # Broadcasting will handle the batch and sequence dimensions
        return scores + biases

    def _update_routing_biases(self, expert_counts, total_tokens):
        """Update expert routing biases based on load."""
        expected_load = total_tokens / self.n_routed
        
        for i, expert in enumerate(self.routed_experts):
            current_load = expert_counts[i]
            if current_load > expected_load:
                expert.routing_bias.data -= self.bias_update_speed
            elif current_load < expected_load:
                expert.routing_bias.data += self.bias_update_speed

    def _compute_sequence_balance_loss(self, routing_probs, top_k_mask):
        """Compute sequence-wise balance loss."""
        # Normalize probabilities
        normalized_probs = routing_probs / (routing_probs.sum(dim=-1, keepdim=True) + 1e-5)
        
        # Calculate expert assignment fractions
        f_i = top_k_mask.float().sum(dim=1) / (self.top_k * routing_probs.size(1))
        
        # Calculate expert selection probabilities
        P_i = normalized_probs.mean(dim=1)
        
        # Compute balance loss
        balance_loss = self.balance_factor * (f_i * P_i).sum()
        
        return balance_loss

    def forward(self, x):
        """
        Forward pass through the Mixture of Experts layer.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, d_model]
            
        Returns:
            Output tensor after processing through shared and routed experts
        """
        batch_size, seq_len, d_model = x.shape
        
        # 1. Compute expert scores and routing
        raw_scores = self._compute_expert_scores(x)
        routing_scores = torch.sigmoid(raw_scores)
        
        # Apply load balancing bias for routing decisions
        balanced_scores = self._apply_load_balancing(raw_scores)
        balanced_routing_scores = torch.sigmoid(balanced_scores)
        
        # Get top-k experts based on balanced scores
        top_k_balanced_scores, top_k_indices = torch.topk(
            balanced_routing_scores, self.top_k, dim=-1
        )
        
        # Use original scores for gating
        top_k_scores = torch.gather(routing_scores, -1, top_k_indices)
        top_k_scores = F.softmax(top_k_scores, dim=-1)
        
        # 2. Process through shared experts
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x)
        
        # 3. Process through routed experts
        routed_output = torch.zeros_like(x)
        expert_counts = torch.zeros(self.n_routed, device=x.device)
        
        # Create mask for top-k selection
        top_k_mask = torch.zeros_like(balanced_routing_scores)
        top_k_mask.scatter_(-1, top_k_indices, 1.0)
        
        for i in range(self.top_k):
            expert_indices = top_k_indices[..., i]  # [batch, seq_len]
            expert_scores = top_k_scores[..., i].unsqueeze(-1)  # [batch, seq_len, 1]
            
            for expert_idx in range(self.n_routed):
                expert_mask = (expert_indices == expert_idx)
                if not expert_mask.any():
                    continue
                
                # Track expert usage
                expert_counts[expert_idx] += expert_mask.sum().item()
                
                # Process tokens
                expert_mask = expert_mask.unsqueeze(-1)  # [batch, seq_len, 1]
                expert_input = x * expert_mask
                expert_output = self.routed_experts[expert_idx](expert_input)
                routed_output = routed_output + (expert_output * expert_scores * expert_mask)
        
        # 4. Update load balancing
        total_tokens = batch_size * seq_len
        self._update_routing_biases(expert_counts, total_tokens)
        
        # 5. Compute sequence balance loss
        self.balance_loss = self._compute_sequence_balance_loss(routing_scores, top_k_mask)
        
        # 6. Return combined output
        return x + shared_output + routed_output

    def get_loss(self):
        """Return the balance loss to be added to the model's main loss."""
        return self.balance_loss
    
class LayerNorm(nn.Module):
    """LayerNorm implementation with optional bias parameter."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Block(nn.Module):
    """
    Transformer block with MLA attention and MoE feed-forward network.
    Implements the standard architecture of:
    1. Layer norm
    2. Self-attention
    3. Residual connection
    4. Layer norm
    5. MLP/MoE
    6. Residual connection
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MLA(
            d_model=config.n_embd,
            n_heads=config.n_head,
            max_len=config.block_size,
            rope_theta=10000.0
        )
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MoE(config)

    def forward(self, x, kv_cache=None, past_length=0):
        # Apply layer norm first to the input
        normed_x = self.ln_1(x)
        # Then pass normalized input to attention
        attn_out, new_kv_cache = self.attn(normed_x, kv_cache, past_length)
        # Add residual connection
        x = x + attn_out
        # Apply second layer norm and MLP
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache

@dataclass
class GPTConfig:
    """Configuration class for GPT model hyperparameters."""
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # MLA specific parameters
    rope_theta: float = 10000.0  # RoPE theta parameter
    n_shared_experts: int = 2          # Number of shared experts
    n_routed_experts: int = 8          # Number of routed experts
    top_k_experts: int = 2             # Number of experts to route to
    bias_update_speed: float = 0.01    # Speed of bias updates for load balancing
    balance_factor: float = 0.01       # Weight of sequence-wise balance loss

class GPT(nn.Module):
    """
    GPT Language Model with MLA attention and MoE feed-forward network.
    This implementation supports efficient inference with KV caching.
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying

        # Initialize all weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings are included as they are used as weights in the final layer.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize weights for linear and embedding layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass through the GPT model.
        
        Args:
            idx: Input token indices of shape [batch_size, sequence_length]
            targets: Optional target tokens for computing loss
            
        Returns:
            logits: Output logits
            loss: Language modeling loss if targets provided, otherwise None
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Forward the GPT model
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Initialize a kv_cache for each layer
        kv_caches = [None] * len(self.transformer.h)
        
        # Process each layer with its own kv_cache
        for i, block in enumerate(self.transformer.h):
            x, kv_caches[i] = block(x, kv_cache=kv_caches[i], past_length=0)
        
        x = self.transformer.ln_f(x)

        # Loss calculation
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """
        Model surgery to decrease the block size if necessary.
        
        For example, we may load the GPT2 pretrained model checkpoint (block size 1024)
        but want to use a smaller block size for some smaller, simpler model.
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        Initialize a GPT model from a pretrained GPT-2 checkpoint.
        
        Args:
            model_type: One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            override_args: Optional dict of parameter overrides
            
        Returns:
            Initialized GPT model with weights from the pretrained checkpoint
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # Configuration based on model type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
            
        # Create model with new configuration
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # Load weights from HuggingFace model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy weights while handling transpositions
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Transpose Conv1D weights
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Direct copy for other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure the optimizer for training the model.
        
        Args:
            weight_decay: Weight decay factor
            learning_rate: Learning rate
            betas: Beta parameters for Adam
            device_type: 'cpu' or 'cuda' to determine if fused optimizer can be used
            
        Returns:
            Configured optimizer
        """
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Apply weight decay only to 2D parameters (weights)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Use fused AdamW if available and on CUDA
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate model flops utilization (MFU) specifically for RTX 2060.
        
        Args:
            fwdbwd_per_iter: Number of forward-backward passes per iteration
            dt: Time per iteration in seconds
            
        Returns:
            mfu: Model flops utilization as a fraction of theoretical maximum
        """
        # Calculate model flops
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        # Calculate achieved FLOPS
        flops_achieved = flops_per_iter * (1.0/dt)
        
        # RTX 2060 theoretical FP16 performance: 13 TFLOPS
        flops_promised = 13e12
        
        mfu = flops_achieved / flops_promised
        return mfu

    def get_gpu_stats(self, fwdbwd_per_iter, dt):
        """
        Get comprehensive GPU stats using only PyTorch.
        
        Args:
            fwdbwd_per_iter: Number of forward-backward passes per iteration
            dt: Time per iteration in seconds
            
        Returns:
            dict: GPU statistics including MFU and memory usage
        """
        mfu = self.estimate_mfu(fwdbwd_per_iter, dt)
        allocated = torch.cuda.memory_allocated()/1024**2
        reserved = torch.cuda.memory_reserved()/1024**2
        
        return {
            'mfu': mfu,
            'memory_allocated_mb': allocated,
            'memory_reserved_mb': reserved,
            'memory_percentage': (allocated / 6144) * 100  # 6144 is total VRAM in MB for RTX 2060
        }

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text by sampling from the model's distribution.
        
        Args:
            idx: Initial token indices of shape [batch_size, sequence_length]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If specified, restricts sampling to the top k most likely tokens
            
        Returns:
            idx: Extended sequence with generated tokens appended
        """
        for _ in range(max_new_tokens):
            # Crop sequence if too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Forward pass
            logits, _ = self(idx_cond)
            # Get logits for the last token and apply temperature
            logits = logits[:, -1, :] / temperature
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx