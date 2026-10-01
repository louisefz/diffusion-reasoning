"""ELF transformer model."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modules.layers import (
    Attention, BottleneckTextProj, FinalLayer, RMSNorm, SwiGLUFFN,
    TextRotaryEmbeddingFast, TimestepEmbedder,
    DEFAULT_KERNEL_INIT, DEFAULT_BIAS_INIT, NORMAL_INIT_002,
    _make_linear,
)


class ELFBlock(nn.Module):
    """ELF Transformer block."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    def forward(self, x: torch.Tensor, rope_fn: Optional[nn.Module] = None,
                attention_mask: Optional[torch.Tensor] = None,
                deterministic: bool = True) -> torch.Tensor:
        x_normed = self.norm1(x)
        attn_out = self.attn(x_normed, rope_fn, attention_mask=attention_mask,
                             deterministic=deterministic)
        x = x + attn_out

        x_normed = self.norm2(x)
        mlp_out = self.mlp(x_normed, deterministic=deterministic)
        x = x + mlp_out
        return x


class ELF(nn.Module):
    """Text ELF Transformer."""

    def __init__(
        self,
        text_encoder_dim: int,
        max_length: int,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 128,
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 4,
        num_model_mode_tokens: int = 0,
        vocab_size: int = 0,
        gradient_checkpointing: bool = False,
        reasoning_loops: int = 1,
        reasoning_loop_start: int = 10,
        reasoning_loop_end: int = 11,
        reasoning_loop_scale: float = 1.0,
        reasoning_loop_randomize: bool = False,
        reasoning_loop_min: int = 1,
        reasoning_loop_max: int = 1,
        reasoning_memory_tokens: int = 0,
        reasoning_memory_inner_time: bool = True,
        reasoning_memory_direct_coupling: bool = False,
        reasoning_memory_bottleneck: bool = False,
    ):
        super().__init__()
        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.bottleneck_dim = bottleneck_dim
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.vocab_size = vocab_size
        self.gradient_checkpointing = gradient_checkpointing
        self.reasoning_loops = int(reasoning_loops)
        self.reasoning_loop_start = int(reasoning_loop_start)
        self.reasoning_loop_end = int(reasoning_loop_end)
        self.reasoning_loop_scale = float(reasoning_loop_scale)
        self.reasoning_loop_randomize = bool(reasoning_loop_randomize)
        self.reasoning_loop_min = int(reasoning_loop_min)
        self.reasoning_loop_max = int(reasoning_loop_max)
        self.reasoning_memory_tokens = int(reasoning_memory_tokens)
        self.reasoning_memory_inner_time = bool(reasoning_memory_inner_time)
        self.reasoning_memory_direct_coupling = bool(reasoning_memory_direct_coupling)
        self.reasoning_memory_bottleneck = bool(reasoning_memory_bottleneck)
        if not 1 <= self.reasoning_loop_start <= self.reasoning_loop_end <= depth:
            raise ValueError(
                "Reasoning loop window must use one-based block indices within the model")
        if self.reasoning_loops < 1:
            raise ValueError("reasoning_loops must be >= 1")
        if not 1 <= self.reasoning_loop_min <= self.reasoning_loop_max:
            raise ValueError("Invalid randomized reasoning-loop range")
        if not 0.0 < self.reasoning_loop_scale <= 1.0:
            raise ValueError("reasoning_loop_scale must lie in (0, 1]")
        self.last_reasoning_loops = 1
        if self.reasoning_memory_tokens < 0:
            raise ValueError("reasoning_memory_tokens must be non-negative")
        if self.reasoning_memory_tokens > 0:
            self.reasoning_memory = nn.Parameter(torch.empty(
                1, self.reasoning_memory_tokens, hidden_size))
            NORMAL_INIT_002(self.reasoning_memory)
            self.reasoning_time_embedder = TimestepEmbedder(hidden_size)
            if self.reasoning_memory_direct_coupling:
                self.reasoning_memory_norm = RMSNorm(hidden_size, eps=1e-6)
                self.reasoning_memory_out = _make_linear(
                    hidden_size, hidden_size, bias=True,
                    kernel_init=NORMAL_INIT_002, bias_init=DEFAULT_BIAS_INIT)
                # sigmoid(-4) ~= 0.018: initially conservative but with a
                # non-zero gradient so flow loss can learn causal use.
                self.reasoning_memory_gate = nn.Parameter(torch.tensor(-4.0))

        # Self-conditioning input projection (only used when input is [z, x_pred]).
        self.self_cond_proj = _make_linear(2 * text_encoder_dim, text_encoder_dim, bias=True)

        # Text bottleneck projection.
        self.text_proj = BottleneckTextProj(text_encoder_dim, hidden_size, bottleneck_dim)

        # Time / SC-CFG embedders + learned prefix tokens.
        if num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive for prefix time conditioning")
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, num_time_tokens, hidden_size))
        NORMAL_INIT_002(self.t_emb_tokens)

        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(hidden_size)
            self.self_cond_cfg_tokens = nn.Parameter(torch.empty(1, num_self_cond_cfg_tokens, hidden_size))
            NORMAL_INIT_002(self.self_cond_cfg_tokens)

        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(torch.empty(1, num_model_mode_tokens, hidden_size))
            NORMAL_INIT_002(self.mode_tokens)

        head_dim = hidden_size // num_heads
        prefix_total = num_model_mode_tokens + num_time_tokens
        if num_self_cond_cfg_tokens > 0:
            prefix_total += num_self_cond_cfg_tokens
        self.feat_rope = TextRotaryEmbeddingFast(
            dim=head_dim, pt_seq_len=max_length, num_empty_token=prefix_total,
        )
        if self.reasoning_memory_tokens > 0:
            self.reasoning_feat_rope = TextRotaryEmbeddingFast(
                dim=head_dim,
                pt_seq_len=max_length + self.reasoning_memory_tokens,
                num_empty_token=prefix_total,
            )

        self.blocks = nn.ModuleList()
        q1, q3 = depth // 4, depth // 4 * 3
        for i in range(depth):
            in_drop_range = q3 > i >= q1
            self.blocks.append(ELFBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if in_drop_range else 0.0,
                proj_drop=proj_drop if in_drop_range else 0.0,
            ))

        # Final flow-matching output head.
        self.final_layer = FinalLayer(hidden_size, patch_size=1, out_channels=text_encoder_dim)

        # Factored decoder unembedding: hidden -> text_encoder_dim -> vocab.
        bn = text_encoder_dim
        self.proj_kernel = nn.Parameter(torch.empty(hidden_size, bn))
        self.proj_bias = nn.Parameter(torch.empty(bn))
        self.unembed_kernel = nn.Parameter(torch.empty(bn, vocab_size))
        self.unembed_bias = nn.Parameter(torch.empty(vocab_size))
        DEFAULT_KERNEL_INIT(self.proj_kernel)
        DEFAULT_BIAS_INIT(self.proj_bias)
        DEFAULT_KERNEL_INIT(self.unembed_kernel)
        DEFAULT_BIAS_INIT(self.unembed_bias)

    def build_context(self, t: torch.Tensor,
                      self_cond_cfg_scale: Optional[torch.Tensor] = None) -> list:
        B = t.shape[0]
        prefix_tokens = []

        time_emb = self.t_embedder(t)  # (B, hidden)
        prefix_tokens.append(
            self.t_emb_tokens.expand(B, -1, -1) + time_emb.unsqueeze(1)
        )

        if self_cond_cfg_scale is not None and self.num_self_cond_cfg_tokens > 0:
            sc_emb = self.self_cond_cfg_embedder(self_cond_cfg_scale)
            prefix_tokens.append(
                self.self_cond_cfg_tokens.expand(B, -1, -1) + sc_emb.unsqueeze(1)
            )
        return prefix_tokens

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        self_cond_cfg_scale: Optional[torch.Tensor] = None,
        decoder_step_active: Optional[bool] = None,
        reasoning_loop_counts: Optional[torch.Tensor] = None,
        reasoning_readout_positions: Optional[torch.Tensor] = None,
        reasoning_state_token_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x: (N, S, C) or (N, S, 2C) with self-cond. t: (N,). attention_mask: (N, S), 1=valid."""
        B = x.shape[0]

        # Self-conditioning: input is [z, x_pred] when 2x encoder dim
        with torch.amp.autocast('cuda', enabled=False):
            if x.shape[-1] == 2 * self.text_encoder_dim:
                x = self.self_cond_proj(x.float())
            x = self.text_proj(x.float())
            context_prefix_tokens = self.build_context(t, self_cond_cfg_scale)

        # Prepend learnable model-mode tokens (gated by decoder_step_active).
        # decoder_step_active may be None / Python bool / (B,) tensor — the last
        # form supports per-example branching at training time.
        model_mode_offset = 0
        if self.num_model_mode_tokens > 0:
            mode_tokens = self.mode_tokens.expand(B, -1, -1)
            if decoder_step_active is None:
                active_gate = 0.0
            elif isinstance(decoder_step_active, torch.Tensor) and decoder_step_active.dim() > 0:
                active_gate = decoder_step_active.to(mode_tokens.dtype).view(-1, 1, 1)
            else:
                active_gate = float(decoder_step_active)
            mode_tokens = mode_tokens * active_gate
            x = torch.cat([mode_tokens, x], dim=1)
            model_mode_offset = self.num_model_mode_tokens
            if attention_mask is not None:
                mode_mask = torch.ones((B, self.num_model_mode_tokens),
                                       dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([mode_mask, attention_mask], dim=1)

        prefix_len = 0
        if context_prefix_tokens:
            prefix_tokens = torch.cat(context_prefix_tokens, dim=1)
            prefix_len = prefix_tokens.shape[1]
            x = torch.cat([prefix_tokens, x], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones((B, prefix_len),
                                         dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        use_checkpoint = self.gradient_checkpointing and self.training and torch.is_grad_enabled()

        def run_block(block: ELFBlock, hidden: torch.Tensor,
                      block_attention_mask: Optional[torch.Tensor] = attention_mask,
                      rope_fn: Optional[nn.Module] = self.feat_rope) -> torch.Tensor:
            if use_checkpoint:
                def _block_forward(hidden: torch.Tensor, block: ELFBlock = block) -> torch.Tensor:
                    return block(hidden, rope_fn=rope_fn, attention_mask=block_attention_mask,
                                 deterministic=deterministic)
                return checkpoint(_block_forward, hidden, use_reentrant=False)
            return block(hidden, rope_fn=rope_fn, attention_mask=block_attention_mask,
                         deterministic=deterministic)

        loop_start = self.reasoning_loop_start - 1
        loop_end = self.reasoning_loop_end  # exclusive Python index
        for block in self.blocks[:loop_start]:
            x = run_block(block, x)

        if reasoning_loop_counts is not None:
            sampled_loops = reasoning_loop_counts.to(device=x.device).long().clamp(
                min=1, max=self.reasoning_loop_max)
            max_loops = self.reasoning_loop_max
            self.last_reasoning_loops = sampled_loops
        elif self.training and self.reasoning_loop_randomize:
            # Keep the recurrent graph static for torch.compile.  We compute up
            # to K_max and use tensor gates to sample a batch-level K uniformly;
            # no `.item()` graph break and every shared parameter remains in the
            # autograd graph under DDP.
            sampled_loops = torch.randint(
                self.reasoning_loop_min, self.reasoning_loop_max + 1, (B,),
                device=x.device)
            max_loops = self.reasoning_loop_max
            self.last_reasoning_loops = sampled_loops
        else:
            sampled_loops = None
            max_loops = self.reasoning_loops
            self.last_reasoning_loops = self.reasoning_loops

        reasoning_hiddens = []
        if self.reasoning_memory_tokens > 0:
            # The token state after the prefix blocks acts as a fixed external
            # field during inner reasoning. A compact memory evolves under the
            # tied block dynamics; the token output is re-read from the latest
            # memory after each active iteration.
            base_x = x
            token_result = x
            memory = self.reasoning_memory.expand(B, -1, -1)
            if attention_mask is None:
                loop_attention_mask = None
            else:
                memory_mask = torch.ones(
                    (B, self.reasoning_memory_tokens), dtype=attention_mask.dtype,
                    device=attention_mask.device)
                loop_attention_mask = torch.cat([attention_mask, memory_mask], dim=1)
            for loop_index in range(max_loops):
                if self.reasoning_memory_inner_time:
                    tau = torch.full(
                        (B,), float(loop_index), device=x.device, dtype=torch.float32)
                    tau_embedding = self.reasoning_time_embedder(tau).unsqueeze(1)
                else:
                    tau_embedding = torch.zeros(
                        (B, 1, self.hidden_size), device=x.device, dtype=memory.dtype)
                joined = torch.cat([base_x, memory + tau_embedding.to(memory.dtype)], dim=1)
                candidate = joined
                for block in self.blocks[loop_start:loop_end]:
                    candidate = run_block(
                        block, candidate, loop_attention_mask, self.reasoning_feat_rope)
                candidate_tokens = candidate[:, :base_x.shape[1]]
                candidate_memory = candidate[:, base_x.shape[1]:] - tau_embedding.to(candidate.dtype)
                if sampled_loops is None:
                    gate = 1.0
                else:
                    gate = (sampled_loops > loop_index).to(candidate.dtype).view(B, 1, 1)
                memory = memory + gate * self.reasoning_loop_scale * (candidate_memory - memory)
                if not self.reasoning_memory_bottleneck:
                    token_result = token_result + gate * self.reasoning_loop_scale * (
                        candidate_tokens - token_result)
                if reasoning_readout_positions is not None:
                    reasoning_hiddens.append(memory.mean(dim=1))
            x = token_result
            if self.reasoning_memory_direct_coupling:
                memory_drive = self.reasoning_memory_out(
                    self.reasoning_memory_norm(memory.mean(dim=1)))
                if self.reasoning_memory_bottleneck:
                    coupling = torch.ones((), device=x.device, dtype=x.dtype)
                else:
                    coupling = torch.sigmoid(self.reasoning_memory_gate).to(x.dtype)
                x = x + coupling * memory_drive.to(x.dtype).unsqueeze(1)
        else:
            # Checkpoint-compatible path: K=1 is exactly the original network.
            for block in self.blocks[loop_start:loop_end]:
                x = run_block(block, x)
            if reasoning_readout_positions is not None:
                rows = torch.arange(B, device=x.device)
                readout_indices = (
                    reasoning_readout_positions.to(device=x.device).long()
                    + prefix_len + model_mode_offset)
                reasoning_hiddens.append(x[rows, readout_indices])
            for loop_index in range(1, max_loops):
                before = x
                candidate = x
                for block in self.blocks[loop_start:loop_end]:
                    candidate = run_block(block, candidate)
                if sampled_loops is None:
                    gate = 1.0
                else:
                    gate = (sampled_loops > loop_index).to(candidate.dtype).view(B, 1, 1)
                x = before + gate * self.reasoning_loop_scale * (candidate - before)
                if reasoning_readout_positions is not None:
                    reasoning_hiddens.append(x[rows, readout_indices])

        for block in self.blocks[loop_end:]:
            x = run_block(block, x)

        x = x[:, prefix_len + model_mode_offset:]

        # Factored decoder unembedding: hidden -> text_encoder_dim -> vocab
        with torch.amp.autocast('cuda', enabled=False):
            decoder_logits = None
            if decoder_step_active is not None:
                x_f32 = x.float()
                hidden = F.gelu(x_f32 @ self.proj_kernel + self.proj_bias, approximate="tanh")
                decoder_logits = hidden @ self.unembed_kernel + self.unembed_bias
            output = self.final_layer(x.float())
            reasoning_logits = None
            if reasoning_hiddens:
                reasoning_hidden = torch.stack(reasoning_hiddens, dim=1).float()
                reasoning_hidden = F.gelu(
                    reasoning_hidden @ self.proj_kernel + self.proj_bias,
                    approximate="tanh",
                )
                if reasoning_state_token_ids is None:
                    reasoning_logits = reasoning_hidden @ self.unembed_kernel + self.unembed_bias
                else:
                    # A scale-controlled semantic readout. Cosine similarity
                    # removes the arbitrary norm of the native unembedding
                    # logits while retaining its learned token geometry.
                    token_directions = self.unembed_kernel.index_select(
                        1, reasoning_state_token_ids.to(self.unembed_kernel.device).long()
                    ).transpose(0, 1)
                    reasoning_logits = 10.0 * (
                        F.normalize(reasoning_hidden, dim=-1)
                        @ F.normalize(token_directions.float(), dim=-1).transpose(0, 1)
                    )
        if reasoning_readout_positions is not None:
            return output, decoder_logits, reasoning_logits
        return output, decoder_logits


# Model factory functions
def ELF_B(**kwargs): return ELF(depth=12, hidden_size=768,  num_heads=12, **kwargs)
def ELF_M(**kwargs): return ELF(depth=24, hidden_size=1056, num_heads=16, **kwargs)
def ELF_L(**kwargs): return ELF(depth=32, hidden_size=1280, num_heads=16, **kwargs)

ELF_models = {
    'ELF-B': ELF_B, 'ELF-M': ELF_M, 'ELF-L': ELF_L,
}
