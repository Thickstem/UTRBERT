import math
import torch.nn as nn
from performer_pytorch.performer_pytorch import (
    Performer,
    FixedPositionalEmbedding,
    AxialPositionalEmbedding,
    AbsolutePositionalEmbedding,
    Always,
    cast_tuple,
    default,
    exists,
)

SEQ_LEN = 5000


class BertPooler(nn.Module):  # Extracted from scBERT repository "Identity"
    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.dim, cfg.dim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class PerformerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        num_tokens = cfg.num_tokens
        max_seq_len = cfg.max_seq_len
        dim = cfg.dim
        depth = cfg.depth
        heads = cfg.heads
        dim_head = cfg.dim_head
        local_attn_heads = cfg.local_attn_heads
        local_window_size = cfg.local_window_size
        causal = cfg.causal
        ff_mult = cfg.ff_mult
        nb_features = cfg.nb_features
        feature_redraw_interval = cfg.feature_redraw_interval
        reversible = cfg.reversible
        ff_chunks = cfg.ff_chunks
        ff_glu = cfg.ff_glu
        emb_dropout = cfg.emb_dropout
        ff_dropout = cfg.ff_dropout
        attn_dropout = cfg.attn_dropout
        generalized_attention = cfg.generalized_attention
        kernel_fn = nn.ReLU()
        use_scalenorm = cfg.use_scalenorm
        use_rezero = cfg.use_rezero
        cross_attend = cfg.cross_attend
        no_projection = cfg.no_projection
        tie_embed = cfg.tie_embed
        rotary_position_emb = cfg.rotary_position_emb
        axial_position_emb = cfg.axial_position_emb
        axial_position_shape = cfg.axial_position_shape
        auto_check_redraw = cfg.auto_check_redraw
        qkv_bias = cfg.qkv_bias
        attn_out_bias = cfg.attn_out_bias
        shift_tokens = cfg.shift_tokens
        local_attn_heads = cast_tuple(local_attn_heads)

        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)

        if rotary_position_emb:
            self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        elif axial_position_emb:
            axial_position_shape = default(
                axial_position_shape, (math.ceil(max_seq_len / 64), 64)
            )
            self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = Always(None)

        self.dropout = nn.Dropout(emb_dropout)

        self.performer = Performer(
            dim,
            depth,
            heads,
            dim_head,
            local_attn_heads,
            local_window_size,
            causal,
            ff_mult,
            nb_features,
            feature_redraw_interval,
            reversible,
            ff_chunks,
            generalized_attention,
            kernel_fn,
            use_scalenorm,
            use_rezero,
            ff_glu,
            ff_dropout,
            attn_dropout,
            cross_attend,
            no_projection,
            auto_check_redraw,
            qkv_bias,
            attn_out_bias,
            shift_tokens,
        )
        self.norm = nn.LayerNorm(dim)
        self.pooler = BertPooler(cfg)
        self.classifier = nn.Linear(cfg.dim, cfg.label_num)
        self.to_out = nn.Sequential(self.pooler, self.classifier)

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, return_encodings=False, **kwargs):
        b, n, device = *x.shape, x.device
        assert (
            n <= self.max_seq_len
        ), f"sequence length {n} must be less than the max sequence length {self.max_seq_len}"

        # token and positional embeddings
        x = self.token_emb(x)
        x += self.pos_emb(x)

        x = self.dropout(x)

        # performer layers

        layer_pos_emb = self.layer_pos_emb(x)
        x = self.performer(x, pos_emb=layer_pos_emb, **kwargs)

        # norm and to logits
        x = self.norm(x)

        if return_encodings:
            return x

        if exists(self.to_out):
            return self.to_out(x)

        return x @ self.token_emb.weight.t()
