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


class OutLayer(nn.Module):  # Extracted from scBERT repository "Identity"
    def __init__(self, main_dim, dropout=0.0, h_dim=100, out_dim=1):
        super(OutLayer, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, main_dim))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:, None, :, :]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class PerformerModel:
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
        kernel_fn = cfg.kernel_fn
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
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

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
