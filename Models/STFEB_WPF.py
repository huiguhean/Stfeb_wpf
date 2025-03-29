import torch.nn as nn
import torch
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
                             query @ key
                     ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class STFEB_WPF(nn.Module):
    def __init__(
            self,
            num_nodes=134,
            in_steps=36,
            out_steps=12,
            steps_per_day=144,
            input_dim=10,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=12,
            doy_embedding_dim=24,
            moy_embedding_dim=12,
            spatial_embedding_dim=40,
            adaptive_embedding_dim=0,
            feed_forward_dim=256,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
            use_mixed_proj=True,
    ):
        super().__init__()

        self.temperature = nn.Parameter(torch.tensor(0.07))  # 可学习温度系数
        self.mem_num = 160
        self.mem_dim = 40
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.doy_embedding_dim = doy_embedding_dim
        self.moy_embedding_dim = moy_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = 1 * (
                input_embedding_dim * 1
                + self.mem_dim
                + tod_embedding_dim
                + doy_embedding_dim
                + moy_embedding_dim
                + spatial_embedding_dim
                + adaptive_embedding_dim
            # + 74
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim - 3, input_embedding_dim)

        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        self.doy_embedding = nn.Embedding(366, doy_embedding_dim)
        self.moy_embedding = nn.Embedding(12, moy_embedding_dim)
        self._init_weights()
        self.spatial_embedding_dim = spatial_embedding_dim
        self.node_emb = nn.Parameter(
            torch.empty(self.num_nodes, self.spatial_embedding_dim)
        )
        nn.init.xavier_uniform_(self.node_emb)

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        # -----------------------------------------------------------------------------------

        self.ln1 = nn.LayerNorm(in_steps)
        self.dropout1 = nn.Dropout(dropout)
        self.memory = self.construct_memory()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.tod_embedding.weight)
        nn.init.kaiming_uniform_(self.doy_embedding.weight)
        nn.init.kaiming_uniform_(self.moy_embedding.weight)
        # nn.init.kaiming_normal_(self.input_proj01.weight)

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim),
                                             requires_grad=True)  # (M, d)[20,40]
        # nn.init.xavier_normal_(memory_dict['Memory'])  # 对 Memory 进行 Xavier 初始化
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.model_dim - self.mem_dim, self.mem_dim),
                                         requires_grad=True)  # project to query[64,64]  main01

        for name, param in memory_dict.items():  # memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_memory(self, h_t: torch.Tensor):
        query = torch.matmul(h_t, self.memory['Wq'])  # (B, N, d)
        query_norm = F.normalize(query, p=2, dim=-1)  # 临时变量
        memory_norm = F.normalize(self.memory['Memory'], p=2, dim=-1)  # 临时变量
        att_score = torch.matmul(query_norm, memory_norm.t()) / self.temperature  # 余弦相似度 # 温度系数
        att_score = torch.softmax(att_score, dim=-1)
        _, topk_indices = torch.topk(att_score, k=self.mem_num // 4, dim=-1)  # (B, N, mem_num // 2)
        mask = torch.zeros_like(att_score)  # (B, N, M)
        mask.scatter_(-1, topk_indices, 1.0)  # (B, N, M)
        att_score = att_score * mask  # (B, N, M)
        # att_score = torch.softmax(att_score, dim=-1)  # (B, N, M)
        att_score = att_score / (att_score.sum(dim=-1, keepdim=True) + 1e-8)
        value = torch.matmul(att_score, self.memory['Memory'])  # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory['Memory'][ind[..., 0]]  # B, N, d
        neg = self.memory['Memory'][ind[..., 1]]  # B, N, d
        return value, query, pos, neg  # 64,207,64

    def forward(self, x):
        # input X :[batchsize, nodenum, history, dims]
        x = x.permute(0, 2, 1, 3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., -3]  # 16,12,170
        if self.doy_embedding_dim > 0:
            doy = x[..., -2]  # 16,12,170
        if self.moy_embedding_dim > 0:
            moy = x[..., -1]  # 16,12,170
        x = x[..., : -3]  # 256,36,33,7
        x1 = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)16,12,170,24
        features = [x1]

        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                tod.long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.doy_embedding_dim > 0:
            doy_emb = self.doy_embedding(
                doy.long()
            )  # (batch_size, in_steps, num_nodes, doy_embedding_dim)
            features.append(doy_emb)
        if self.moy_embedding_dim > 0:
            moy_emb = self.moy_embedding(
                moy.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(moy_emb)

        if self.spatial_embedding_dim > 0:
            node_emb = self.node_emb
            # 扩展到符合batch和历史输入步数的形状
            spatial_emb = node_emb.expand(
                batch_size, self.in_steps, *node_emb.shape
            )
            features.append(spatial_emb)

        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        x1, query, pos, neg = self.query_memory(x)
        features.append(x1)
        x = torch.cat(features, dim=-1)
        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)
        out = out.squeeze(-1)
        return out.permute(0, 2, 1), query, pos, neg#out X:[batchsize, nodenum, predLong]

