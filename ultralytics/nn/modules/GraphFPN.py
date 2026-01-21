# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class GATSpatialAttention(nn.Module):
#     """
#     单头 GAT 空间注意力，对应论文式 (1) 的 M(·)。
#     输入:
#         x: [N, C]   所有节点特征
#         edge_index: [2, E]  有向边 (src → dst)
#     输出:
#         out: [N, C]
#     """
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.lin = nn.Linear(in_channels, out_channels, bias=False)
#         # a^T [Wh_i || Wh_j]
#         self.att = nn.Linear(2 * out_channels, 1, bias=False)
#         self.leaky_relu = nn.LeakyReLU(0.2)

#     def forward(self, x, edge_index):
#         # 线性变换
#         Wh = self.lin(x)                  # [N, C']
#         src, dst = edge_index             # [E], [E]
#         Wh_i = Wh[dst]                    # 目标节点特征
#         Wh_j = Wh[src]                    # 源节点特征

#         # 计算注意力系数 e_ij
#         e_ij = self.leaky_relu(
#             self.att(torch.cat([Wh_i, Wh_j], dim=-1))
#         )                                 # [E, 1]
#         # 按目标节点进行 softmax 归一化
#         alpha = self.edge_softmax(dst, e_ij.squeeze(-1))  # [E]

#         # 消息聚合
#         out = torch.zeros_like(Wh)
#         out.index_add_(0, dst, Wh_j * alpha.unsqueeze(-1))

#         return out

#     @staticmethod
#     def edge_softmax(dst_index, attn_scores):
#         """
#         对每个 dst 节点的入边做 softmax。
#         dst_index: [E]
#         attn_scores: [E]
#         """
#         max_per_dst = torch.zeros_like(attn_scores).index_reduce(
#             0, dst_index, attn_scores, reduce='amax', include_self=False
#         )
#         e = (attn_scores - max_per_dst[dst_index]).exp()
#         denom = torch.zeros_like(attn_scores).index_reduce(
#             0, dst_index, e, reduce='sum', include_self=False
#         )
#         return e / (denom + 1e-8)


# class LocalChannelWiseAttention(nn.Module):
#     """
#     论文式 (2): 基于 local neighborhood 的 channel-wise attention (类似 SENet，但局部).
#     对每个节点 i: 对 i 及其邻居特征做平均 -> FC + sigmoid -> 与 h_i 相乘
#     """
#     def __init__(self, channels):
#         super().__init__()
#         self.fc = nn.Linear(channels, channels)

#     def forward(self, x, edge_index):
#         """
#         x: [N, C]
#         edge_index: [2, E] (src -> dst)
#         """
#         N, C = x.shape
#         src, dst = edge_index

#         # 对于每个节点 i，统计 {i}∪N_i 的平均特征
#         # 先把自身特征也当作“一条自环边”
#         all_src = torch.cat([torch.arange(N, device=x.device), src], dim=0)
#         all_dst = torch.cat([torch.arange(N, device=x.device), dst], dim=0)

#         # 聚合求和
#         sum_feat = torch.zeros_like(x)
#         sum_feat.index_add_(0, all_dst, x[all_src])

#         # 统计次数
#         ones = torch.ones(all_src.shape[0], device=x.device)
#         deg = torch.zeros(N, device=x.device)
#         deg.index_add_(0, all_dst, ones)
#         deg = deg.clamp(min=1.0).unsqueeze(-1)

#         avg_feat = sum_feat / deg          # [N, C]

#         # FC + sigmoid
#         att = torch.sigmoid(self.fc(avg_feat))  # [N, C]
#         return x * att


# class LocalChannelSelfAttention(nn.Module):
#     """
#     论文式 (3): local channel self-attention
#         h_i <- β X h_i + h_i
#     其中 X 为基于 {i}∪N_i 特征构造的 C×C 相似度矩阵.
#     """
#     def __init__(self, channels):
#         super().__init__()
#         self.beta = nn.Parameter(torch.tensor(0.0))  # 初始化为 0

#     def forward(self, x, edge_index):
#         """
#         x: [N, C]
#         edge_index: [2, E]
#         """
#         N, C = x.shape
#         src, dst = edge_index

#         # 构造每个节点的局部邻域索引列表
#         # 注意：这里是一个朴素实现，效率一般，但逻辑上贴论文。
#         device = x.device
#         neighbor_list = [[] for _ in range(N)]
#         # 自己也算一个“邻居”
#         for i in range(N):
#             neighbor_list[i].append(i)
#         # 加入真正的邻居
#         for s, d in zip(src.tolist(), dst.tolist()):
#             neighbor_list[d].append(s)

#         out = x.clone()
#         for i in range(N):
#             idx = torch.tensor(neighbor_list[i], device=device, dtype=torch.long)  # [K]
#             A = x[idx]                           # [K, C]
#             # X = A^T A ∈ R^{C×C}
#             X = torch.matmul(A.t(), A)          # [C, C]
#             X = F.softmax(X, dim=-1)            # 对每一行 softmax
#             out[i] = self.beta * (X @ x[i]) + x[i]

#         return out


# class GraphFPNLayer(nn.Module):
#     """
#     一层 GraphFPN layer（无论是 contextual 还是 hierarchical），
#     结构：GAT spatial attention -> local channel-wise -> local channel self-attention -> 残差.
#     """
#     def __init__(self, channels):
#         super().__init__()
#         self.gat = GATSpatialAttention(channels, channels)
#         self.lca = LocalChannelWiseAttention(channels)
#         self.lsa = LocalChannelSelfAttention(channels)
#         self.norm = nn.LayerNorm(channels)

#     def forward(self, x, edge_index):
#         """
#         x: [N, C]
#         edge_index: [2, E]
#         """
#         # GAT 空间 attention
#         h = self.gat(x, edge_index)          # [N, C]
#         h = F.relu(h)

#         # 局部 channel-wise attention
#         h = self.lca(h, edge_index)

#         # 局部 channel self-attention
#         h = self.lsa(h, edge_index)

#         # 残差 + LN
#         out = self.norm(x + h)
#         return out

# class GraphFPNCore(nn.Module):
#     """
#     论文中的 GraphFPN 主体：
#         - 同一批节点
#         - 两套边：contextual_edges, hierarchical_edges
#         - 三组 GNN layers：CGL-1, HGL, CGL-2，每组层数相同（Table 4,6）
#     """
#     def __init__(self, channels, num_layers_each_group=3):
#         super().__init__()
#         self.L = num_layers_each_group

#         self.cgl1 = nn.ModuleList(
#             [GraphFPNLayer(channels) for _ in range(self.L)]
#         )
#         self.hgl = nn.ModuleList(
#             [GraphFPNLayer(channels) for _ in range(self.L)]
#         )
#         self.cgl2 = nn.ModuleList(
#             [GraphFPNLayer(channels) for _ in range(self.L)]
#         )

#     def forward(self, x, edge_ctx, edge_hier):
#         """
#         x: [N, C] 图上所有节点的初始特征 (来自 CNN→GNN 映射)
#         edge_ctx:  [2, E_ctx] 上下文边（同尺度内部，相邻 superpixel）
#         edge_hier:[2, E_hier] 层次边（跨尺度，祖先/后代 superpixel）
#         """
#         # CGL-1
#         for layer in self.cgl1:
#             x = layer(x, edge_ctx)

#         # HGL
#         for layer in self.hgl:
#             x = layer(x, edge_hier)

#         # CGL-2
#         for layer in self.cgl2:
#             x = layer(x, edge_ctx)

#         return x

# class GraphFPNNeck(nn.Module):
#     """
#     YOLOv8 可用的 GraphFPN Neck（概念上等价于论文里的 GraphFPN+FPN 部分）。

#     用法（示意）：
#         feats_in: list of feature maps [C3, C4, C5,...], 每个形状 [B, C_i, H_i, W_i]
#         labels:   list of superpixel label maps [L3, L4, L5,...], 每个 [B, H_i, W_i]，值为 [0, N_i-1]

#         neck = GraphFPNNeck(in_channels=[256,512,1024], out_channels=256, ...)
#         feats_out = neck(feats_in, labels)

#     你可以在 YOLOv8 的 head 里把 feats_out 直接喂给 Detect。
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels=256,
#                  hidden_dim=256,
#                  num_layers_each_group=3):
#         super().__init__()
#         assert isinstance(in_channels, (list, tuple))
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.hidden_dim = hidden_dim
#         self.num_scales = len(in_channels)

#         # 1) 把各尺度 CNN 特征通道映射到图上的 hidden_dim
#         self.cnn_to_hidden = nn.ModuleList([
#             nn.Conv2d(c_in, hidden_dim, kernel_size=1, bias=False)
#             for c_in in in_channels
#         ])

#         # 2) GraphFPN 核心
#         self.graph_fpn = GraphFPNCore(hidden_dim, num_layers_each_group)

#         # 3) GNN 输出映回 CNN 各尺度 + 1×1Conv 融合
#         #    这里与原始 CNN/FPN 特征 concat 后再压回 out_channels
#         self.fuse_convs = nn.ModuleList([
#             nn.Conv2d(hidden_dim + c_in, out_channels, kernel_size=1, bias=False)
#             for c_in in in_channels
#         ])

#     # ---------- 辅助函数：CNN→GNN 映射 ----------

#     @staticmethod
#     def _build_grid2node(labels):
#         """
#         labels: [B, H, W] int，值范围 [0, N_i-1]
#         返回:
#             grid2node: [B, H, W] (就是 labels 本身)
#             num_nodes_per_img: list[int]，每张图的 superpixel 数
#         """
#         B, H, W = labels.shape
#         num_nodes_per_img = []
#         for b in range(B):
#             num_nodes = int(labels[b].max().item()) + 1
#             num_nodes_per_img.append(num_nodes)
#         return labels, num_nodes_per_img

#     @staticmethod
#     def _cnn_to_node_features(feat, labels, proj_conv, hidden_dim):
#         """
#         feat:   [B, C, H, W]  CNN 特征
#         labels: [B, H, W]     superpixel 标号
#         proj_conv: nn.Conv2d 把 C -> hidden_dim
#         返回:
#             node_feats: list[Tensor]，长度 B，每个 [N_i, hidden_dim]
#         """
#         B, C, H, W = feat.shape
#         device = feat.device

#         x = proj_conv(feat)       # [B, hidden_dim, H, W]
#         _, Cg, _, _ = x.shape
#         x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, Cg]

#         node_feats_list = []
#         for b in range(B):
#             lab = labels[b]                     # [H, W]
#             num_nodes = int(lab.max().item()) + 1
#             hf = torch.zeros(num_nodes, Cg, device=device)
#             cnt = torch.zeros(num_nodes, 1, device=device)

#             # 把每个 grid cell 累加到对应 superpixel
#             flat_lab = lab.view(-1)             # [H*W]
#             flat_x   = x[b].view(-1, Cg)        # [H*W, Cg]

#             hf.index_add_(0, flat_lab, flat_x)
#             cnt.index_add_(0, flat_lab, torch.ones_like(flat_lab, dtype=torch.float32).unsqueeze(-1))

#             # 平均 (这里用 mean 代替文中的 max+min+FC，简化版本。如果你想完全复现，可在此处实现式(4))
#             hf = hf / cnt.clamp(min=1.0)

#             node_feats_list.append(hf)          # [N_b, Cg]

#         return node_feats_list  # list of length B

#     # ---------- 辅助函数：构造边 ----------

#     @staticmethod
#     def _build_contextual_edges(labels_list):
#         """
#         根据每个尺度的 label 图构造 contextual edges（同尺度相邻 superpixel）。
#         labels_list: list of [B, H, W]
#         返回:
#             edge_index_ctx: [2, E_ctx]，基于“整个 batch 的所有尺度节点 flatten 后”的全局编号
#             node_offsets: dict[(b, s) -> offset]，后续映射回 CNN 会用到
#             total_nodes: int
#         """
#         device = labels_list[0].device
#         B = labels_list[0].shape[0]
#         num_scales = len(labels_list)

#         node_offsets = {}
#         cur_offset = 0
#         total_nodes = 0

#         # 先统计每张图、每个尺度的节点数，并记录 offset
#         num_nodes_bs = {}
#         for s in range(num_scales):
#             labels = labels_list[s]
#             for b in range(B):
#                 n = int(labels[b].max().item()) + 1
#                 num_nodes_bs[(b, s)] = n
#                 node_offsets[(b, s)] = cur_offset
#                 cur_offset += n
#         total_nodes = cur_offset

#         edges = []

#         # 在每个尺度、每张图上，根据 label 图的 4-邻域构造 superpixel 邻接
#         for s in range(num_scales):
#             labels = labels_list[s]   # [B, H, W]
#             B, H, W = labels.shape
#             for b in range(B):
#                 lab = labels[b]       # [H, W]
#                 offset = node_offsets[(b, s)]

#                 # 右邻域
#                 if W > 1:
#                     a = lab[:, :-1]
#                     b_right = lab[:, 1:]
#                     mask = (a != b_right)
#                     idx = torch.nonzero(mask, as_tuple=False)  # [E1, 2], (y,x)
#                     if idx.numel() > 0:
#                         n1 = a[idx[:, 0], idx[:, 1]]
#                         n2 = b_right[idx[:, 0], idx[:, 1]]
#                         n1 = n1 + offset
#                         n2 = n2 + offset
#                         edges.append(torch.stack([n1, n2], dim=0))
#                         edges.append(torch.stack([n2, n1], dim=0))  # 双向

#                 # 下邻域
#                 if H > 1:
#                     a = lab[:-1, :]
#                     b_down = lab[1:, :]
#                     mask = (a != b_down)
#                     idx = torch.nonzero(mask, as_tuple=False)
#                     if idx.numel() > 0:
#                         n1 = a[idx[:, 0], idx[:, 1]]
#                         n2 = b_down[idx[:, 0], idx[:, 1]]
#                         n1 = n1 + offset
#                         n2 = n2 + offset
#                         edges.append(torch.stack([n1, n2], dim=0))
#                         edges.append(torch.stack([n2, n1], dim=0))

#         if len(edges) == 0:
#             edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
#         else:
#             edge_index = torch.cat(edges, dim=1)         # [2, E_all]
#             edge_index = torch.unique(edge_index, dim=1) # 去重

#         return edge_index, node_offsets, total_nodes

#     @staticmethod
#     def _build_hierarchical_edges(labels_list, node_offsets):
#         """
#         构造相邻尺度之间的 hierarchical edges（祖先–后代），简单实现：
#         - 把 coarse label 图上采样到 fine 尺度大小
#         - 对每个位置的 (fine_label, coarse_label) 形成一条双向边

#         labels_list: list[labels_s]，每个 [B, H_s, W_s]
#         node_offsets: 与 _build_contextual_edges 返回的相同，保证全局编号一致
#         返回:
#             edge_index_hier: [2, E_hier]
#         """
#         device = labels_list[0].device
#         B = labels_list[0].shape[0]
#         num_scales = len(labels_list)

#         edges = []

#         for s in range(num_scales - 1):
#             lab_fine = labels_list[s]         # [B, Hf, Wf]
#             lab_coarse = labels_list[s + 1]   # [B, Hc, Wc]

#             B, Hf, Wf = lab_fine.shape

#             # 上采样 coarse label 到 fine 分辨率
#             up_coarse = F.interpolate(
#                 lab_coarse.float().unsqueeze(1),  # [B,1,Hc,Wc]
#                 size=(Hf, Wf),
#                 mode='nearest'
#             ).squeeze(1).long()   # [B, Hf, Wf]

#             for b in range(B):
#                 lf = lab_fine[b]      # [Hf, Wf]
#                 lc = up_coarse[b]     # [Hf, Wf]

#                 off_f = node_offsets[(b, s)]
#                 off_c = node_offsets[(b, s + 1)]

#                 n_f = lf + off_f
#                 n_c = lc + off_c

#                 pair = torch.stack(
#                     [n_f.reshape(-1), n_c.reshape(-1)], dim=0
#                 )  # [2, Hf*Wf]

#                 # 去重
#                 pair = torch.unique(pair, dim=1)
#                 # 双向
#                 edges.append(pair)
#                 edges.append(torch.flip(pair, dims=[0]))

#         if len(edges) == 0:
#             edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
#         else:
#             edge_index = torch.cat(edges, dim=1)
#             edge_index = torch.unique(edge_index, dim=1)

#         return edge_index

#     @staticmethod
#     def _prune_hierarchical_edges(node_feats, edge_index, keep_ratio=0.5):
#         """
#         按论文描述，用 cosine 相似度 prune hierarchical 边：
#         对每个节点 i，保留 incident edges 中相似度 top 50%.
#         node_feats: [N, C]
#         edge_index: [2, E]
#         """
#         device = node_feats.device
#         src, dst = edge_index

#         # 计算边上的 cosine 相似度
#         x_src = node_feats[src]
#         x_dst = node_feats[dst]
#         cos_sim = F.cosine_similarity(x_src, x_dst, dim=-1)  # [E]

#         # 收集每个节点的入边
#         N = node_feats.shape[0]
#         edge_ids_per_node = [[] for _ in range(N)]
#         for eid, d in enumerate(dst.tolist()):
#             edge_ids_per_node[d].append(eid)
#         for eid, s in enumerate(src.tolist()):
#             edge_ids_per_node[s].append(eid)

#         keep_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=device)

#         for i in range(N):
#             ids = edge_ids_per_node[i]
#             if len(ids) == 0:
#                 continue
#             sims = cos_sim[ids]
#             k = max(1, int(len(ids) * keep_ratio))
#             topk = torch.topk(sims, k, largest=True).indices
#             chosen = torch.tensor(ids, device=device, dtype=torch.long)[topk]
#             keep_mask[chosen] = True

#         pruned_edge_index = edge_index[:, keep_mask]
#         return pruned_edge_index

#     # ---------- GNN → CNN 映射 ----------

#     def _node_to_cnn(self, node_feats_all, labels_list, node_offsets, total_nodes, proj_convs):
#         """
#         node_feats_all: [N_total, hidden_dim]（整个 batch & 多尺度 flatten 后的节点特征）
#         labels_list: list[labels_s], 每个 [B, H_s, W_s]
#         node_offsets: dict[(b,s) -> offset]
#         proj_convs: list[nn.Conv2d]，用于融合 (graph_feat + 原CNN特征)
#         返回:
#             fused_feats: list[Tensor]，每个尺度一个 feature map，形状 [B, out_channels, H_s, W_s]
#         """
#         device = node_feats_all.device
#         B = labels_list[0].shape[0]
#         num_scales = len(labels_list)

#         fused_feats = []

#         for s in range(num_scales):
#             labels = labels_list[s]       # [B, H, W]
#             B, H, W = labels.shape
#             conv_fuse = self.fuse_convs[s]
#             # 注意：这里假设你在外面把原 CNN 特征传进来（见 forward 中的 feats_in）
#             # 先占位，真实融合在 forward 里完成
#             fused_feats.append((labels, H, W))

#         # 实际融合逻辑在 forward 里，为了拿到原 feats_in 一起使用
#         return fused_feats

#     # ---------- forward：整合上面所有步骤 ----------

#     def forward(self, feats_in, labels_list):
#         """
#         feats_in:   list[Tensor]，长度 = num_scales，每个 [B, C_i, H_i, W_i]
#         labels_list:list[Tensor]，长度 = num_scales，每个 [B, H_i, W_i]，为 superpixel label
#         返回:
#             feats_out: list[Tensor]，长度 = num_scales，每个 [B, out_channels, H_i, W_i]
#         """
#         assert len(feats_in) == self.num_scales
#         assert len(labels_list) == self.num_scales

#         B = feats_in[0].shape[0]
#         device = feats_in[0].device

#         # 1）各尺度 CNN → 节点特征（注意：这里按尺度/图片分开）
#         node_feats_per_scale = []   # list of list: [s][b]: [N_b_s, hidden_dim]
#         labels_per_scale = []

#         for s in range(self.num_scales):
#             feat = feats_in[s]           # [B, C_s, H_s, W_s]
#             labels = labels_list[s]      # [B, H_s, W_s]
#             labels_per_scale.append(labels)

#             node_feats_list = self._cnn_to_node_features(
#                 feat, labels, self.cnn_to_hidden[s], self.hidden_dim
#             )  # len B, each [N_b_s, hidden_dim]
#             node_feats_per_scale.append(node_feats_list)

#         # 2）flatten 所有 scale & batch 的节点，构建全局 node_feats_all & offsets
#         node_offsets = {}
#         cur_offset = 0
#         nodes_all = []

#         for s in range(self.num_scales):
#             for b in range(B):
#                 f = node_feats_per_scale[s][b]    # [N_b_s, hidden_dim]
#                 n = f.shape[0]
#                 node_offsets[(b, s)] = cur_offset
#                 nodes_all.append(f)
#                 cur_offset += n

#         node_feats_all = torch.cat(nodes_all, dim=0)  # [N_total, hidden_dim]
#         total_nodes = node_feats_all.shape[0]

#         # 3）构造 contextual edges & hierarchical edges
#         edge_ctx, node_offsets_ctx, total_nodes_ctx = self._build_contextual_edges(labels_per_scale)
#         # node_offsets_ctx 与 node_offsets 理论上是一致的（都是按照 (b,s) 顺序）

#         edge_hier = self._build_hierarchical_edges(labels_per_scale, node_offsets_ctx)

#         # 4）根据初始节点特征 prune hierarchical edges
#         if edge_hier.numel() > 0:
#             edge_hier = self._prune_hierarchical_edges(node_feats_all, edge_hier, keep_ratio=0.5)

#         # 5）丢进 GraphFPN 核心
#         node_feats_refined = self.graph_fpn(node_feats_all, edge_ctx, edge_hier)  # [N_total, hidden_dim]

#         # 6）把节点特征拷回 CNN 特征图，再与原 CNN 特征 concat + 1×1 conv fuse
#         feats_out = []

#         for s in range(self.num_scales):
#             labels = labels_per_scale[s]                  # [B, H_s, W_s]
#             B, H, W = labels.shape
#             feat_orig = feats_in[s]                       # [B, C_s, H_s, W_s]
#             C_s = feat_orig.shape[1]

#             fused = torch.zeros(B, self.hidden_dim, H, W, device=device)

#             for b in range(B):
#                 lab = labels[b]                           # [H, W]
#                 off = node_offsets[(b, s)]
#                 node_ids = lab + off                      # [H, W] -> global node index

#                 # 从 node_feats_refined 里取出对应节点特征
#                 g = node_feats_refined[node_ids.view(-1)] # [H*W, hidden_dim]
#                 g = g.view(H, W, self.hidden_dim).permute(2, 0, 1)  # [Cg, H, W]
#                 fused[b] = g

#             # concat(graph_feat, 原 CNN 特征) → 1×1 conv → out_channels
#             cat = torch.cat([fused, feat_orig], dim=1)    # [B, hidden_dim + C_s, H, W]
#             out = self.fuse_convs[s](cat)                 # [B, out_channels, H, W]
#             feats_out.append(out)

#         return feats_out


import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseGAT(nn.Module):
    """
    稀疏图注意力（兼容 YOLOv8）
    稀疏策略：每个节点仅保留 top-k 条注意力边
    """
    def __init__(self, dim, k=2):
        super().__init__()
        self.k = k
        self.fc = nn.Linear(dim, dim, bias=False)
        self.att = nn.Linear(2 * dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        """
        x: [N, C]
        edge_index: [2, E]
        """
        src, dst = edge_index
        N, C = x.shape

        h = self.fc(x)

        h_src = h[src]
        h_dst = h[dst]

        e = self.leaky_relu(self.att(torch.cat([h_src, h_dst], dim=1))).squeeze()

        # -------- 稀疏化：每个 dst 只保留 top-k 边 --------
        # 排序
        # dst_group 表示所有边按 dst 聚合
        idx_sorted = torch.argsort(e, descending=True)  # 大 → 小

        # 保留 top-k
        keep_mask = torch.zeros_like(e, dtype=torch.bool)
        count = torch.zeros(N, dtype=torch.long, device=x.device)

        for i in idx_sorted:
            d = dst[i]
            if count[d] < self.k:
                keep_mask[i] = True
                count[d] += 1

        # 稀疏化后的边
        src = src[keep_mask]
        dst = dst[keep_mask]
        e = e[keep_mask]
        h_src = h_src[keep_mask]

        # -------- softmax --------
        max_per_dst = torch.full((N,), -1e9, device=x.device)
        max_per_dst.index_put_((dst,), e, accumulate=True)
        max_per_dst = max_per_dst[dst]

        e_exp = torch.exp(e - max_per_dst)
        sum_per_dst = torch.zeros(N, device=x.device)
        sum_per_dst.index_put_((dst,), e_exp, accumulate=True)
        sum_per_dst = sum_per_dst[dst]

        alpha = e_exp / (sum_per_dst + 1e-8)

        # -------- 聚合 --------
        out = torch.zeros_like(h)
        out.index_put_((dst,), h_src * alpha.unsqueeze(1), accumulate=True)

        return out


# ----------------------------
# 图层（GAT + 通道注意力）
# ----------------------------
class GraphFPNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gat = SparseGAT(dim)
        self.channel_att = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, edge):
        h = self.gat(x, edge)
        att = self.channel_att(h)
        h = h * att
        return self.norm(x + h)


# ----------------------------
# 构建 Grid Graph
# ----------------------------
def build_grid_edges(h, w, device):
    """
    构建 4-邻域 grid graph，节点按 row-major 编号（0~HW-1）
    """
    edges = []

    for y in range(h):
        for x in range(w):
            id = y * w + x
            # 右邻
            if x + 1 < w:
                edges.append((id, id + 1))
                edges.append((id + 1, id))
            # 下邻
            if y + 1 < h:
                edges.append((id, id + w))
                edges.append((id + w, id))

    if len(edges) == 0:
        return torch.empty(2, 0, dtype=torch.long, device=device)

    edges = torch.tensor(edges, dtype=torch.long, device=device).t()
    return edges


# ----------------------------
# GraphFPN Neck（YOLOv8 版本）
# ----------------------------
class GraphFPNNeck(nn.Module):
    """
    YOLOv8 兼容版本：
    forward(self, xs) 接受多尺度 CNN 特征，不需要 labels_list。
    """
    def __init__(self, in_channels, out_channels=256, hidden_dim=256, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_scales = len(in_channels)
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers

        # CNN→hidden 映射
        self.proj = nn.ModuleList([
            nn.Conv2d(c, hidden_dim, 1, bias=False)
            for c in in_channels
        ])

        # 多层图网络（每尺度独立）
        self.layers = nn.ModuleList([
            nn.ModuleList([GraphFPNLayer(hidden_dim) for _ in range(num_layers)])
            for _ in range(self.num_scales)
        ])

        # 融合层
        self.fuse = nn.ModuleList([
            nn.Conv2d(hidden_dim + c, out_channels, 1, bias=False)
            for c in in_channels
        ])

    def forward(self, xs):
        """
        xs: list of feature maps
            [P3, P4, P5] = [B,C,H,W]
        """

        B = xs[0].shape[0]
        device = xs[0].device

        results = []

        for s, x in enumerate(xs):
            B, C, H, W = x.shape
            x_h = self.proj[s](x)  # [B, hidden, H, W]

            # 构建 grid graph（只构建一次即可）
            edge = build_grid_edges(H, W, device)  # [2, E]

            # 展平为节点
            x_nodes = x_h.permute(0, 2, 3, 1).reshape(B, H * W, self.hidden_dim)

            outputs = []
            for b in range(B):
                node = x_nodes[b]  # [N, hidden]
                for layer in self.layers[s]:
                    node = layer(node, edge)
                outputs.append(node)

            x_graph = torch.stack(outputs, dim=0).reshape(B, H, W, self.hidden_dim)
            x_graph = x_graph.permute(0, 3, 1, 2)  # [B, hidden, H, W]

            # 融合回 CNN（concat）
            fused = torch.cat([x_graph, x], dim=1)
            fused = self.fuse[s](fused)  # 输出 out_channels

            results.append(fused)

        return results
