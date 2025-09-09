import torch
import torch.nn as nn
from dgllife.model.gnn.gat import GAT
from dgl.nn.pytorch import Set2Set
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import GINConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import init
from pre_model import config
import math

args = config.parse()

class WeightFusion(nn.Module):

    def __init__(self, feat_views, feat_dim, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(WeightFusion, self).__init__()
        self.feat_views = feat_views
        self.feat_dim = feat_dim
        self.weight = Parameter(torch.empty((1, 1, feat_views), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(int(feat_dim), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:

        return sum([input[i] * weight for i, weight in enumerate(self.weight[0][0])]) + self.bias




class Model(nn.Module):
    def __init__(self, in_feats=64, hidden_feats=None, num_step_set2set=2,
                 num_layer_set2set=3,
                 dropout=args.dropout, device='cpu'):
        super(Model, self).__init__()
        self.device = device
        self.num_features = 84
        if hidden_feats is None:
            hidden_feats = [64, 128]
        self.final_hidden_feats = hidden_feats[-1]
        # self.norm_layer_module = nn.LayerNorm(self.final_hidden_feats).to(device)
        self.gnn1 = GINModule(in_feats, hidden_feats, dropout, num_step_set2set, num_layer_set2set)
        self.gnn2 = GATModule()

        self.attention = nn.MultiheadAttention(384, 8)

        self.mlp1 = nn.Linear(384, 1)

        self.sigmoid = nn.Sigmoid()
        # self.separator=Separator()
        self.Layernorm = nn.LayerNorm(384)
        self.fc_proj = nn.Linear(256, 384)
        self.fusion = WeightFusion(feat_views=2, feat_dim=384, device=self.device)


    def forward(self, padded_smiles_batch, batch, return_node_importance=False):
        # get graph input
        # aug_times = 4
        ck = list()
        self.batch_cache = batch
        batch_size = padded_smiles_batch.size(0)
        # # graph
        graph1, node_feat, loss2 = self.gnn1(batch.x, batch.edge_index, batch.batch)
        ck.append(graph1)
        graph2, attn_weights = self.gnn2(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        ck.append(graph2)
        # graph = graph2 + graph1
        graph = self.fusion(torch.stack(ck, dim=0))
        graph_x = self.Layernorm(graph).view(batch_size, 1, -1)

        out = self.mlp1(graph_x.squeeze(1))
        # out = self.mlp1(fused_graph.squeeze(1))

        # loss = 1 - F.cosine_similarity(graph1, graph2, dim=-1).mean()

        batch_b, _ = graph1.size()
        x1_abs = graph1.norm(dim=1)
        x2_abs = graph2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', graph1, graph2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / 0.1)
        pos_sim = sim_matrix[range(batch_b), range(batch_b)]
        loss1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss1).mean()


        if return_node_importance:
            return out, graph_x
        return out, loss

        # return out, alignment_loss
        # # return out, 0

    def predict(self, smiles, graphs, atom_feats, fp_t):
        return self.sigmoid(self.forward(smiles, graphs, atom_feats, fp_t))




class GINModule(nn.Module):
    def __init__(self, in_feats=64, hidden_feats=None, dropout=args.dropout, num_step_set2set=6,
                 num_layer_set2set=3):
        super(GINModule, self).__init__()
        self.conv = GAT(in_feats, hidden_feats)
        self.readout = Set2Set(input_dim=hidden_feats[-1],
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.norm = GraphNorm(hidden_feats[-1] * 2)
        self.fc = nn.Sequential(nn.Linear(hidden_feats[-1] * 2, hidden_feats[-1]), nn.ReLU(),
                                nn.Dropout(p=dropout))
        num_features_xd = 84

        self.conv1 = GINConv(nn.Linear(num_features_xd, num_features_xd))
        self.conv2 = GINConv(nn.Linear(num_features_xd, num_features_xd * 10))
        self.fc_g = nn.Sequential(
            nn.Linear(num_features_xd * 10 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 384)
        )
        self.relu = nn.ReLU()
        self.vq = VectorQuantize(dim=num_features_xd * 10,
                                 codebook_size=4000,
                                 commitment_weight=0.1,
                                 decay=0.9)

    def vector_quantize(self, f, vq_model):
        v_f, indices, v_loss = vq_model(f)

        return v_f, v_loss

    def forward(self, x, edge_index, batch):
        x_g1 = self.relu(self.conv1(x, edge_index))
        x_g2 = self.relu(self.conv2(x_g1, edge_index))  # 节点级特征

        # 保存节点特征用于后续分析
        self.node_features = x_g2

        # 池化得到图级特征
        x_g = torch.cat([gmp(x_g2, batch), gap(x_g2, batch)], dim=1)
        x_g = self.fc_g(x_g)
        return x_g, x_g2, 0


class GATModule(torch.nn.Module):
    def __init__(self, num_features_xd=84, output_dim=384, heads=args.head, edge_dim=11, dropout=args.dropout):
        super(GATModule, self).__init__()
        self.gat1 = GATConv(num_features_xd, output_dim, heads=args.head, edge_dim=11, dropout=dropout)
        self.gat2 = GATConv(output_dim * heads, output_dim, heads=1, edge_dim=11, dropout=dropout)

    def forward(self, x1, edge_index, edge_attr, batch):
        x1, weight = self.gat1(x1, edge_index, edge_attr, return_attention_weights=True)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)

        x1, weight2 = self.gat2(x1, edge_index, edge_attr, return_attention_weights=True)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x_mean = gmp(x1, batch)

        return x_mean, weight2


