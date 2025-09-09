from tqdm import tqdm
import torch
from torch.nn import Module
from torch.nn import functional as F
from models.transition import ContigousTransition, GeneralCategoricalTransition
from models.graph import NodeEdgeNet
from torch_geometric.nn import GINConv

from .common import *
from .diffusion import *


class DiffTox(Module):
    def __init__(self,
                 config,
                 num_node_types,
                 num_edge_types,  # explicit bond type: 0, 1, 2, 3, 4
                 **kwargs
                 ):
        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.bond_len_loss = getattr(config, 'bond_len_loss', False)

        # # define beta and alpha
        self.define_betas_alphas(config.diff)

        # # embedding
        node_dim = config.node_dim
        edge_dim = config.edge_dim
        time_dim = config.diff.time_dim
        self.node_embedder = nn.Linear(num_node_types, node_dim - time_dim, bias=False)  # element type
        self.edge_embedder = nn.Linear(num_edge_types, edge_dim - time_dim, bias=False)  # bond type
        self.time_emb = nn.Sequential(
            GaussianSmearing(stop=self.num_timesteps, num_gaussians=time_dim, type_='linear'),
        )

        # # denoiser
        if config.denoiser.backbone == 'NodeEdgeNet':
            self.denoiser = NodeEdgeNet(node_dim, edge_dim, **config.denoiser)
        else:
            raise NotImplementedError(config.denoiser.backbone)

        # # decoder
        self.node_decoder = MLP(node_dim, num_node_types, node_dim)
        self.edge_decoder = MLP(edge_dim, num_edge_types, edge_dim)

    def define_betas_alphas(self, config):
        self.num_timesteps = config.num_timesteps
        self.categorical_space = getattr(config, 'categorical_space', 'discrete')

        # try to get the scaling
        if self.categorical_space == 'continuous':
            self.scaling = getattr(config, 'scaling', [1., 1., 1.])
        else:
            self.scaling = [1., 1., 1.]  # actually not used for discrete space (defined for compatibility)

        pos_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_pos
        )
        assert self.scaling[0] == 1, 'scaling for pos should be 1'
        self.pos_transition = ContigousTransition(pos_betas)

        node_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_atom
        )
        if self.categorical_space == 'discrete':
            init_prob = config.diff_atom.init_prob
            self.node_transition = GeneralCategoricalTransition(node_betas, self.num_node_types,
                                                                init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_node = self.scaling[1]
            self.node_transition = ContigousTransition(node_betas, self.num_node_types, scaling_node)
        else:
            raise ValueError(self.categorical_space)

        # # diffusion for edge type
        edge_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_bond
        )
        if self.categorical_space == 'discrete':
            init_prob = config.diff_bond.init_prob
            self.edge_transition = GeneralCategoricalTransition(edge_betas, self.num_edge_types,
                                                                init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_edge = self.scaling[2]
            self.edge_transition = ContigousTransition(edge_betas, self.num_edge_types, scaling_edge)
        else:
            raise ValueError(self.categorical_space)

    def sample_time(self, num_graphs, device, **kwargs):
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        pt = torch.ones_like(time_step).float() / self.num_timesteps
        return time_step, pt

    def add_noise(self, node_type, node_pos, batch_node,
                  halfedge_type, halfedge_index, batch_halfedge,
                  num_mol, t, bond_predictor=None, **kwargs):
        num_graphs = num_mol
        device = node_pos.device

        time_step = t * torch.ones(num_graphs, device=device).long()

        pos_pert = self.pos_transition.add_noise(node_pos, time_step, batch_node)
        node_pert = self.node_transition.add_noise(node_type, time_step, batch_node)
        halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, batch_halfedge)
        # edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        # batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        if self.categorical_space == 'discrete':
            h_node_pert, log_node_t, log_node_0 = node_pert
            h_halfedge_pert, log_halfedge_t, log_halfedge_0 = halfedge_pert
        else:
            h_node_pert, h_node_0 = node_pert
            h_halfedge_pert, h_halfedge_0 = halfedge_pert
        return [h_node_pert, pos_pert, h_halfedge_pert]

    def get_loss(self, node_type, node_pos, batch_node,
                 halfedge_type, halfedge_index, batch_halfedge,
                 num_mol
                 ):
        num_graphs = num_mol
        device = node_pos.device

        time_step, _ = self.sample_time(num_graphs, device)

        pos_pert = self.pos_transition.add_noise(node_pos, time_step, batch_node)
        node_pert = self.node_transition.add_noise(node_type, time_step, batch_node)
        halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, batch_halfedge)
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)  # undirected edges
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        if self.categorical_space == 'discrete':
            h_node_pert, log_node_t, log_node_0 = node_pert
            h_halfedge_pert, log_halfedge_t, log_halfedge_0 = halfedge_pert
        else:
            h_node_pert, h_node_0 = node_pert
            h_halfedge_pert, h_halfedge_0 = halfedge_pert

        h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)

        preds = self(
            h_node_pert, pos_pert, batch_node,
            h_edge_pert, edge_index, batch_edge,
            time_step,
        )
        pred_node = preds['pred_node']
        pred_pos = preds['pred_pos']
        pred_halfedge = preds['pred_halfedge']

        loss_pos = F.mse_loss(pred_pos, node_pos)
        if self.bond_len_loss == True:
            bond_index = halfedge_index[:, halfedge_type > 0]
            true_length = torch.norm(node_pos[bond_index[0]] - node_pos[bond_index[1]], dim=-1)
            pred_length = torch.norm(pred_pos[bond_index[0]] - pred_pos[bond_index[1]], dim=-1)
            loss_len = F.mse_loss(pred_length, true_length)

        if self.categorical_space == 'discrete':
            log_node_recon = F.log_softmax(pred_node, dim=-1)
            log_node_post_true = self.node_transition.q_v_posterior(log_node_0, log_node_t, time_step, batch_node,
                                                                    v0_prob=True)
            log_node_post_pred = self.node_transition.q_v_posterior(log_node_recon, log_node_t, time_step, batch_node,
                                                                    v0_prob=True)
            kl_node = self.node_transition.compute_v_Lt(log_node_post_true, log_node_post_pred, log_node_0, t=time_step,
                                                        batch=batch_node)
            loss_node = torch.mean(kl_node) * 100
            log_halfedge_recon = F.log_softmax(pred_halfedge, dim=-1)
            log_edge_post_true = self.edge_transition.q_v_posterior(log_halfedge_0, log_halfedge_t, time_step,
                                                                    batch_halfedge, v0_prob=True)
            log_edge_post_pred = self.edge_transition.q_v_posterior(log_halfedge_recon, log_halfedge_t, time_step,
                                                                    batch_halfedge, v0_prob=True)
            kl_edge = self.edge_transition.compute_v_Lt(log_edge_post_true,
                                                        log_edge_post_pred, log_halfedge_0, t=time_step,
                                                        batch=batch_halfedge)
            loss_edge = torch.mean(kl_edge) * 100
        else:
            loss_node = F.mse_loss(pred_node, h_node_0) * 30
            loss_edge = F.mse_loss(pred_halfedge, h_halfedge_0) * 30

        # total
        loss_total = loss_pos + loss_node + loss_edge + (loss_len if self.bond_len_loss else 0)

        loss_dict = {
            'loss': loss_total,
            'loss_pos': loss_pos,
            'loss_node': loss_node,
            'loss_edge': loss_edge,
        }
        if self.bond_len_loss == True:
            loss_dict['loss_len'] = loss_len
        return loss_dict

    def forward(self, h_node_pert, pos_pert, batch_node,
                h_edge_pert, edge_index, batch_edge, t):
        """
        Predict Mol at step `0` given perturbed Mol at step `t` with hidden dims and time step
        """
        time_embed_node = self.time_emb(t.index_select(0, batch_node))
        h_node_pert = torch.cat([self.node_embedder(h_node_pert), time_embed_node], dim=-1)
        time_embed_edge = self.time_emb(t.index_select(0, batch_edge))
        h_edge_pert = torch.cat([self.edge_embedder(h_edge_pert), time_embed_edge], dim=-1)

        # 2 diffuse to get the updated node embedding and bond embedding
        h_node, pos_node, h_edge = self.denoiser(
            h_node=h_node_pert,
            pos_node=pos_pert,
            h_edge=h_edge_pert,
            edge_index=edge_index,
            node_time=t.index_select(0, batch_node).unsqueeze(-1) / self.num_timesteps,
            edge_time=t.index_select(0, batch_edge).unsqueeze(-1) / self.num_timesteps,
        )

        n_halfedges = h_edge.shape[0] // 2
        pred_node = self.node_decoder(h_node)
        pred_halfedge = self.edge_decoder(h_edge[:n_halfedges] + h_edge[n_halfedges:])
        pred_pos = pos_node

        return {
            'pred_node': pred_node,
            'pred_pos': pred_pos,
            'pred_halfedge': pred_halfedge,
        }  # at step 0

    @torch.no_grad()
    def sample(self, n_graphs, batch_node, halfedge_index, batch_halfedge):

        device = batch_node.device

        n_nodes_all = len(batch_node)
        n_halfedges_all = len(batch_halfedge)

        node_init = self.node_transition.sample_init(n_nodes_all)
        pos_init = self.pos_transition.sample_init([n_nodes_all, 3])
        halfedge_init = self.edge_transition.sample_init(n_halfedges_all)

        if self.categorical_space == 'discrete':
            _, h_node_init, log_node_type = node_init
            _, h_halfedge_init, log_halfedge_type = halfedge_init
        else:
            h_node_init = node_init
            h_halfedge_init = halfedge_init

        node_traj = torch.zeros([self.num_timesteps + 1, n_nodes_all, h_node_init.shape[-1]],
                                dtype=h_node_init.dtype, device=device)
        pos_traj = torch.zeros([self.num_timesteps + 1, n_nodes_all, 3],
                               dtype=pos_init.dtype, device=device)
        halfedge_traj = torch.zeros([self.num_timesteps + 1, n_halfedges_all, h_halfedge_init.shape[-1]],
                                    dtype=h_halfedge_init.dtype, device=device)

        node_traj[0] = h_node_init
        pos_traj[0] = pos_init
        halfedge_traj[0] = h_halfedge_init

        h_node_pert = h_node_init
        pos_pert = pos_init
        h_halfedge_pert = h_halfedge_init

        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)

        for i, step in tqdm(enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long, device=device)
            h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)

            preds = self(
                h_node_pert, pos_pert, batch_node,
                h_edge_pert, edge_index, batch_edge,
                time_step,
            )
            pred_node = preds['pred_node']
            pred_pos = preds['pred_pos']
            pred_halfedge = preds['pred_halfedge']

            pos_prev = self.pos_transition.get_prev_from_recon(
                x_t=pos_pert, x_recon=pred_pos, t=time_step, batch=batch_node)

            if self.categorical_space == 'discrete':
                log_node_recon = F.log_softmax(pred_node, dim=-1)

                with torch.no_grad():
                    prob_node = log_node_recon.exp()
                    entropy_node = -torch.sum(prob_node * log_node_recon, dim=-1)
                    mask_entropy = torch.zeros_like(entropy_node, dtype=torch.bool)
                    unique_batches = batch_node.unique()

                    alpha_bar_t = self.node_transition.get_alpha_bar(
                        t_normalized=time_step / self.num_timesteps)
                    mask_ratios = sin_mask_ratio_adapter(
                        1 - alpha_bar_t,
                        max_deviation=0.4,
                        center=0.1
                    )

                    for mask_ratio, b in zip(mask_ratios, unique_batches):
                        nodes_in_b = (batch_node == b)
                        entropy_b = entropy_node[nodes_in_b]
                        threshold = torch.quantile(entropy_b, 1 - mask_ratio)
                        mask_entropy[nodes_in_b] = entropy_b > threshold

                log_node_type_new = self.node_transition.q_v_posterior(
                    log_node_recon, log_node_type, time_step, batch_node, v0_prob=True)
                log_node_type = torch.where(mask_entropy[:, None], log_node_type_new, log_node_type)

                node_type_prev = log_sample_categorical(log_node_type)
                h_node_prev = self.node_transition.onehot_encode(node_type_prev)

                log_edge_recon = F.log_softmax(pred_halfedge, dim=-1)
                log_halfedge_type = self.edge_transition.q_v_posterior(
                    log_edge_recon, log_halfedge_type, time_step, batch_halfedge, v0_prob=True)
                halfedge_type_prev = log_sample_categorical(log_halfedge_type)
                h_halfedge_prev = self.edge_transition.onehot_encode(halfedge_type_prev)

            else:
                h_node_prev = self.node_transition.get_prev_from_recon(
                    x_t=h_node_pert, x_recon=pred_node, t=time_step, batch=batch_node)
                h_halfedge_prev = self.edge_transition.get_prev_from_recon(
                    x_t=h_halfedge_pert, x_recon=pred_halfedge, t=time_step, batch=batch_halfedge)


            node_traj[i + 1] = h_node_prev
            pos_traj[i + 1] = pos_prev
            halfedge_traj[i + 1] = h_halfedge_prev

            pos_pert = pos_prev
            h_node_pert = h_node_prev
            h_halfedge_pert = h_halfedge_prev

        return {
            'pred': [pred_node, pred_pos, pred_halfedge],
            'traj': [node_traj, pos_traj, halfedge_traj],
        }


def sin_mask_ratio_adapter(beta_t_bar, max_deviation=0.2, center=0.5):
    adjusted = beta_t_bar * torch.pi * 0.5
    sine = torch.sin(adjusted)
    adjustment = sine * max_deviation
    mask_ratio = center + adjustment
    return mask_ratio.squeeze()

