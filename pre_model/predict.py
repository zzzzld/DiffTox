import torch
import numpy as np
from tox_model import Model
from dataset import smile_to_graph, smile_w2v_pad
from pubchemfp import GetPubChemFPs
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import config


def get_atom_importance_scores(smiles, model_path="output/best_model.pth"):
    # 1. 构建分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无效的SMILES字符串")

    # 2. 计算分子指纹
    fp = []
    fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)  # 167
    fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)  # 441
    fp_pubcfp = GetPubChemFPs(mol)  # 881
    fp_ecfp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp.extend(fp_maccs)
    fp.extend(fp_phaErGfp)
    fp.extend(fp_pubcfp)
    fp.extend(fp_ecfp2)
    fp = np.array(fp)

    # 3. 图结构和特征
    c_size, features, edge_index = smile_to_graph(smiles)
    _, _, smi_embedding_matrix = smile_w2v_pad(smiles, 100, 100)
    token1 = np.array(smi_embedding_matrix)

    graph_data = Data(
        x=torch.Tensor(features),
        edge_index=torch.LongTensor(edge_index).transpose(1, 0)
    )
    graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)

    # 4. 构造模型输入
    padded_smiles_batch = torch.tensor(token1).unsqueeze(0)
    fp_select = torch.tensor(fp, dtype=torch.float).unsqueeze(0)

    # 5. 加载模型
    args = config.parse()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(
        in_feats=84,
        hidden_feats=args.hidden_feats,
        rnn_embed_dim=args.rnn_embed_dim,
        blstm_dim=args.rnn_hidden_dim,
        blstm_layers=args.rnn_layers,
        fp_2_dim=args.fp_dim,
        dropout=args.dropout,
        num_heads=args.head,
        device=device).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.train()  # 归因时需train模式

    # 6. 开启梯度追踪
    graph_data = graph_data.to(device)
    graph_data.x = graph_data.x.detach().clone().requires_grad_()

    # 7. 前向传播
    output, out_graph, _, _ = model(
        padded_smiles_batch.to(device),
        graph_data,
        fp_select.to(device),
        return_node_importance=True
    )

    # 8. 只对图模态输出做反向传播
    out_graph.sum().backward()

    # 9. 获取每个节点的归因分数
    node_importance = graph_data.x.grad.abs().sum(dim=1)
    importance_scores = node_importance.detach().cpu().numpy()

    # 10. 返回分数（可加mol对象）
    return importance_scores
