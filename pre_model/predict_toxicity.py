from pre_model.dataset import smile_to_graph, smile_w2v_pad
from pre_model.pubchemfp import GetPubChemFPs
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from torch_geometric.data import Data
import torch
import pre_model.config as config

def predict_toxicity(smiles, tox_model, device):
    # 检查 SMILES 有效性
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析SMILES: {smiles}")

    # 分子指纹提取
    # fp = []
    # fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)  # 167 bits
    # fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)  # 441
    # fp_pubcfp = GetPubChemFPs(mol)  # 881
    # fp_ecfp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)  # 1024
    # fp.extend(fp_maccs)
    # fp.extend(fp_phaErGfp)
    # fp.extend(fp_pubcfp)
    # fp.extend(fp_ecfp2)
    # fp = np.array(fp)

    # 图结构和特征
    c_size, features, edge_index = smile_to_graph(smiles)
    smile_, smi_word_index, smi_embedding_matrix = smile_w2v_pad(smiles, 100, 100)
    token1 = np.array(smi_embedding_matrix)

    # 构造图数据
    graph_data = Data(
        x=torch.Tensor(features),
        edge_index=torch.LongTensor(edge_index).transpose(1, 0)
    )
    graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
    padded_smiles_batch = torch.tensor(token1).unsqueeze(0)
    # fp_select = torch.tensor(fp, dtype=torch.float).unsqueeze(0)

    # 加载模型参数
    args = config.parse()
    model = tox_model(
        in_feats=84,
        hidden_feats=args.hidden_feats,
        dropout=args.dropout,
        device=device).to(device)

    model.load_state_dict(torch.load(r".\pre_model\output\best_model.pth", map_location=device))
    model.eval()

    # 推理预测
    with torch.no_grad():
        output, _ = model(
            padded_smiles_batch.to(device),
            graph_data.to(device),
            # fp_select.to(device)
        )
        prob = torch.sigmoid(output).item()

    return prob, None, None