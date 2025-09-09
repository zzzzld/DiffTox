import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import warnings
import pickle

from tox_model import Model
from utils_tox import set_random_seed, evaluate
from dataset import dataprocess, collate
from sklearn.model_selection import train_test_split
from rdkit.Chem.Draw import rdMolDraw2D
import config
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem
from torch_geometric.data import Data
from PIL import Image
import io
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def train(args, train_loader, val_loader, model, loss_func, optimizer, scheduler):
    best_val_auc = -np.inf
    patience_counter = 0

    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        all_metrics = []

        for batch in train_loader:
            batch = batch.to(args.device)
            labels = batch.labels

            # 修改这里：模型现在返回两个值（预测结果和对齐损失）
            # preds, alignment_loss = model(batch)
            preds, alignment_loss = model(batch.padded_smiles_batch, batch)
            # 计算总损失：分类损失 + 对齐损失
            loss = loss_func(preds, labels.unsqueeze(1)) + 0.01 * alignment_loss
            # loss = loss_func(preds, labels) + 0.1 * alignment_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            metrics = evaluate(labels.unsqueeze(1).cpu(), preds.detach().cpu())
            all_metrics.append(metrics)

        mean_metrics = np.mean(all_metrics, axis=0)
        print(f"[Epoch {epoch}] Loss: {total_loss:.4f}  |  AUC: {mean_metrics[6]:.4f}  AUPRC: {mean_metrics[7]:.4f}")

        # 验证集评估
        val_metrics = evaluate_loader(val_loader, model, args.device)
        val_auc = val_metrics['AUC']
        print(f"Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

def evaluate_loader(loader, model, device):
    model.eval()
    total_preds = []
    total_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds, _ = model(batch.padded_smiles_batch, batch)
            total_preds.append(preds.cpu())
            total_labels.append(batch.labels.unsqueeze(1).cpu())

    total_preds = torch.cat(total_preds, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    AC, F1, SN, SP, CCR, MCC, AUC, AUPRC = evaluate(total_labels, total_preds)
    return {'AC': AC, 'F1': F1, 'SN': SN, 'SP': SP, 'CCR': CCR, 'MCC': MCC, 'AUC': AUC, 'AUPRC': AUPRC}


def importance_to_color(score):
    """
    将 0~1 的分数映射为红-白渐变色。
    score = 0 -> 白色；score = 1 -> 红色
    """
    return (1.0, 1.0 - score, 1.0 - score)  # R:1, G&B随分数降低

def visualize_mol_importance(mol, importance_scores, save_path=None):
    scores = np.array(importance_scores)
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # 原子颜色映射（高亮颜色）
    atom_colors = {i: importance_to_color(score) for i, score in enumerate(norm_scores)}

    # 键颜色根据两个原子的平均重要性
    highlight_bonds = []
    bond_colors = {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        avg_score = (norm_scores[a1] + norm_scores[a2]) / 2
        bond_idx = bond.GetIdx()
        highlight_bonds.append(bond_idx)
        bond_colors[bond_idx] = importance_to_color(avg_score)

    # RDKit绘图
    drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
    draw_options = drawer.drawOptions()
    draw_options.addAtomIndices = True
    draw_options.useDefaultAtomPalette = False
    draw_options.comicMode = False  # ✅ 确保启用 HTML 渲染

    # 设置 atomLabels，控制显示和颜色
    black = (0.0, 0.0, 0.0)
    draw_options.atomPalette = {
        7: black,  # N
        8: black,  # O
        9: black,  # F
        16: black,  # S
        17: black,  # Cl
        # 你还可以继续添加其他杂原子
    }

    # 设置标签：碳不显示，杂原子显式设置（才能使用 palette 颜色）
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() == 6:
            draw_options.atomLabels[idx] = ''
        else:
            draw_options.atomLabels[idx] = atom.GetSymbol()

    # 画分子图
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors
    )
    drawer.FinishDrawing()
    mol_img = Image.open(io.BytesIO(drawer.GetDrawingText()))

    # matplotlib 画颜色图注
    fig, ax = plt.subplots(figsize=(4, 0.4))
    fig.subplots_adjust(bottom=0.5)

    cmap = plt.cm.Reds
    norm = plt.Normalize(vmin=0, vmax=1)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal'
    )
    cb.set_label('Importance Score')

    # 保存图注为图像
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, transparent=True)
    plt.close(fig)
    buf.seek(0)
    legend_img = Image.open(buf)

    # 拼接分子图 + 图注
    total_height = mol_img.height + legend_img.height
    total_img = Image.new('RGBA', (max(mol_img.width, legend_img.width), total_height), (255, 255, 255, 255))
    total_img.paste(mol_img, (0, 0))
    total_img.paste(legend_img, (0, mol_img.height))

    # 保存或展示
    if save_path:
        total_img.save(save_path)
        print(f"[INFO] 图像已保存至 {save_path}")
    else:
        total_img.show()



def main(args):
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    set_random_seed(args.seed)

    output_dir = './output_images'
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

    source = 'neph'
    data_path = f'./data/{source}/data_neph_dev_ori.csv'

    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    train_idx = train_df.index
    test_idx = test_df.index

    dataset = dataprocess(data_path, './data/neph/dataset_2.pkl')

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = Model(
        in_feats=84,
        hidden_feats=args.hidden_feats,
        dropout=args.dropout,
        device=args.device
    ).to(args.device)

    # model = Model().to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    loss_func = torch.nn.BCEWithLogitsLoss()
    # loss_func = {'reg': torch.nn.MSELoss(reduction="none")}

    print(">>> Start Training ...")
    train(args, train_loader, test_loader, model, loss_func, optimizer, scheduler)

    print(">>> Evaluate on test set ...")
    model.load_state_dict(torch.load(os.path.join(args.output, 'best_model.pth')))
    test_metrics = evaluate_loader(test_loader, model, args.device)

    print("\n[Test Set Results]")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\n>>> Analyzing node importance...")
    model.eval()

    sample_idx = 53
    sample_data = test_dataset[sample_idx]

    padded_smiles_batch = torch.tensor(sample_data.token).unsqueeze(0).to(args.device)
    # fp_select = torch.tensor(sample_data.fp, dtype=torch.float).unsqueeze(0).to(args.device)

    graph_data = Data(
        x=sample_data.x.to(args.device),
        edge_index=sample_data.edge_index.to(args.device)
    )
    graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long, device=args.device)

    # 开启节点特征梯度追踪
    graph_data.x = graph_data.x.detach().clone().requires_grad_()

    # 临时切换到训练模式避免 RNN backward 错误
    model.train()

    # forward，返回融合输出和三个模态输出
    output, out_graph = model(padded_smiles_batch, graph_data, return_node_importance=True)

    # 只对图模态输出做反向传播
    out_graph.sum().backward()

    node_importance = graph_data.x.grad.abs().sum(dim=1)
    importance_scores = node_importance.detach().cpu().numpy()

    model.eval()

    mol = Chem.MolFromSmiles(sample_data.smiles)
    save_path = os.path.join(output_dir, 'node_importance.png')
    visualize_mol_importance(mol, importance_scores, save_path=save_path)

    print(f"Node importance visualization saved to: {save_path}")

if __name__ == '__main__':
    args = config.parse()
    main(args)
