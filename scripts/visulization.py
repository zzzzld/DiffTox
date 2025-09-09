import gradio as gr
import torch
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import pandas as pd
import os

from models.model import DiffTox
from utils.transforms import *
from utils.transforms import FeaturizeMol
from utils.sample import seperate_outputs
from utils.reconstruct import reconstruct_from_generated_with_edges, MolReconsError
from pre_model.predict_toxicity import predict_toxicity
from pre_model.tox_model import Model
from utils.misc import load_config

# ----------------------
# Parameters and Device
# ----------------------
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
config_path = './configs/sample/DiffTox.yml'
config = load_config(config_path)

# ----------------------
# Load checkpoint
# ----------------------
ckpt = torch.load(config.model.checkpoint, map_location=device)
train_config = ckpt['config']

# ----------------------
# Featurizer
# ----------------------
featurizer = FeaturizeMol(
    train_config.chem.atomic_numbers,
    train_config.chem.mol_bond_types,
    use_mask_node=train_config.transform.use_mask_node,
    use_mask_edge=train_config.transform.use_mask_edge
)

# ----------------------
# MolDiff
# ----------------------
model = DiffTox(
    config=train_config.model,
    num_node_types=featurizer.num_node_types,
    num_edge_types=featurizer.num_edge_types
).to(device)
model.load_state_dict(ckpt['model'])
model.eval()

# Toxicity model
# ----------------------
tox_model = Model(
    in_feats=84,
    hidden_feats=[64, 384],
    dropout=0.6,
    device=device
).to(device)


# ----------------------
# Generate simple HTML table (no color gradient)
# ----------------------
def create_table(df):
    html = "<table style='border-collapse: collapse; width: 100%;'>"
    html += "<tr><th style='border: 1px solid black;'>SMILES</th>"
    html += "<th style='border: 1px solid black;'>Toxicity Probability</th></tr>"
    for _, row in df.iterrows():
        html += f"<tr><td style='border:1px solid black; padding:5px;'>{row['SMILES']}</td>"
        html += f"<td style='border:1px solid black; padding:5px;'>{row['Toxicity Probability']:.2f}</td></tr>"
    html += "</table>"
    return html


# ----------------------
# Callback function
# ----------------------
def generate_molecules(num_mols: int):
    pool = {'finished': [], 'failed': []}

    n_graphs = max(16, num_mols)
    batch_holder = make_data_placeholder(n_graphs=n_graphs, device=device)
    batch_node = batch_holder['batch_node']
    halfedge_index = batch_holder['halfedge_index']
    batch_halfedge = batch_holder['batch_halfedge']

    with torch.no_grad():
        outputs = model.sample(
            n_graphs=n_graphs,
            batch_node=batch_node,
            halfedge_index=halfedge_index,
            batch_halfedge=batch_halfedge,
        )

    safe_outputs = {}
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                safe_outputs[key] = [value.unsqueeze(0).cpu().numpy()]
            else:
                safe_outputs[key] = [v.cpu().numpy() for v in value]
        elif isinstance(value, list):
            safe_outputs[key] = [v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in value]
        else:
            safe_outputs[key] = value
    outputs = safe_outputs

    try:
        output_list = seperate_outputs(
            outputs, n_graphs,
            batch_node.cpu().numpy(),
            halfedge_index.cpu().numpy(),
            batch_halfedge.cpu().numpy()
        )
    except Exception as e:
        print("Decoding failed:", e)
        return "", None, None, None

    for output_mol in output_list:
        mol_info = featurizer.decode_output(
            pred_node=output_mol['pred'][0],
            pred_pos=output_mol['pred'][1],
            pred_halfedge=output_mol['pred'][2],
            halfedge_index=output_mol['halfedge_index'],
        )
        try:
            rdmol = reconstruct_from_generated_with_edges(mol_info)
        except MolReconsError:
            continue

        smiles = Chem.MolToSmiles(rdmol)
        prob, _, _ = predict_toxicity(smiles, Model, device)

        if prob is not None and prob <= 0.8:
            pool['finished'].append({
                'SMILES': smiles,
                'Toxicity Probability': float(prob)
            })

        if len(pool['finished']) >= num_mols:
            break

    results = sorted(pool['finished'], key=lambda x: x['Toxicity Probability'])
    df = pd.DataFrame(results)

    # Bar chart
    colors = ['green' if p <= 0.3 else 'yellow' if p <= 0.6 else 'red' for p in df['Toxicity Probability']]
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(df)), df['Toxicity Probability'], color='blue')
    plt.xticks(range(len(df)), [f'Mol{i + 1}' for i in range(len(df))], rotation=45)
    plt.ylabel('Toxicity Probability')
    plt.title('Toxicity Probability Distribution')
    plt.tight_layout()

    # Molecule image with toxicity labels
    mols = [Chem.MolFromSmiles(s) for s in df['SMILES'][:5]]
    legends = [f"{p:.2f}" for p in df['Toxicity Probability'][:5]]
    mol_img = Draw.MolsToImage(mols, molsPerRow=min(5, len(mols)), subImgSize=(150, 150), legends=legends)

    # CSV
    csv_path = "generated_molecules.csv"
    df.to_csv(csv_path, index=False)

    return create_table(df), plt, mol_img, csv_path


# ----------------------
# Gradio Interface (Button right, chart below)
# ----------------------
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='color:darkblue;'>DiffTox Molecule Generation Platform</h1>")
    gr.Markdown("Select the number of molecules to generate and output SMILES along with toxicity prediction results.")

    # Top row: slider + button
    with gr.Row():
        num_input = gr.Slider(1, 50, step=1, label="Number of molecules", scale=4)
        run_btn = gr.Button("Generate Molecules", variant="primary", scale=1)

    # Result display row
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Result Table")
            output_df = gr.HTML()
        with gr.Column(scale=1):
            gr.Markdown("### Toxicity Bar Chart")
            output_plot = gr.Plot(show_label=False)
        with gr.Column(scale=1):
            gr.Markdown("### Molecule Structures (Top 5)")
            output_img = gr.Image(type="pil", show_label=False)

    # Download CSV row
    with gr.Row():
        output_file = gr.File(label="Download CSV")

    # Button click
    run_btn.click(fn=generate_molecules, inputs=num_input,
                  outputs=[output_df, output_plot, output_img, output_file])

demo.launch()
