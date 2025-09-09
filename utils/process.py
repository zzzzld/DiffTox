import sys
import shutil
import argparse
import itertools
import json
import pickle
sys.path.append('.')
# import torch.utils.tensorboard

from models.model import DiffTox
from utils.sample import seperate_outputs
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAMILY_MAPPING = {'Donor': 1, 'Acceptor': 2}

def process_files(sdf_file, atomic_numbers):
    # Assuming getPharamacophoreCoords and FAMILY_MAPPING are defined elsewhere
    # def getPharamacophoreCoords(mol): ...
    # FAMILY_MAPPING = {...}

    # Process the SDF file to get ligand information
    mol = Chem.SDMolSupplier(sdf_file)[0]
    if mol is None:
        raise ValueError("Could not read molecule from SDF file")


    ele_to_nodetype = {ele: i for i, ele in enumerate(atomic_numbers)}
    ele_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        ele = atom.GetAtomicNum()
        ele_list.append(ele)
    node_type = np.array([ele_to_nodetype[ele] for ele in ele_list])
    positions = mol.GetConformer().GetPositions()

    return {
        'node_type': node_type,
        'positions': positions,
    }
