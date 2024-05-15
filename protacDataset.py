import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data, Batch

LIGAND_ATOM_TYPE = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
FASTA_CHAR = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
              "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
              "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
              "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}
EDGE_ATTR = {'1': 1, '2': 2, '3': 3, 'ar': 4, 'am': 5}


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def mol2graph(path, ATOM_TYPE, geometric=False):
    with open(path) as f:
        lines = f.readlines()
    try:
        atom_end_line = lines.index('@<TRIPOS>UNITY_ATOM_ATTR\n')
    except ValueError:
        atom_end_line = lines.index('@<TRIPOS>BOND\n')

    atom_lines = lines[lines.index('@<TRIPOS>ATOM\n') + 1:atom_end_line]
    bond_lines = lines[lines.index('@<TRIPOS>BOND\n') + 1:]
    atoms = []
    positions = []
    for atom in atom_lines:
        ele = atom.split()[5].split('.')[0]
        atoms.append(ATOM_TYPE.index(ele)
                     if ele in ATOM_TYPE
                     else len(ATOM_TYPE))
        positions.append([eval(atom.split()[2]), eval(atom.split()[3]), eval(atom.split()[4])])
    edge_1 = [int(i.split()[1]) - 1 for i in bond_lines]
    edge_2 = [int(i.split()[2]) - 1 for i in bond_lines]
    edge_attr = [EDGE_ATTR[i.split()[3]] for i in bond_lines]
    x = torch.tensor(atoms)
    edge_idx = torch.tensor([edge_1 + edge_2, edge_2 + edge_1])
    edge_attr = torch.tensor(edge_attr + edge_attr)
    if not geometric:
        positions = torch.tensor(positions)
        tdEdge = to_scipy_sparse_matrix(edge_idx, edge_attr).todense()
        tdEdge = torch.from_numpy(np.array(tdEdge, dtype=np.float32).flatten())
        graph = Data(x=x, pos=positions, edge=tdEdge)
    else:
        graph = Data(x=x, edge_index=edge_idx, edge_attr=edge_attr)
    return graph


class PROTACDataset(Dataset):
    def __init__(self, data_dir, clean_data, geometric=False, sequence = False):
        self.data_dir = data_dir
        self.data = pd.read_csv(clean_data)
        self.geometric = geometric
        self.sequnce = sequence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        compound_id, target_sequence, e3_ligase_sequence, label = row['compound id'], row['target_sequence'], row[
            'e3_ligase_sequence'], row['label']
        self.label = np.int(label)
        warhead_file = self.data_dir + 'warhead_{}.mol2'.format(compound_id)
        ligase_ligand_file = self.data_dir + 'e3_ligand_{}.mol2'.format(compound_id)
        linker_file = self.data_dir + 'linker_{}.mol2'.format(compound_id)

        warhead = mol2graph(warhead_file, LIGAND_ATOM_TYPE, self.geometric)
        ligase_ligand = mol2graph(ligase_ligand_file, LIGAND_ATOM_TYPE, self.geometric)
        linker = mol2graph(linker_file, LIGAND_ATOM_TYPE, self.geometric)
        target_sequence = label_sequence(target_sequence, FASTA_CHAR)
        e3_ligase_sequence = label_sequence(e3_ligase_sequence, FASTA_CHAR)

        sample = {
            "ligase_ligand": ligase_ligand,
            "ligase": e3_ligase_sequence,
            "target_ligand": warhead,
            "target": target_sequence,
            "linker": linker,
            "label": np.int(label),
        }
        return sample


def collater(data_list):
    batch = {}
    ligase_ligand = [x["ligase_ligand"] for x in data_list]
    ligase = [x["ligase"] for x in data_list]
    target_ligand = [x["target_ligand"] for x in data_list]
    target = [x["target"] for x in data_list]
    linker = [x["linker"] for x in data_list]
    label = [x["label"] for x in data_list]

    batch["ligase_ligand"] = Batch.from_data_list(ligase_ligand)
    batch["ligase"] = torch.from_numpy(np.array(ligase))
    batch["target_ligand"] = Batch.from_data_list(target_ligand)
    batch["target"] = torch.from_numpy(np.array(target))
    batch["linker"] = Batch.from_data_list(linker)
    batch["label"] = torch.tensor(label)
    return batch
