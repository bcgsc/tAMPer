import torch
import torch.nn.functional as F
import math
import Bio.PDB
from Bio.PDB.DSSP import DSSP
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from aminoacids import aa_charge, ss_mapping, aa_1_mapping, aa_polar


class ProteinGraphBuilder:
    def __init__(self, max_distance=10, max_num_neighbors=128):
        self.max_distance = max_distance
        self.radius_graph = RadiusGraph(max_distance,
                                        loop=False,
                                        max_num_neighbors=max_num_neighbors)

    def build_graph(self, id: str,
                    pdb_file: str,
                    amd: int = -1,
                    label: int = -1, ) -> Data:
        parser = Bio.PDB.PDBParser()
        structure = parser.get_structure(id, pdb_file)

        model = next(structure.get_models())
        chain = list(next(model.get_chains()))
        seq_length = len(chain)

        aa = np.full((seq_length, 1), dtype=np.float32, fill_value=np.nan)
        ss = np.full((seq_length, 1), dtype=np.float32, fill_value=np.nan)

        ca_coordinates = np.full((seq_length, 3), dtype=np.float32, fill_value=np.nan)
        c_coordinates = np.full((seq_length, 3), dtype=np.float32, fill_value=np.nan)
        n_coordinates = np.full((seq_length, 3), dtype=np.float32, fill_value=np.nan)

        sequence = ""

        dssp_feats = DSSP(model, pdb_file, dssp='mkdssp')
        aa_keys = list(dssp_feats.keys())

        for idx in range(seq_length):
            feats = dssp_feats[aa_keys[idx]]

            ca_coordinates[idx] = chain[idx]['CA'].get_vector()[:3]
            c_coordinates[idx] = chain[idx]['C'].get_vector()[:3]
            n_coordinates[idx] = chain[idx]['N'].get_vector()[:3]

            sequence += feats[1]
            aa[idx] = aa_1_mapping[feats[1]]
            ss[idx] = ss_mapping[feats[2]]

        forward_vec, backward_vec = orientations(ca_cords=ca_coordinates)

        side_chain_vec = side_chains(origin=ca_coordinates,
                                     n=n_coordinates,
                                     c=c_coordinates)

        angles = dihedral_angles(np.stack([n_coordinates,
                                           ca_coordinates,
                                           c_coordinates], 1))

        node_v = np.concatenate([forward_vec,
                                 backward_vec,
                                 side_chain_vec], axis=1)

        out = Data(
            id=id,
            seq=sequence,
            ss=torch.from_numpy(ss).long(),
            aa=torch.from_numpy(aa).long(),
            amd=torch.tensor(amd).to(torch.float32),
            x=angles.to(torch.float32),
            v=torch.from_numpy(node_v.reshape((node_v.shape[0], node_v.shape[1] // 3, 3))).to(torch.float32),
            y=torch.tensor(label).to(torch.float32),
            pos=torch.from_numpy(ca_coordinates).to(torch.float32))

        self.radius_graph(out)
        self.add_edge_feats(out)

        return out

    @staticmethod
    def add_edge_feats(data: Data):
        edge_vec = data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]]
        rbf_feats = rbf(edge_vec.norm(dim=-1), D_count=16)
        pos_feats = positional_embeddings(data.edge_index, num_embeddings=16)
        data.edge_s = torch.cat([rbf_feats, pos_feats], dim=1)
        data.edge_v = normalize(edge_vec).view(edge_vec.size(0), edge_vec.size(1) // 3, 3)

        return data


class RadiusGraph(object):
    def __init__(self, r, loop=False, max_num_neighbors=32):
        self.r = r
        self.loop = loop
        self.max_num_neighbors = max_num_neighbors

    def __call__(self, data: Data) -> Data:
        pos = data.pos.clone()
        batch = data.batch if "batch" in data else None
        edge_index = radius_graph(
            pos, self.r, batch, self.loop, self.max_num_neighbors,
        )
        data.edge_index = edge_index

        return data


def normalize(ndarray, dim=-1):
    return np.divide(ndarray, np.linalg.norm(ndarray, axis=dim, keepdims=True))


def orientations(ca_cords):
    forward = normalize(ca_cords[1:] - ca_cords[:-1])
    forward = np.vstack([forward, np.zeros((3,))])
    backward = normalize(ca_cords[:-1] - ca_cords[1:])
    backward = np.vstack([np.zeros((3,)), backward])
    return forward, backward


def dihedral_angles(cords, eps=1e-7):
    X = torch.reshape(torch.from_numpy(cords), [3 * cords.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def side_chains(origin, n, c):
    c, n = normalize(c - origin), normalize(n - origin)
    bisector = normalize(c + n)
    perp = normalize(np.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def rbf(D, D_min=0., D_max=20., D_count=16):
    D_mu = torch.linspace(D_min, D_max, D_count)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def positional_embeddings(edge_index,
                          num_embeddings=None):
    d = edge_index[0] - edge_index[1]
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    pos_encoding = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return pos_encoding