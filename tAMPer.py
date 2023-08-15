import torch
from torch.nn.utils.rnn import (
    pad_packed_sequence,
    pack_padded_sequence
)
from torch.nn import (
    Module,
    MultiheadAttention,
    LeakyReLU,
    LayerNorm,
    GRU,
    Linear,
    Dropout3d,
    Dropout
)
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
import torch_geometric.nn as gnn
from GConvs import GVP_GNN


class tAMPer(Module):

    def __init__(self,
                 input_modality: str,
                 node_dims: tuple,
                 edge_dims: tuple,
                 node_h_dim: tuple,
                 edge_h_dim: tuple,
                 seq_input_dim: int,
                 gru_hidden_dim: int,
                 gru_layers: int,
                 douts: dict,
                 num_gnn_layers: int,
                 tox_classes: int = 1,
                 ss_classes: int = 8):

        super().__init__()

        self.modal = input_modality
        self.GNN = GVP_GNN(node_in_dim=node_dims,
                           node_h_dim=node_h_dim,
                           edge_in_dim=edge_dims,
                           edge_h_dim=edge_h_dim,
                           num_layers=num_gnn_layers,
                           drop_rate=0.0)

        self.GRU = GRU(input_size=seq_input_dim,
                       hidden_size=gru_hidden_dim,
                       num_layers=gru_layers,
                       bidirectional=True,
                       batch_first=True)

        # * 2 for bidirectional
        gnn_dim = node_h_dim[0]
        att_dim = gnn_dim + 2 * gru_hidden_dim

        self.MHAttention = MultiheadAttention(
            embed_dim=gnn_dim + 2 * gru_hidden_dim if input_modality == 'all' else gnn_dim,
            num_heads=8,
            bias=True,
            batch_first=True)

        self.dropout = torch.nn.ModuleDict({
            'seq': Dropout3d(p=douts['seq']),
            'graph': Dropout3d(p=douts['strct']),
            'out': Dropout()
        })

        self.LayerNorm = LayerNorm(normalized_shape=2 * gru_hidden_dim)
        # +1 is added for the C_terminal amidation modification
        self.act = LeakyReLU()
        self.tox_fc = Linear(att_dim if input_modality == 'all' else gnn_dim, tox_classes)
        self.ss_fc = Linear(att_dim if input_modality == 'all' else gnn_dim, ss_classes)

    def forward(self, sequences, graphs):

        if self.modal != 'structure':
            # sequence processing
            idx, lengths = torch.unique(graphs.batch, return_counts=True)
            dense_batch, mask = to_dense_batch(x=sequences,
                                               batch=graphs.batch)

            pack_sequence = pack_padded_sequence(dense_batch,
                                                 lengths=lengths.to(torch.int64).cpu(),
                                                 batch_first=True,
                                                 enforce_sorted=False)

            packed_output, _ = self.GRU(pack_sequence)
            res_embedding, _ = pad_packed_sequence(packed_output, batch_first=True)
            res_embedding = self.act(self.LayerNorm(res_embedding).unsqueeze(3))
            h_seq = self.dropout['seq'](res_embedding).squeeze()

        if self.modal != 'sequence':
            # structure processing
            strct_feats = self.GNN(h_V=(graphs.x, graphs.v),
                                   edge_index=graphs.edge_index,
                                   h_E=(graphs.edge_s, graphs.edge_v))

            h_strct, mask = to_dense_batch(x=strct_feats, batch=graphs.batch)
            h_strct = self.dropout['graph'](h_strct.unsqueeze(3)).squeeze()

        # peptide feature vector
        if self.modal == 'sequence':
            h_pep = h_seq
        elif self.modal == 'structure':
            h_pep = h_strct
        else:
            h_pep = torch.cat([h_seq, h_strct], dim=2)

        h_pep, weights = self.MHAttention(h_pep, h_pep, h_pep, key_padding_mask=~mask)
        h_pep_mean = gnn.global_mean_pool(h_pep[mask], batch=graphs.batch)
        h_pep_mean = self.dropout['out'](h_pep_mean)
        # toxicity
        # h_pep_mean = torch.cat([h_pep_mean, graphs.amd.unsqueeze(1)], dim=1)

        out = {'tx': self.tox_fc(h_pep_mean),
               'ss': self.ss_fc(h_pep[mask]),
               'weights': weights}

        return out
