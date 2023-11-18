import torch
from torch.nn.utils.rnn import (
    pad_packed_sequence,
    pack_padded_sequence
)
from torch.nn import (
    Module,
    MultiheadAttention,
    LayerNorm,
    GRU,
    Linear,
    Dropout3d,
    Dropout
)
from torch_geometric.utils import to_dense_batch
import torch_geometric.nn as gnn
from GConvs import GVP_GNN


class tAMPer(Module):

    def __init__(self,
                 input_modality: str,
                 node_dims: tuple,
                 edge_dims: tuple,
                 node_hdim: tuple,
                 edge_hdim: tuple,
                 seq_input_dim: int,
                 gru_hdim: int,
                 n_heads: int,
                 n_grus: int,
                 n_gnns: int,
                 n_tx: int = 1,
                 n_ss: int = 8):

        super().__init__()

        self.modal = input_modality
        self.GNN = GVP_GNN(node_in_dim=node_dims,
                           node_h_dim=node_hdim,
                           edge_in_dim=edge_dims,
                           edge_h_dim=edge_hdim,
                           num_layers=n_gnns,
                           drop_rate=0.0)

        self.GRU = GRU(input_size=seq_input_dim,
                       hidden_size=int(gru_hdim / 2),
                       num_layers=n_grus,
                       bidirectional=True,
                       batch_first=True)

        # * 2 for bidirectional
        gnn_dim = node_hdim[0]
        att_dim = gnn_dim + gru_hdim

        self.MHAttention = MultiheadAttention(
            embed_dim=att_dim if input_modality == 'all' else gnn_dim,
            num_heads=n_heads,
            bias=True,
            batch_first=True)

        self.dropout = torch.nn.ModuleDict({
            'seq': Dropout3d(),
            'graph': Dropout3d(),
            'out': Dropout()})

        self.LayerNorm = torch.nn.ModuleDict({
            'seq': LayerNorm(normalized_shape=gru_hdim),
            'graph': LayerNorm(normalized_shape=gnn_dim),
            'att': LayerNorm(normalized_shape=att_dim if input_modality == 'all' else gnn_dim)
        })
        # +1 is added for the C_terminal amidation modification
        self.tox_fc = Linear(att_dim + 1 if input_modality == 'all' else gnn_dim + 1, n_tx)
        self.ss_fc = Linear(att_dim if input_modality == 'all' else gnn_dim, n_ss)

    def forward(self, sequences, graphs):

        if self.modal != 'structure':
            # sequence processing
            _, lengths = torch.unique(graphs.batch, return_counts=True)
            dense_batch, mask = to_dense_batch(x=sequences,
                                               batch=graphs.batch)

            pack_sequence = pack_padded_sequence(dense_batch,
                                                 lengths=lengths.to(torch.int64).cpu(),
                                                 batch_first=True,
                                                 enforce_sorted=False)

            packed_output, _ = self.GRU(pack_sequence)
            h_seq, _ = pad_packed_sequence(packed_output, batch_first=True)
            h_seq = self.LayerNorm['seq'](h_seq).unsqueeze(3)
            h_seq = self.dropout['seq'](h_seq).squeeze(3)

        if self.modal != 'sequence':
            # structure processing
            strct_feats = self.GNN(h_V=(graphs.x, graphs.v),
                                   edge_index=graphs.edge_index,
                                   h_E=(graphs.edge_s, graphs.edge_v))

            h_strct, mask = to_dense_batch(x=strct_feats, batch=graphs.batch)
            h_strct = self.LayerNorm['graph'](h_strct)
            h_strct = self.dropout['graph'](h_strct.unsqueeze(3)).squeeze(3)

        # peptide feature vector
        if self.modal == 'sequence':
            h_pep = h_seq
        elif self.modal == 'structure':
            h_pep = h_strct
        else:
            h_pep = torch.cat([h_seq, h_strct], dim=2)

        h_pep, att_weights = self.MHAttention(h_pep, h_pep, h_pep, key_padding_mask=~mask)
        h_pep = self.LayerNorm['att'](h_pep)

        h_pep_mean = gnn.global_mean_pool(h_pep[mask], batch=graphs.batch)
        h_pep_mean = self.dropout['out'](h_pep_mean)
        # toxicity
        h_pep_mean = torch.cat([h_pep_mean, graphs.amd.unsqueeze(1)], dim=1)

        out = {'tx': self.tox_fc(h_pep_mean),
               'ss': self.ss_fc(h_pep[mask]),
               'att_weights': att_weights}
        return out


class preGNNs(Module):

    def __init__(self,
                 node_dims: tuple,
                 edge_dims: tuple,
                 node_hdim: tuple,
                 edge_hdim: tuple,
                 n_gnns: int,
                 n_aa: int = 20):
        super().__init__()

        self.GNNs = GVP_GNN(node_in_dim=node_dims,
                            node_h_dim=node_hdim,
                            edge_in_dim=edge_dims,
                            edge_h_dim=edge_hdim,
                            num_layers=n_gnns,
                            drop_rate=0.0)

        self.fc = Linear(node_hdim[0], n_aa)

    def forward(self, graphs):
        h_strct = self.GNNs(h_V=(graphs.x, graphs.v),
                            edge_index=graphs.edge_index,
                            h_E=(graphs.edge_s, graphs.edge_v))

        out = self.fc(h_strct)

        return out