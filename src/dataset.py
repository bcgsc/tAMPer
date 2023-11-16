import os
import torch
import zipfile
from loguru import logger
from torch_geometric.data import Data, Dataset
from peptideGraph import ProteinGraphBuilder
from utils import read_fasta, merge
import torch.nn.functional as F


class ToxicityData(Dataset):
    def __init__(self, seqs_file: list,
                 pdbs_path: str,
                 max_d: int,
                 device: torch.device,
                 embeddings_dir: str = None):

        # if pdbs_path is equal to None, the corresponding structures should be predicted
        # TODO: a script for predicting structures with COLABFOLD would be needed later on - There you go!

        super().__init__()
        self.graphs = list()

        if len(seqs_file) == 2:
            self.data = merge(seqs_file)
        else:
            self.data = read_fasta(seqs_file[0])

        self.device = device
        self.pdbs_path = pdbs_path
        self.graph_builder = ProteinGraphBuilder(max_distance=max_d,
                                                 max_num_neighbors=128)

        logger.info(f"Loading structures from {pdbs_path}")
        self.add_structures()
        logger.info(f"Number of structures: {len(self.graphs)}")

        logger.info(f"Loading sequence embeddings from {embeddings_dir}")
        self.seq_embeddings = self.load_embeddings(embeddings_dir)

    def load_embeddings(self, embeddings_dir: str = None):

        embedding_ids = {self.graphs[i].id: torch.empty(0) for i in range(len(self.graphs))}

        if embeddings_dir:
            for i in range(len(self.graphs)):
                path = os.path.join(embeddings_dir, f"{self.graphs[i].id}.pt")
                embedding = torch.load(path, map_location=torch.device('cpu'))
                embedding_ids[self.graphs[i].id] = embedding.to(torch.float32)

        return embedding_ids

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        return self.graphs[idx]

    def append(self, graph):
        self.graphs.append(graph)

    def get_ids(self):
        ids = []
        for graph in self.graphs:
            ids.append(graph.id)
        return ids

    def add_structures(self):
        for i in range(len(self.data)):
            self.add_single_structure(i)

    def add_single_structure(self, index: int):

        pdb_dir = f"{self.pdbs_path}/{self.data[index]['id']}"
        file = f'{pdb_dir}.pdb'

        if os.path.isdir(pdb_dir):
            for pdb_file in os.listdir(pdb_dir):
                self.graphs.append(self.graph_builder.build_graph(
                    id=self.data[index]['id'],
                    pdb_file=os.path.join(pdb_dir, pdb_file),
                    amd=self.data[index]['AMD'],
                    label=self.data[index]['label']))

        elif os.path.isfile(file):
            self.graphs.append(self.graph_builder.build_graph(
                id=self.data[index]['id'],
                pdb_file=file,
                amd=self.data[index]['AMD'],
                label=self.data[index]['label']))
        else:
            # the case the pdb files is zipped
            pdbs_zip = os.path.join(self.pdbs_path, f"{self.data[index]['id']}.result.zip")
            with zipfile.ZipFile(pdbs_zip, "r") as zip_ref:
                for file in zip_ref.namelist():
                    if "_relaxed_rank_" in file and file.endswith('.pdb'):
                        pdb_file = zip_ref.extract(file, path=pdb_dir)
                        self.graphs.append(self.graph_builder.build_graph(
                            id=self.data[index]['id'],
                            pdb_file=pdb_file,
                            amd=self.data[index]['AMD'],
                            label=self.data[index]['label']))
        return


class SimpleData(Dataset):
    def __init__(self,
                 pos_seqs: str,
                 neg_seqs: str,
                 graphs_dir: str,
                 embeddings_dir: str, ):

        # if pdbs_path is equal to None, the corresponding structures should be predicted
        # TODO: a script for predicting structures with COLABFOLD would be needed later on - There you go!

        super().__init__()
        self.graphs_dir = graphs_dir
        self.embeddings_dir = embeddings_dir

        self.graphs = list()
        self.data = merge(pos_fasta=pos_seqs, neg_fasta=neg_seqs)

        logger.info(f"Loading structures from {graphs_dir}")
        self.load_structures()
        logger.info(f"Number of structures: {len(self.graphs)}")

        logger.info(f"Loading sequence embeddings from {embeddings_dir}")
        self.load_embeddings()

    def load_embeddings(self):
        seq_embeddings = torch.load(self.embeddings_dir)
        for graph in self.graphs:
            graph.embeddings = seq_embeddings[graph.id]

    def len(self) -> int:
        return len(self.graphs)

    def n_class(self, label: int):
        count = 0
        for seq in self.data:
            if int(seq['label']) == label:
                count += 1
        return count

    def get(self, idx: int) -> Data:
        return self.graphs[idx]

    def load_structures(self):
        graphs_strcutures = torch.load(self.graphs_dir)
        for seq in self.data:
            for graph in graphs_strcutures[seq['id']]:
                self.graphs.append(graph)


class PreData(Dataset):
    def __init__(self,
                 tr_graphs: str,
                 val_graphs: str):

        super().__init__()
        self.graphs = list()

        self.load_structures(tr_graphs)
        self.load_structures(val_graphs)

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        return self.graphs[idx]

    def load_structures(self, dir):
        graphs_strcutures = torch.load(dir)
        IDs = list(graphs_strcutures.keys())
        for id_ in IDs:
            for graph in graphs_strcutures[id_]:
                self.graphs.append(graph)

