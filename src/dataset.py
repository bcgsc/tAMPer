import os
import torch
import zipfile
from loguru import logger
from torch_geometric.data import Data, Dataset
from peptideGraph import ProteinGraphBuilder
from utils import read_fasta, merge
import torch.nn.functional as F
from embeddings import ESM2_Embeddings


class ToxicityData(Dataset):
    def __init__(self,
                 pos_seqs: str,
                 neg_seqs: str,
                 pdbs_path: str,
                 max_d: int,
                 embedding_model: str,):

        # if pdbs_path is equal to None, the corresponding structures should be predicted
        # TODO: a script for predicting structures with COLABFOLD would be needed later on - There you go!

        super().__init__()
        self.graphs = list()
        self.pdbs_path = pdbs_path
        self.graph_builder = ProteinGraphBuilder(max_distance=max_d,
                                                 max_num_neighbors=128)
        self.esm = ESM2_Embeddings(model_variant=embedding_model)

        self.data = merge(pos_fasta=pos_seqs, neg_fasta=neg_seqs)

        logger.info(f"Loading structures from {pdbs_path}")
        self.add_structures()
        logger.info(f"Number of structures: {len(self.graphs)}")

        logger.info(f"Loading sequence embeddings from {embeddings_dir}")
        self.load_embeddings()

    def load_embeddings(self):
        seqs, ids = [], []
        for record in self.data:
            seqs.append(record['seq'])
            ids.append(record['id'])
        embeddings = self.esm.generate_embeddings(sequences=seqs, ids=ids)
        for graph in self.graphs:
            graph.embeddings = embeddings[graph.id]

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        return self.graphs[idx]

    def add_structures(self):
        for index in range(len(self.data)):
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
