import os
import torch
import zipfile
from loguru import logger
from torch_geometric.data import Data, Dataset
from peptideGraph import ProteinGraphBuilder
from utils import read_fasta, merge
import torch.nn.functional as F
from embeddings import ESM2_Embeddings
from ESMfolding import esm2fold
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class ToxicityData(Dataset):
    def __init__(self,
                 seqs: str = None,
                 pos_seqs: str = None,
                 neg_seqs: str = None,
                 pdbs_path: str = None,
                 max_d: int = 12,
                 device: torch.device = torch.device('cpu'),
                 prediction_tool: str = 'ColabFold',
                 embedding_model: str = 't12',) -> None:

        super().__init__()
        self.graphs = list()
        self.pdbs_path = pdbs_path
        self.graph_builder = ProteinGraphBuilder(max_distance=max_d,
                                                 max_num_neighbors=128)
        self.esm = ESM2_Embeddings(model_variant=embedding_model)

        if pos_seqs and neg_seqs:
            self.data = merge(pos_fasta=pos_seqs, neg_fasta=neg_seqs)
        else:
            self.data = read_fasta(fasta_file=seqs)

        logger.info(f"Loading structures from {pdbs_path}")

        self.add_structures(tool=prediction_tool,
                            device=device)
        
        logger.info(f"Number of structures: {len(self.graphs)}")

        logger.info(f"Generating sequence embeddings")
        self.load_embeddings()

    def load_embeddings(self,):

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

    def add_structures(self,
                       tool: str = 'ColabFold',
                       device: torch.device = torch.device('cpu')) -> None:

        if tool == 'ColabFold':

            for index in range(len(self.data)):

                pdbs_zip = os.path.join(self.pdbs_path, f"{self.data[index]['id']}.result.zip")

                with zipfile.ZipFile(pdbs_zip, "r") as zip_ref:
                    for file in zip_ref.namelist():
                        if "_relaxed_" in file and file.endswith('.pdb'):

                            pdb_file = zip_ref.extract(file, path=self.pdbs_path)
                            self.graphs.append(self.graph_builder.build_graph(
                                id=self.data[index]['id'],
                                pdb_file=pdb_file,
                                amd=self.data[index]['AMD'],
                                label=self.data[index]['label']))
        else:
            esm2fold(my_data=self.data, result_file=self.pdbs_path, device=device)
            
            for index in range(len(self.data)):
                pdb_file = os.path.join(self.pdbs_path, f"{self.data[index]['id']}.pdb")
                self.graphs.append(self.graph_builder.build_graph(
                                   id=self.data[index]['id'],
                                   pdb_file=pdb_file,
                                   amd=self.data[index]['AMD'],
                                   label=self.data[index]['label']))

