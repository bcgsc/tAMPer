import torch
import esm
import gc
from typing import List, Dict


class ESM2_Embeddings:
    def __init__(self, model_variant: str):

        if model_variant == 't36':
            embedding_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        elif model_variant == 't33':
            embedding_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        elif model_variant == 't30':
            embedding_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        elif model_variant == 't12':
            embedding_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        else:
            embedding_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

        self.embedding_model = embedding_model
        self.alphabet = alphabet
        self.n_layers = int(model_variant[1:])

    def generate_embeddings(self, sequences: List[str], ids: List[str]) -> Dict[str, torch.tensor]:

        tokenizer = self.alphabet.get_batch_converter()
        device = torch.device("cpu")

        self.embedding_model.to(device)
        self.embedding_model.eval()

        sequence_ids = list(zip(ids, sequences))

        ind = 0
        batch_size = 16
        n_seqs = len(sequence_ids)
        embedding_ids = {}

        while ind < n_seqs:
            if ind + batch_size > n_seqs:
                batch_size = n_seqs - ind
            _, _, batch_tokens = tokenizer(sequence_ids[ind:ind + batch_size])
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
            with torch.no_grad():
                batch_tokens = batch_tokens.to(device)
                results = self.embedding_model(batch_tokens, repr_layers=[self.n_layers])
            token_representations = results["representations"][self.n_layers].to(device)

            for i, tokens_len in enumerate(batch_lens):
                embedding_ids[sequence_ids[ind + i][0]] = token_representations[i, 1:tokens_len - 1].cpu()
            ind += batch_size

        del self.embedding_model
        del tokenizer
        gc.collect()

        return embedding_ids


