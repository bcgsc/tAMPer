aa_1_mapping = {
    # 20 common amino acids
    'A': 0,
    'R': 1,
    'N': 2,
    'D': 3,
    'C': 4,
    'Q': 5,
    'E': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'L': 10,
    'K': 11,
    'M': 12,
    'F': 13,
    'P': 14,
    'S': 15,
    'T': 16,
    'W': 17,
    'Y': 18,
    'V': 19
}

aa_1_mapping_inv = [
    'A',
    'R',
    'N',
    'D',
    'C',
    'Q',
    'E',
    'G',
    'H',
    'I',
    'L',
    'K',
    'M',
    'F',
    'P',
    'S',
    'T',
    'W',
    'Y',
    'V',
]

aa_3_mapping = {
    # 20 common amino acids
    'ALA': 0,
    'ARG': 1,
    'ASN': 2,
    'ASP': 3,
    'CYS': 4,
    'GLN': 5,
    'GLU': 6,
    'GLY': 7,
    'HIS': 8,
    'ILE': 9,
    'LEU': 10,
    'LYS': 11,
    'MET': 12,
    'PHE': 13,
    'PRO': 14,
    'SER': 15,
    'THR': 16,
    'TRP': 17,
    'TYR': 18,
    'VAL': 19
}

aa_3_mapping_inv = [
    'ALA',
    'ARG',
    'ASN',
    'ASP',
    'CYS',
    'GLN',
    'GLU',
    'GLY',
    'HIS',
    'ILE',
    'LEU',
    'LYS',
    'MET',
    'PHE',
    'PRO',
    'SER',
    'THR',
    'TRP',
    'TYR',
    'VAL',
]

ss_mapping = {
    "G": 0,
    "H": 1,
    "I": 2,
    "T": 3,
    "E": 4,
    "B": 5,
    "S": 6,
    "-": 7
}
# tensor([[0.0000],
#         [0.7419],
#         [0.1613],
#         [0.0000],
#         [0.0000],
#         [0.0000],
#         [0.0000],
#         [0.0968]])


aa_charge = {
    # 20 common amino acids
    'A': 0,
    'R': 1,
    'N': 0,
    'D': -1,
    'C': 0,
    'Q': 0,
    'E': -1,
    'G': 0,
    'H': 1,
    'I': 0,
    'L': 0,
    'K': 1,
    'M': 0,
    'F': 0,
    'P': 0,
    'S': 0,
    'T': 0,
    'W': 0,
    'Y': 0,
    'V': 0
}

aa_hydropathy_score = {
    'A': 1.8,
    'R': -4.5,
    'N': -3.5,
    'D': -3.5,
    'C': 2.5,
    'G': -0.4,
    'Q': -3.5,
    'E': -3.5,
    'H': -3.2,
    'I': 4.5,
    'L': 3.8,
    'K': -3.9,
    'M': 1.9,
    'F': 2.8,
    'P': -1.6,
    'S': -0.8,
    'T': -0.7,
    'W': -0.9,
    'Y': -1.3,
    'V': 4.2
}

boman_scores = {
    'L': 4.92,
    'I': 4.92,
    'V': 4.04,
    'F': 2.98,
    'M': 2.35,
    'W': 2.33,
    'A': 1.81,
    'C': 1.28,
    'G': 0.94,
    'Y': -0.14,
    'T': -2.57,
    'S': -3.40,
    'H': -4.66,
    'Q': -5.54,
    'K': -5.55,
    'N': -6.64,
    'E': -6.81,
    'D': -8.72,
    'R': -14.92,
    'P': 0.0
}