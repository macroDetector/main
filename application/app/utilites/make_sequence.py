import numpy as np

def make_seq(data:np.ndarray, seq_len:int, stride:int) -> np.array:
    # ==== Seq 작업 ====
    seq_len = seq_len
    seq_stride = stride
    sequences = []

    for j in range(0, len(data) - seq_len + 1, seq_stride):
        seq = data[j : j + seq_len]
        sequences.append(seq)

    return np.array(sequences) 

