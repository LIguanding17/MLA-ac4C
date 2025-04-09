import numpy as np
import torch
import pandas as pd

from itertools import product
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, matthews_corrcoef, roc_curve, precision_recall_curve, auc

onehot_dict = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'U': [0, 0, 0, 1],
    'X': [0, 0, 0, 0]
}

eiip_dict = {
    'A': 0.1260,
    'C': 0.1340,
    'G': 0.0806,
    'U': 0.1335,
    'X': 0.0
}

ncp_dict = {
    'A': [1, 1, 1],
    'C': [0, 1, 0],
    'G': [1, 0, 0],
    'U': [0, 0, 1],
    'X': [0, 0, 0]
}


def get_positional_encoding(length, d_model):
    """
    using standard Transformer (sin/cos)
    length: sequence length
    d_model: encoding dimension (must be the same or uniform as one-hot dimension)
    """
    pos = np.arange(length)[:, np.newaxis]  # shape (length, 1)
    i = np.arange(d_model)[np.newaxis, :]  # shape (1, d_model)
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates

    # apply sin to even indices; cos to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads.reshape(-1)


def get_kmers(sequence, k):
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]


def get_pseknc_vector(sequence, k=3):
    bases = ['A', 'C', 'G', 'U']
    all_kmers = [''.join(p) for p in product(bases, repeat=k)]

    kmers = get_kmers(sequence, k)
    kmer_counts = Counter(kmers)

    # Normalized frequency vector
    total = sum(kmer_counts.values())
    freq_vector = [kmer_counts[kmer] / total if total > 0 else 0 for kmer in all_kmers]

    return np.array(freq_vector)


# All kinds of encoding and splicing of RNA sequences
def encode_sequence(seq):
    # One-Hot encoding
    onehot_encoded = np.array([onehot_dict[nuc] for nuc in seq]).reshape(-1)
    pos_encoding = get_positional_encoding(201, 4)
    feature1 = onehot_encoded + pos_encoding

    # EIIP encoding
    eiip_encoded = np.array([eiip_dict[nuc] for nuc in seq])

    # NCP encoding
    ncp_encoded = np.concatenate([ncp_dict[nuc] for nuc in seq])

    # PseKNC encoding
    pseknc_encoded = get_pseknc_vector(seq)

    lst = [feature1, eiip_encoded, ncp_encoded, pseknc_encoded]
    lst2 = [lst[i] for i in range(len(lst))]
    combined_vector = np.concatenate([lst2[i] for i in range(len(lst2))])
    return combined_vector.tolist()


# Read CSV files and encode each sequence
def process_csv_and_encode(file_path):
    df = pd.read_csv(file_path)

    sequence_length = len(df.iloc[0]['Sequence'])
    mid_position = sequence_length // 2

    def replace_middle(seq):
        return seq[max(0, mid_position - 100):min(mid_position + 101, sequence_length)]

    sequences = df['Sequence'].apply(replace_middle)
    labels = df['Label']

    encoded_data = [encode_sequence(seq) for seq in sequences]

    encoded_data = np.array(encoded_data)
    labels = np.array(labels)

    encoded_data = torch.tensor(encoded_data)
    labels = torch.tensor(labels)
    return encoded_data, labels


def calculate_metrics(y_true, y_pred_prob):
    """
    Calculate and return various classification indicators
    y_true: Actual label (binary label)
    y_pred_prob: Predicted probability (continuous probability)
    return: Dictionary of various classification indicators
    """
    y_true_flat = y_true.ravel()
    y_pred_flat = (y_pred_prob.ravel() > 0.5).astype(int)
    TP = np.sum((y_true_flat == 1) & (y_pred_flat >= 0.5), dtype=np.float64)
    TN = np.sum((y_true_flat == 0) & (y_pred_flat < 0.5), dtype=np.float64)
    FP = np.sum((y_true_flat == 0) & (y_pred_flat >= 0.5), dtype=np.float64)
    FN = np.sum((y_true_flat == 1) & (y_pred_flat < 0.5), dtype=np.float64)
    print(f"TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN}")

    acc = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, average='binary', zero_division=1)
    recall = recall_score(y_true_flat, y_pred_flat, average='binary')
    f1 = f1_score(y_true_flat, y_pred_flat, average='binary')
    mcc = matthews_corrcoef(y_true_flat, y_pred_flat)
    sn = recall
    sp = TN / (FP + TN)
    try:
        auc_value = roc_auc_score(y_true_flat, y_pred_prob.ravel())
    except ValueError as e:
        auc_value = 'Undefined (only one class present)'

    try:
        prc = average_precision_score(y_true_flat, y_pred_prob.ravel())
    except ValueError as e:
        prc = 'Undefined (only one class present)'

    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "AUC": auc_value,
        "PRC": prc,
        "MCC": mcc,
        "SN": sn,
        "SP": sp,
    }
