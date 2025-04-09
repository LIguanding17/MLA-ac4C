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
    'X': [0, 0, 0, 0]  # 不确定碱基
}

eiip_dict = {
    'A': 0.1260,
    'C': 0.1340,
    'G': 0.0806,
    'U': 0.1335,
    'X': 0.0
}

# NCP编码的字典
ncp_dict = {
    'A': [1, 1, 1],
    'C': [0, 1, 0],
    'G': [1, 0, 0],
    'U': [0, 0, 1],
    'X': [0, 0, 0]
}


def get_positional_encoding(length, d_model):
    """
    使用标准Transformer的位置编码（sin/cos）
    length: 序列长度
    d_model: 编码维度（必须和one-hot维度相同或统一）
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

    sequence = ''.join([b if b in bases else 'N' for b in sequence])  # 清理非法字符

    kmers = get_kmers(sequence, k)
    kmer_counts = Counter(kmers)

    # 归一化频率向量
    total = sum(kmer_counts.values())
    freq_vector = [kmer_counts[kmer] / total if total > 0 else 0 for kmer in all_kmers]

    return np.array(freq_vector)


# DNA序列进行各类编码并拼接
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


# 读取CSV文件并对每个序列进行编码
def process_csv_and_encode(file_path):
    df = pd.read_csv(file_path)

    sequence_length = len(df.iloc[0]['Sequence'])
    mid_position = sequence_length // 2  # 中间位置

    def replace_middle(seq):
        return seq[mid_position - 100:mid_position + 100 + 1]

    sequences = df['Sequence'].apply(replace_middle)
    labels = df['Label']

    # 对每个序列进行编码
    encoded_data = [encode_sequence(seq) for seq in sequences]

    # 将结果转换为一个 numpy ndarray
    encoded_data = np.array(encoded_data)  # 合并为单一的 numpy.ndarray
    labels = np.array(labels)  # 合并为单一的 numpy.ndarray

    # 将 numpy ndarray 转换为 PyTorch tensor
    encoded_data = torch.tensor(encoded_data)
    labels = torch.tensor(labels)
    return encoded_data, labels


def calculate_metrics(y_true, y_pred_prob):
    """
    计算并返回各种分类指标
    :param y_true: 实际标签 (二值标签)
    :param y_pred_prob: 预测概率 (连续概率)
    :return: 各种分类指标的字典
    """
    y_true_flat = y_true.ravel()
    y_pred_flat = (y_pred_prob.ravel() > 0.5).astype(int)  # 使用阈值0.5进行二值化
    TP = np.sum((y_true_flat == 1) & (y_pred_flat >= 0.5), dtype=np.float64)
    TN = np.sum((y_true_flat == 0) & (y_pred_flat < 0.5), dtype=np.float64)
    FP = np.sum((y_true_flat == 0) & (y_pred_flat >= 0.5), dtype=np.float64)
    FN = np.sum((y_true_flat == 1) & (y_pred_flat < 0.5), dtype=np.float64)
    print(f"TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN}")

    # 计算各项指标
    acc = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, average='binary', zero_division=1)
    recall = recall_score(y_true_flat, y_pred_flat, average='binary')
    f1 = f1_score(y_true_flat, y_pred_flat, average='binary')
    mcc = matthews_corrcoef(y_true_flat, y_pred_flat)
    sn = recall
    sp = TN / (FP + TN)
    # AUC和PRC可能会遇到只有一个类的情况
    try:
        auc_value = roc_auc_score(y_true_flat, y_pred_prob.ravel())  # AUC 使用概率值
    except ValueError as e:
        auc_value = 'Undefined (only one class present)'

    try:
        prc = average_precision_score(y_true_flat, y_pred_prob.ravel())  # PRC 使用概率值
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
