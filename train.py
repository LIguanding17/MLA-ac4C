import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import pandas as pd
import numpy as np
import datetime
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import sys
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, matthews_corrcoef, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from config.config import *
from sklearn.model_selection import StratifiedKFold
import os
from torch.optim import Adam
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import Model
from functions import *

init_seeds(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={device}')

file_path = './data/train_data.csv'
features, labels = process_csv_and_encode(file_path)
print(features.shape, features[:10], labels.shape, labels[:10])
kf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=SEED)

for fold, (train_idx, test_idx) in enumerate(kf.split(features, labels.numpy())):
    print(f'Fold {fold + 1}')
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    print(f"Train label distribution: {np.bincount(train_labels)}")
    print(f"Test label distribution: {np.bincount(test_labels)}")
    train_dataset = TensorDataset(features[train_idx], labels[train_idx])
    test_dataset = TensorDataset(features[test_idx], labels[test_idx])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model = Model(features.shape[1], lstm_hidden_size, lstm_layers, attention_heads).to(device)
    # optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # optimizer = SGDMomentumWithNoise(model.parameters(), lr=learning_rate, momentum=momentum, noise_std=noise_std)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5,
                                                     min_lr=learning_rate / 10)
    # criterion = FocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        all_outputs = []
        all_labels = []
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            batch_data, batch_labels = batch
            batch_data, batch_labels = batch_data.to(device).float(), batch_labels.to(device).float()
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_outputs.append(output.cpu().detach().numpy())
            all_labels.append(batch_labels.cpu().detach().numpy())


        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        total_loss /= len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {total_loss}')
        # metrics = calculate_metrics(all_labels, all_outputs, 0.5)
        # print(
        #     f'Fold {fold + 1} Performance: Accuracy: {metrics["Accuracy"]:.4f}, Precision: {metrics["Precision"]:.4f}, Recall: {metrics["Recall"]:.4f}, F1-score: {metrics["F1-score"]:.4f}, AUC: {metrics["AUC"]:.4f}, PRC: {metrics["PRC"]:.4f}, MCC: {metrics["MCC"]:.4f}')
        if loss < best_val_loss:
            best_val_loss = loss
            # torch.save(model.state_dict(), f'./model/model_weights_fold_{fold + 1}.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            # print(f'\n{patience_counter}\n')

        if patience_counter >= train_patience:
            print("Early stopping")
            break

    model_dir = './model'
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), f'./model/model_fold_{fold + 1}.pth')

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        all_outputs = []
        all_labels = []
        for batch in tqdm(test_loader, unit='batch'):
            batch_data, batch_labels = batch
            batch_data, batch_labels = batch_data.to(device).float(), batch_labels.to(device).float()
            output = model(batch_data)
            loss = criterion(output, batch_labels)
            val_loss += loss.item()
            all_outputs.append(output.cpu().detach().numpy())
            all_labels.append(batch_labels.cpu().detach().numpy())


        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        val_loss /= len(test_loader)
        scheduler.step(val_loss)

        best_threshold, best_score = find_best_threshold_multi_metric(all_labels, all_outputs, idx=idx)
        metrics = calculate_metrics(all_labels, all_outputs, best_threshold)
        print(f'Fold {fold + 1}, Best threshold: {best_threshold}, Best {idx}: {best_score}')


        print()
        print(
            f'Test Performance: Accuracy: {metrics["Accuracy"]}, Precision: {metrics["Precision"]}, Recall: {metrics["Recall"]}, F1-score: {metrics["F1-score"]}, AUC: {metrics["AUC"]}, PRC: {metrics["PRC"]}, MCC: {metrics["MCC"]}')
        print()
