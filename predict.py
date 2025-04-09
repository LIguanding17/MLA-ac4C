import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import Model
from functions import *

file_path = './data/test_data.csv'
features, labels = process_csv_and_encode(file_path)

dataset = TensorDataset(features, labels)
val_loader = DataLoader(dataset, batch_size=256, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={device}')
model = Model().to(device)

model.load_state_dict(torch.load(f'./model/model.pth'))
model.eval()

fold_performance = []
all_val_labels = []
all_outputs = []

with torch.no_grad():
    for batch in tqdm(val_loader, unit='batch'):
        batch_data, batch_labels = batch
        batch_data, batch_labels = batch_data.to(device).float(), batch_labels.to(device).float()
        outputs = model(batch_data)
        all_val_labels.append(batch_labels.cpu().detach().numpy())
        all_outputs.append(outputs.cpu().detach().numpy())

# Merge the results of all batches
all_val_labels = np.concatenate(all_val_labels, axis=0)
all_outputs = np.concatenate(all_outputs, axis=0)

# Evaluate model performance
metrics = calculate_metrics(all_val_labels, all_outputs)
fold_performance.append(metrics)
print(
    f'Performance: SN: {metrics["SN"]}, SP: {metrics["SP"]}, Accuracy: {metrics["Accuracy"]}, MCC: {metrics["MCC"]}, AUC: {metrics["AUC"]}')
