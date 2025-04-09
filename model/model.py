import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.02)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)

        self.residual_connection1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)  # 用于调整通道
        self.residual_connection2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3)  # 用于调整通道

    def forward(self, x):
        residual = self.residual_connection1(x)  # 残差连接
        out = self.conv1(x)
        out = self.bn1(out)
        out += residual  # 残差加上卷积输出
        out = self.dropout1(self.relu(out))
        residual = residual + self.residual_connection2(out)

        out = self.conv2(out)
        out = self.bn1(out)
        out += residual  # 残差加上卷积输出
        out = self.dropout2(self.relu(out))
        residual = residual + self.residual_connection2(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # 残差加上卷积输出
        out = self.dropout3(self.relu(out))

        return out


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        attention = self.dropout(attention)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.embed_size)
        out = self.fc_out(out)
        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # ResNet
        self.res_block1 = ResidualBlock(868, 128, kernel_size=7, stride=3)
        self.res_block2 = ResidualBlock(128, 256, kernel_size=7, stride=3)
        self.res_block3 = ResidualBlock(256, 256, kernel_size=5, stride=2)
        self.res_block4 = ResidualBlock(256, 256, kernel_size=5, stride=2)

        # BiGRU
        self.bigru = BiGRU(input_size=804, hidden_size=128, num_layers=1)

        # Multi-head Attention
        self.attention = MultiHeadAttention(embed_size=256, heads=8)
        self.attention1 = MultiHeadAttention(embed_size=512, heads=8)

        # Fully connected layer
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        self.bn1 = nn.BatchNorm1d(804)
        self.bn2 = nn.BatchNorm1d(868)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x1, x2 = torch.split(x, [804, 868], dim=1)

        x1 = self.bn1(x1.unsqueeze(2))
        x2 = self.bn2(x2.unsqueeze(2))

        x2 = self.res_block1(x2)
        x2 = self.res_block2(x2)
        x2 = self.res_block3(x2)

        # 将卷积输出转为 LSTM 所需的输入维度 [batch_size, sequence_length, features]
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)

        # BiLSTM
        x1 = self.bigru(x1)

        # Attention
        x1 = self.attention(x1, x1, x1)
        x2 = self.attention(x2, x2, x2)
        x = torch.cat((x1, x2), dim=2)

        x = self.attention1(x, x, x)
        x = torch.mean(x, dim=1)

        # Fully connected layer
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        x = torch.sigmoid(x.squeeze())

        return x
