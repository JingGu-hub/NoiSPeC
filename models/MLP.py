import torch.nn as nn
import sys

class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()

        self.dropout2 = nn.Dropout(0.1)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()

        self.dropout3 = nn.Dropout(0.1)
        self.dense3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.relu3 = nn.ReLU()

        self.dropout4 = nn.Dropout(0.1)
        self.dense4 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.relu4 = nn.ReLU()

        self.dropout5 = nn.Dropout(0.1)
        self.dense5 = nn.Linear(hidden_size, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.relu5 = nn.ReLU()

        self.dropout6 = nn.Dropout(0.1)
        self.dense6 = nn.Linear(hidden_size, output_size)
        self.bn6 = nn.BatchNorm1d(output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.flatten(x)

        out = self.dropout1(out)
        out = self.dense1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        # out = self.dropout2(out)
        out = self.dense2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.dropout3(out)
        out = self.dense3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        # out = self.dropout4(out)
        out = self.dense4(out)
        out = self.bn4(out)
        out = self.relu4(out)

        out = self.dropout5(out)
        out = self.dense5(out)
        out = self.bn5(out)
        out = self.relu5(out)

        out = self.dropout6(out)
        out = self.dense6(out)
        out = self.bn6(out)
        out = self.softmax(out)

        return out

def mlp(input_size, hidden_size, output_size):
    return MLP(input_size, hidden_size, output_size)