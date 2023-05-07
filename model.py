import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.3)  # lstm
        self.fc_1 = nn.Linear(hidden_size, num_classes)  # fully connected 1
        self.fc = nn.Linear(32, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def init_hidden(self, x, device):
        hidden_state = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device))  # hidden state
        cell_state = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device))  # internal state
        return (hidden_state, cell_state)

    def forward(self, x, device='cpu'):
        # Propagate input through LSTM
        hn, cn = self.init_hidden(x, device=device)
        output, (hn, cn) = self.lstm(x, (hn, cn))  # lstm with input, hidden, and internal state
        hn = hn[-1].view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        # out = self.relu(out)  # relu
        # out = self.fc(out)  # Final Output
        return out


class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # x = self.sig(x)
        return x
