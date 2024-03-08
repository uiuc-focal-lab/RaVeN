import torch
import torch.nn as nn

#   the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the neural network architecture
class SimpleNNSigmoid(nn.Module):
    def __init__(self):
        super(SimpleNNSigmoid, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the neural network architecture
class SimpleNNTanh(nn.Module):
    def __init__(self):
        super(SimpleNNTanh, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(net_name):
    if 'relu' in net_name:
        model = SimpleNN()
        model.load_state_dict(torch.load(net_name))
        return model
    elif 'sigmoid' in net_name:
        model = SimpleNNSigmoid()
        model.load_state_dict(torch.load(net_name))
        return model 
    elif 'tanh' in net_name:
        model = SimpleNNTanh()
        model.load_state_dict(torch.load(net_name))
        return model
    else:
        raise ValueError(f'model : {net_name} not found')
