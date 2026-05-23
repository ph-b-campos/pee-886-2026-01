import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, n_neurons, n_layers):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        
        self.first_layer = self.fc_layer(6, n_neurons)
        
        hidden_layers_list = [self.fc_layer(n_neurons, n_neurons) for _ in range(n_layers)]
        self.hidden_layers = nn.Sequential(*hidden_layers_list)
        
        self.output_layer = nn.Linear(n_neurons, 1)

    def fc_layer(self, n_input, n_output):
        layer = nn.Sequential(
            nn.Linear(n_input, n_output),
            nn.BatchNorm1d(n_output),
            nn.ReLU()
        )
        return layer

    def forward(self, x):
        out = self.first_layer(x)
        out = self.hidden_layers(out)
        out = self.output_layer(out)
        return out