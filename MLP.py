import torch
import torch.nn as nn
import torch.nn.functional as F
class PointCloudBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=4, num_layers=8, skip_layer=4, sample_per_point=10):
        super(PointCloudBackbone, self).__init__()
        
        self.layers = nn.ModuleList()
        self.skip_layer = skip_layer
        
        for i in range(num_layers):
            if i == 0:
                # First layer, input to hidden
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == skip_layer:
                # Skip connection layer, input and hidden concatenated
                self.layers.append(nn.Linear(input_dim + hidden_dim, hidden_dim))
            else:
                # Intermediate layers, hidden to hidden
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Final output layer, hidden to output
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, feature):
        h = feature
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                # Apply skip connection
                h = torch.cat([feature, h], dim=-1)
            h = F.relu(layer(h))
        h = self.output_layer(h)
        return h