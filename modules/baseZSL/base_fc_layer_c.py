import torch
import registry

@registry.register("layer_c","awa2")
class AwaLayerCov(nn.Module):
    def __init__(self):
        super(AwaLayerCov, self).__init__()
        self.fc_layer = nn.Sequential(nn.Linear(85, 1600 ),
                nn.BatchNorm1d(1600),
                nn.LeakyReLU(0.95),
                nn.Linear(1600, 2048),
                nn.Dropout(0.1)
                )
    
    def forward(self, x):
        return self.fc_layer(x)

@registry.register("layer_c","sun")
class SunLayerCov(nn.Module):
    def __init__(self):
        super(AwaLayerCov, self).__init__()
        self.fc_layer = nn.Sequential(
                nn.Linear(102, 1800),
                nn.BatchNorm1d(1800),
                nn.ReLU(),
                nn.Linear(1800,2048),
                nn.Dropout(0.05)
            )
    
    def forward(self, x):
        return self.fc_layer(x)
    
@registry.register("layer_c","cub")
class CubLayerCov(nn.Module):
    def __init__(self):
        super(AwaLayerCov, self).__init__()
        self.fc_layer = nn.Sequential(
                nn.Linear(312, 1200),
                nn.BatchNorm1d(1200),
                nn.ReLU(),
                nn.Linear(1200, 1800),
                nn.Dropout(0.1),
                nn.BatchNorm1d(1800),
                nn.ReLU(),
                nn.Linear(1800, 2048),
                nn.Dropout(0.1),
            )
    
    def forward(self, x):
        return self.fc_layer(x)

@registry.register("layer_c","cub_large")
class CubLargeLayerCov(nn.Module):
    def __init__(self):
        super(AwaLayerCov, self).__init__()
        self.fc_layer = nn.Sequential(
                nn.Linear(1024, 1500),
                nn.BatchNorm1d(1500),
                nn.ReLU(),
                nn.Linear(1500, 1800),
                nn.Dropout(0.3),
                nn.BatchNorm1d(1800),
                nn.ReLU(),
                nn.Linear(1800, 2048),
                nn.Dropout(0.1),
            )
    
    def forward(self, x):
        return self.fc_layer(x)