import registry
import torch

@registry.register("generator","V1")
class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
                
        self.generator = nn.Sequential(
            nn.Linear(2048, 1200),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1200),
            nn.LeakyReLU(0.2),
            nn.Linear(1200, 1200),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1200),
            nn.LeakyReLU(0.2),
            nn.Linear(1200, 2048)
        )
    def forward(self, x):
        return self.generator(x)

