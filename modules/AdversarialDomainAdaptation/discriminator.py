import registry
import torch

@registry.register("disc_cls","lnr_cls")
class Discriminator_Class(nn.Module):
    
    def __init__(self,num_classes):
        super(Discriminator_Class, self).__init__()
                
        self.discriminator = nn.Sequential(
            nn.Linear(2048, 1600),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1600),
            nn.LeakyReLU(0.2),
            nn.Linear(1600, 1)
        )

        self.classify = nn.Sequential(nn.Linear(2048, num_classes),
                                      nn.LogSoftmax(1))

    def forward(self, x):
        valid = self.discriminator(x)
        label = self.classify(x)
        return valid, label

    

    
