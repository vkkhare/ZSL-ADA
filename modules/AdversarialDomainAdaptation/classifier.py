class Classifier(nn.Module):
    
    def __init__(self,num_classes):
        super(Classifier, self).__init__()
                
        self.classifier = nn.Sequential(nn.Linear(2048, num_classes),
#                                       nn.Dropout(0.5),
#                                       nn.BatchNorm1d(800),
#                                       nn.LeakyReLU(0.2),
# #                                       nn.Linear(1600,800),
# #                                       nn.BatchNorm1d(800),
# #                                       nn.LeakyReLU(0.2),
#                                       nn.Linear(800,num_classes),
                                      nn.LogSoftmax(1))

    def forward(self, x):
        return self.classifier(x)
