import torch
import registry

@registry.register('base_zsl','zsl_cls')
class BaseZSLNet(nn.Module):

    def __init__(self, layer_m,layer_c):
        super(BaseZSLNet, self).__init__()
        
        self.FC_layer_m = registry.construct("layer_m",layer_m)
        self.FC_layer_c = registry.construct("layer_c",layer_c)

    def forward(self, x):
        mean = self.FC_layer_m(x)
        cov = 0.5 + torch.sigmoid(self.FC_layer_c(x))
        return mean,cov
    
    def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')