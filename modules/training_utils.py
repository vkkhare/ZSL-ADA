import registry
import torch


@registry.register("loss","mse")(torch.nn.MSELoss)
@registry.register("loss","l1")(torch.nn.L1Loss)
@registry.register("loss","nll")(torch.nn.NLLLoss)

@registry.register("optimizer","rms")(torch.optim.RMSprop)