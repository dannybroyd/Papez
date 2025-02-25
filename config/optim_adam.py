import torch.optim.lr_scheduler
from core.moduledict import ModuleDict, Munch

config = Munch(
    optimizer = ModuleDict(
        module = torch.optim.Adam,
        submodules = [
            ModuleDict(
                module = torch.optim.lr_scheduler.LinearLR,
                start_factor=1.0, 
                end_factor=0.5,
                total_iters=100, 
            )
        ],
        lr = 1e-4,
    ),
)