import os
from core.moduledict import ModuleDict, Munch, import_config_from_path
from core.model import AB_Papez_Piggyback, AB_Trans_Piggyback, AB_MemoryTransformerEncoderLayer_AdaLN
import torch.nn as nn

num_speakers = 2
sample_rate = 8000
tag_suffix = "librimix_original"

token_dim = 256
chunk_size = 250
mem_size = 10
config = import_config_from_path("config/libri2mix_8k.py", "config/optim_adam.py")
config.update(Munch(
    sample_rate = sample_rate,
    num_speakers = num_speakers,
    save_directory = os.path.join("work_dir/",
                                  f'papez/{num_speakers}ppl/{sample_rate}HZ/',
                                  tag_suffix),
    checkpoint_step = 4000,
    evaluation_step = 100,
    total_steps = 10_000_000,
    model = ModuleDict(
        module = AB_Papez_Piggyback,
        submodules = Munch(
            intra_model = ModuleDict(
                module = AB_Trans_Piggyback,
                submodules = Munch(
                    encoder_layer = ModuleDict(
                        module = AB_MemoryTransformerEncoderLayer_AdaLN,
                        d_model=token_dim, 
                        depth = 16,
                        nhead=8, 
                        chunk_size = chunk_size,
                        mem_size = mem_size,
                        dim_feedforward=1024, 
                        dropout=0.1,
                        batch_first=True, 
                        norm_first=False
                    ),
                ),
                num_layers = 16,
                hidden_size = token_dim,
            ),
        )
        encoder_kernel_size=16,
        encoder_in_nchannels=1,
        encoder_out_nchannels=token_dim,
        masknet_numspks=2,
        num_memory_slots = mem_size,
    ),
))