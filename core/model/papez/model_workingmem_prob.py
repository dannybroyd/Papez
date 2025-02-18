import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

class Dual_Path_Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        num_spks=2,
        **kwargs,
    ):  
        super().__init__()
        self.num_spks = num_spks

        self.out_channels = out_channels
        # embedding
        self.embed = nn.Sequential(
                          nn.Conv1d(
                              in_channels = out_channels+1, 
                              out_channels = out_channels,
                              kernel_size = 1,
                          ),
                          nn.InstanceNorm1d(out_channels, affine=True),
                          nn.PReLU(),
                          nn.Conv1d(
                              in_channels = out_channels, 
                              out_channels = out_channels,
                              kernel_size = 1,
                          ),
                        )
        # AWM transformer
        self.transformer = intra_model
        # Mask gen
        self.dislodge = nn.Sequential(
                            nn.Conv1d(
                              in_channels = out_channels, 
                              out_channels = out_channels* self.num_spks,
                              kernel_size = 1,
                            ),
                            nn.InstanceNorm1d(out_channels* self.num_spks, affine=True),
                            nn.PReLU(),
                            nn.Conv1d(
                              in_channels = out_channels* self.num_spks, 
                              out_channels = out_channels* self.num_spks,
                              kernel_size = 1,
                            ),
                            nn.Tanh()
                        )


    def forward(self, x, mem, info = None):
        # concatenate output of encoder and memory slots -> embedding layer
        x_mem = self.embed(torch.cat((x,mem),dim = 2) ) # B, N, L+M
        # permute = np reshape
        x_mem = torch.permute(x_mem, (0,2,1)) # B, L+M, N

        # after embedding, split to memory and sequence tokens
        seq_tokens, mem_tokens = x_mem[:,:x.shape[-1],:], x_mem[:,-mem.shape[-1]:,:] # (B, L, N), (B, M, N)

        # forward transformer on seq_tokens and mem_tokens
        if info is not None: awm_output = self.transformer(seq_tokens, mem_tokens, info = info) # B, L, N
        else: awm_output = self.transformer(seq_tokens, mem_tokens) # B, L, N
        awm_output = torch.permute(awm_output, (0,2,1)) # B, N, L

        # forward mask generation
        mask = self.dislodge(awm_output) # B, 3, L_
        
        B, _, L = mask.shape
        return torch.reshape(mask, (B, self.num_spks, self.out_channels, L)) # B, spks, N, L

class Model(nn.Module):
    def __init__(
        self,
        intra_model,
        encoder_kernel_size=16,
        encoder_in_nchannels=1,
        encoder_out_nchannels=256,
        masknet_numspks=4,
        num_memory_slots = 10,
        **kwargs,
    ):        
        super().__init__()
        self.encoder = nn.Sequential(
                          nn.Conv1d(
                            in_channels = encoder_in_nchannels, 
                            kernel_size=encoder_kernel_size,
                            out_channels = encoder_out_nchannels,
                            stride=encoder_kernel_size // 2,
                            padding='valid',
                          ),
                          nn.InstanceNorm1d(encoder_out_nchannels, affine=True),
                          nn.LeakyReLU(),
                          nn.Conv1d(
                            in_channels = encoder_out_nchannels, 
                            out_channels = encoder_out_nchannels,
                            kernel_size = 1,
                          ),
                        )
        # green blob in fig 1, (embedding + AWM transformer + mask gen)
        self.masknet = Dual_Path_Model(
            in_channels=encoder_out_nchannels,
            out_channels=encoder_out_nchannels,
            intra_model=intra_model,
            num_spks=masknet_numspks,
        )
        self.decoder = nn.Sequential(
                            nn.Conv1d(
                              in_channels = encoder_out_nchannels, 
                              out_channels = encoder_out_nchannels,
                              kernel_size = 1,
                            ),
                            nn.InstanceNorm1d(encoder_out_nchannels, affine=True),
                            nn.LeakyReLU(),
                            nn.ConvTranspose1d(
                                in_channels = encoder_out_nchannels,
                                out_channels=encoder_in_nchannels,
                                kernel_size=encoder_kernel_size,
                                stride=encoder_kernel_size // 2,
                                bias=True,
                            ),
                        )
        # store tensor with memory tokens without impacting backprop
        self.register_buffer('memory_slots',self._memory_slots(encoder_out_nchannels + 1, num_memory_slots)) # 1, N, M 

        self.num_spks = masknet_numspks
        
    def _memory_slots(self, N, M):
        # init memory slots - 1 on the diagonal & last row is all 1's
        mem = torch.unsqueeze(torch.eye(N, M), 0)
        mem[:,-1,:] = 1 # Memory Tag
        return mem

    def forward(self, mix, info = None):
        # B for batch (probably)
        B, _ , T_origin = mix.shape # B, 1, L

        # encoder on mixture
        mix_w = self.encoder(mix) # B, N, L'

        # padding to make sure memory doesn't interfere in self attention
        mix_w_tag = F.pad(mix_w, (0,0,0,1), "constant", 0) # Sequence Tag  (B, N+1, L')

        # duplicate memory slots for each batch
        memory_slots = self.memory_slots.repeat(B,1,1)# B, N+1, M

        # forward green blob & pass info along (if exists)
        if info is not None: est_mask = self.masknet(mix_w_tag,memory_slots, info = info) #  B, n_spks, N, L'
        else:est_mask = self.masknet(mix_w_tag,memory_slots) #  B, n_spks, N, L'

        # duplicate encoded sound for each speaker (2 in our case)
        mix_w = torch.tile(torch.unsqueeze(mix_w, dim = 1), (1,self.num_spks, 1, 1)) #  B, n_spks, N-1, L'

        # multiply encoder with predicted mask to get separation
        sep_h = mix_w * est_mask  # B, n_spks, N-1, L'

        # decoder expects input dim of (batch, channels, length)
        # essentially, we "duplicate" the size of the batch (for each speaker)
        B, n_spks, N_1, L = sep_h.shape
        sep_h = torch.reshape(sep_h, ( B * n_spks, N_1, L)) # ( B*n_spks, N-1, L)

        # forward decoder
        logits = self.decoder(sep_h) # ( B, n_spks, 1, _L)

        # remember output of decoder
        if info is not None: info.store(logits = logits)
        estimate = torch.reshape(logits, ( B, n_spks, -1)) # (B, n_spks, _L)

        # add padding incase we have different length of output compared to input, or cut the output length to match input
        T_est = estimate.shape[2]
        if  T_origin > T_est:
            estimate = F.pad(estimate, (0, T_origin - T_est,  0, 0, 0, 0), "constant", 0)
        else:
            estimate = estimate[:, :, :T_origin]
            
        return estimate


