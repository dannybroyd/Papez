import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

#https://pytorch.org/tutorials/beginner/transformer_tutorial.html

class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dims: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dims, 2) * (-math.log(10000.0) / embedding_dims))
        pe = torch.zeros(max_len, 1, embedding_dims)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class Dual_Computation_Block(nn.Module):
    def __init__(
        self,
        intra_mdl,
        inter_mdl,
        out_channels,
        **kwargs,
    ):
        super().__init__()
        self.intra_mdl = intra_mdl.to("cuda:0")
        self.inter_mdl = inter_mdl.to("cuda:1")


    def forward(self, x, step, info = None): 
        B, N, K, S = x.shape
        x = x.to("cuda:0")
        intra = torch.reshape(torch.permute(x, (0, 3, 2, 1)), (B * S, K, N)) # B*S, K, N
        
        intra = self.intra_mdl(intra, outer_step = step) #(batch, seq, feature) B*S, K, N
            

        intra = torch.reshape(intra, (B, S, K, N)) # B, S, K, N
        intra = intra + torch.permute(x, (0, 3, 2, 1)) # B, S, K, N
        
        intra = intra.to("cuda:1")
        
        intra = torch.permute(intra, (0, 2, 1, 3)) # B, K, S, N
        inter = torch.reshape(intra, (B * K, S, N))
        
        inter = self.inter_mdl(inter, outer_step = step) #(batch, seq, feature) B*K, S, N 
        
        inter = torch.reshape(inter, (B, K, S, N)) # B, K, S, N
        out = torch.permute(inter, (0, 3, 1, 2)) # B, N, K, S
        out = x.to("cuda:1") + out
        
        out = out
        return out


class Dual_Path_Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        inter_model,
        num_layers=1,
        K=200,
        num_spks=2,
        **kwargs,
    ):  
        super().__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers

        self.out_channels = out_channels

        self.embed = nn.Sequential(
                          nn.Conv1d(
                              in_channels = out_channels, 
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
                        ).to("cuda:0")
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
                        ).to("cuda:1")
        
        self.dual_mdl = nn.ModuleList([
                Dual_Computation_Block(
                    intra_model,
                    inter_model,
                    out_channels,
                )
                for idx in range(num_layers)
        ])


    def forward(self, x, info = None):
        x = self.embed(x) # B, N, L
        if info is not None: info.store(embed = x)
        
        # B, N, K, S
        x, gap = self._Segmentation(x, self.K)
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x, step = i) # B, N, K, S

        x = self._over_add(x, gap) # B, N, L_
        
        x = self.dislodge(x) # B, 3, L_
        if info is not None: info.store(dislodge = x)
        
        B, _, L = x.shape
        #return torch.permute(x, (1, 0, 2, 3)) # spks, B, N, L
        return torch.reshape(x, (B, self.num_spks, self.out_channels, L)) # B, spks, N, L

    def _padding(self, input, K):
        B, N, L = input.shape
        P = K // 2

        gap = K - (P + L % K) % K

        if gap > 0 :
            pad = torch.zeros(B, N, gap, device = input.device)
            input = torch.cat([input, pad], dim=2)

        _pad = torch.zeros(B, N, P, device = input.device)
        input = torch.cat([_pad, input, _pad], dim=2)
        return input, gap

    def _Segmentation(self, input, K):
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        input1 = torch.reshape(input[:, :, :-P], (B, N, -1, K))
        input2 = torch.reshape(input[:, :, P:], (B, N, -1, K))
        input = torch.cat([input1, input2], dim=3)
        input = torch.reshape(input, (B, N, -1, K))
        input = torch.permute(input, (0, 1, 3, 2))
        return input, gap

    def _over_add(self, input, gap):
        B, N, K, S = input.shape
        P = K // 2
        input = torch.reshape(torch.permute(input, (0, 1, 3, 2)), (B, N, -1, K * 2))
        input1 = torch.reshape(input[:, :, :, :K], (B, N, -1))[:, :, P:]
        input2 = torch.reshape(input[:, :, :, K:], (B, N, -1))[:, :, :-P]
        input = input1 + input2
        
        if gap > 0:
            input = input[:, :, :-gap]
        return input

class Model(nn.Module):
    def __init__(
        self,
        intra_model,
        inter_model,
        encoder_kernel_size=16,
        encoder_in_nchannels=1,
        encoder_out_nchannels=256,
        masknet_chunksize=250,
        masknet_numlayers=2,
        masknet_numspks=4,
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
                          nn.ReLU(),
                          nn.Conv1d(
                            in_channels = encoder_out_nchannels, 
                            out_channels = encoder_out_nchannels,
                            kernel_size = 1,
                          ),
                        ).to("cuda:0")
        
        self.masknet = Dual_Path_Model(
            in_channels=encoder_out_nchannels,
            out_channels=encoder_out_nchannels,
            intra_model=intra_model,
            inter_model=inter_model,
            num_layers=masknet_numlayers,
            K=masknet_chunksize,
            num_spks=masknet_numspks,
        )
        self.decoder = nn.Sequential(
                            nn.Conv1d(
                              in_channels = encoder_out_nchannels, 
                              out_channels = encoder_out_nchannels,
                              kernel_size = 1,
                            ),
                            nn.InstanceNorm1d(encoder_out_nchannels, affine=True),
                            nn.ReLU(),
                            nn.ConvTranspose1d(
                                in_channels = encoder_out_nchannels,
                                out_channels=encoder_in_nchannels,
                                kernel_size=encoder_kernel_size,
                                stride=encoder_kernel_size // 2,
                                bias=True,
                            ),
                        ).to("cuda:1")
        

        self.num_spks = masknet_numspks

    def forward(self, mix, info = None):
        T_origin = mix.shape[2] # B, 1, L
        mix_w = self.encoder(mix) # B, N, L'
        
        est_mask = self.masknet(mix_w) #  B, n_spks, N, L'
            
        
        mix_w = torch.tile(torch.unsqueeze(mix_w, dim = 1), (1,self.num_spks, 1, 1)) #  B, n_spks, N, L'
        sep_h = mix_w.to("cuda:1") * est_mask  # B, n_spks, N, L'

        B, n_spks, N, L = sep_h.shape
        sep_h = torch.reshape(sep_h, ( B * n_spks, N, L)) # ( B, n_spks, N, L')
        logits = self.decoder(sep_h) # ( B, n_spks, 1, _L)
        estimate = torch.reshape(logits, ( B, n_spks, -1)) # (B, n_spks, _L)
        
        T_est = estimate.shape[2]
        if  T_origin > T_est:
            estimate = F.pad(estimate, (0, T_origin - T_est,  0, 0, 0, 0), "constant", 0)
        else:
            estimate = estimate[:, :, :T_origin]
            
        return estimate


