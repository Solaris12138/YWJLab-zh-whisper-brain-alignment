import torch
import torch.nn as nn
import logging

from .transformer import Transformer
from .position_encoding import add_position_encoding


class TRFModel(nn.Module):

    def __init__(self, n_signals, n_dims, d_model=512, using_tf=True, n_blocks=1, device="cpu"):
        super(TRFModel, self).__init__()
        self.n_signals = n_signals
        self.n_dims = n_dims
        self.d_model = d_model
        self.using_tf = using_tf
        self.device = device

        self.inp_proj = nn.Linear(n_dims, d_model)
        self.regressor = nn.Conv1d(in_channels=d_model, 
                                    out_channels=n_signals, 
                                    kernel_size=3, 
                                    stride=1, 
                                    padding=1, 
                                    dilation=1,
                                    bias=False)
        

        self._init_tf(n_blocks)
        self._init_weights()

        self.to(device)
    
    def _init_tf(self, n_blocks=1, n_heads=8, ffn_ratio=4, init_std=0.02):
        if self.using_tf:
            self.tf = Transformer(n_blocks=n_blocks, 
                                  embedding_dim=self.d_model, 
                                  n_heads=n_heads, 
                                  ffn_ratio=ffn_ratio,
                                  init_std=init_std)
            self.norm = nn.BatchNorm1d(num_features=self.d_model)
        else:
            self.tf = None
            self.norm = nn.BatchNorm1d(num_features=self.d_model)
   
    def _init_weights(self):
        if self.using_tf:
            nn.init.constant_(self.norm.weight, 1.0)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        # x: (batch_size, n_times, n_dims) ->  (batch_size, n_times, n_signals)
        x = self.inp_proj(x)
        if self.using_tf:
            x = add_position_encoding(x)
            x = self.tf(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.regressor(x)
        x = x.transpose(1, 2)

        return x


if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    B, L, D, V = 16, 10, 128, 32

    x = torch.randn(B, L, D).to(device)
    print(f"Input Shape: {x.shape}")

    model = TRFModel(
        n_signals=V,
        n_dims=D,
        d_model=64,
        using_tf=True,
        n_blocks=1,
        device=device
    )

    out = model(x)
    print(f"Output Shape: {out.shape}")