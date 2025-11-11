import torch

d_model = 768
nhead = 12
num_encoder_layers = 8
dim_feedforward = d_model * 4
dropout = 0.1
activation = 'gelu'
layer_norm_eps = 1e-5
batch_first = False
device = torch.device('cuda')
dtype = None
