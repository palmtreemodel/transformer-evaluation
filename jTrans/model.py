import math

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

import dataloader
import tokenizer
from config import *


# code from https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/optimization.html
class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


# code from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, jump):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        pos = self.pe[:x.size(0)]
        if jump is not None and len(jump) == 3 and len(jump[0]) != 0:
            pos = pos.expand_as(x).contiguous()
            pos[jump[2], jump[0]] = x[jump[1], jump[0]]
        x = x + pos
        return self.dropout(x)


class BertLMPredictionHead(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.LayerNorm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, x):
        x = self.dense(x)
        x = F.gelu(x)
        x = self.LayerNorm(x)
        x = self.decoder(x)
        return x


class DummyDecoder(nn.Module):
    def forward(self, x, *args, **kwargs):
        assert False


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Transformer(d_model, nhead, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
                               layer_norm_eps=layer_norm_eps, batch_first=batch_first, device=device, dtype=dtype, custom_decoder=DummyDecoder())
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d = tokenizer.load_tokens()
        self.emb = nn.Embedding(len(self.d), d_model)
        self.mlm = BertLMPredictionHead(len(self.d))
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.func_emb1 = nn.Linear(d_model, d_model)
        self.func_emb2 = nn.Linear(d_model, d_model)

    def forward(self, x, pd, jump, mlm=True, ft=False):
        x = self.emb(x)
        x = self.pos_encoder(x, jump)
        x = self.norm(x)
        x = self.model.encoder(x, src_key_padding_mask=pd)
        if not ft:
            if mlm:
                x = self.mlm(x)
            return x
        else:
            x = x[0]
            x = self.func_emb1(x)
            x = F.gelu(x)
            x = self.func_emb2(x)
            return x


class JumpPredictionTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MyModel()
        self.decoder = nn.Linear(d_model, 512)

    def forward(self, x, pd):
        x = self.model(x, pd, None, mlm=False)
        x = self.decoder(x)
        return x


def train():
    lr = 1e-5
    accum_iter = 4
    model = MyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=10000 // 4, t_total=1000000)
    cur_step = 0
    for epoch in range(100):
        for data, label, idx, pd, jump, jump_data, jump_label in dataloader.data_loader(model.d, 512, 16, 'train'):
            if len(idx[0]) == 0:
                continue
            data = torch.as_tensor(data, device=device).t()
            label = torch.as_tensor(label, device=device)
            idx = torch.as_tensor(idx, device=device)
            pd = torch.as_tensor(pd, device=device)
            jump = torch.as_tensor(jump, device=device)
            jump_data = torch.as_tensor(jump_data, device=device).t()
            jump_label = torch.as_tensor(jump_label, device=device)

            # optimizer.zero_grad()
            output = model(data, pd, jump)
            output = output[idx[1], idx[0]]
            loss = F.cross_entropy(output, label)

            output = model(jump_data, pd, jump)
            output = output[jump[1], jump[0]]
            loss2 = F.cross_entropy(output, jump_label)

            loss = (loss + loss2) / accum_iter
            loss.backward()

            if (cur_step + 1) % accum_iter == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            cur_step += 1
            if cur_step % 500 == 0:
                print(f'{epoch=}, {cur_step=}, loss={loss.item()}')
                torch.save(model.state_dict(), 'model.dat')


def finetune():
    lr = 1e-5
    accum_iter = 4
    model = MyModel().to(device)
    model.load_state_dict(torch.load('model.dat'), strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=10000 // accum_iter, t_total=1000000)
    cur_step = 0
    for epoch in range(100):
        for data, pd, mask, jump in dataloader.finetune_data_loader(model.d, 512, 16, 'train'):
            data = torch.as_tensor(data, device=device).t()
            pd = torch.as_tensor(pd, device=device)
            mask = torch.block_diag(*[torch.from_numpy(v) for v in mask]).to(device)
            #jump = torch.as_tensor(jump, device=device)

            x = model(data, pd, None, ft=True)
            sim = cosine_similarity_torch(x) / 0.05
            sim = non_diagonal(sim)
            mask = non_diagonal(mask)

            test_sim = sim.clone()
            test_sim[~mask] = float('-inf')
            loss = torch.logsumexp(sim, dim=1) - torch.logsumexp(test_sim, dim=1)
            loss = torch.mean(loss) / accum_iter
            loss.backward()

            if (cur_step + 1) % accum_iter == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            cur_step += 1
            if cur_step % 500 == 0:
                print(f'{epoch=}, {cur_step=}, loss={loss.item()}')
                torch.save(model.state_dict(), 'finetune.dat')


def jump_target_training():
    lr = 1e-5
    accum_iter = 4
    model = JumpPredictionTask().to(device)
    model.model.load_state_dict(torch.load('model.dat', map_location=device), strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=10000 // 4, t_total=1000000)
    cur_step = 0
    first_addr = model.model.d['JUMP_ADDR_0']
    for epoch in range(100):
        for _, _, _, pd, jump, jump_data, jump_label in dataloader.data_loader(model.model.d, 512, 16, 'train'):
            jump_label = jump_label - first_addr
            assert np.all((jump_label >= 0) & (jump_label < 512))
            pd = torch.as_tensor(pd, device=device)
            jump = torch.as_tensor(jump, device=device)
            jump_data = torch.as_tensor(jump_data, device=device).t()
            jump_label = torch.as_tensor(jump_label, device=device)

            output = model(jump_data, pd)
            output = output[jump[1], jump[0]]
            loss = F.cross_entropy(output, jump_label)
            loss = loss / accum_iter
            loss.backward()

            if (cur_step + 1) % accum_iter == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            cur_step += 1
            if cur_step % 500 == 0:
                print(f'{epoch=}, {cur_step=}, loss={loss.item()}')
                torch.save(model.state_dict(), 'jump_target.dat')


def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def non_diagonal(x):
    sz = x.size()
    assert sz[0] == sz[1]
    n = sz[0]
    return x.view(-1)[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)


if __name__ == '__main__':
    jump_target_training()
