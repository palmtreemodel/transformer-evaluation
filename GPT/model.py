import glob
import os.path
import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

DB_PATH = 'F:/embedding-files'
PATTERN = re.compile(r'(.+?)-(.+?)-([^-]+)===(.+?).csv')
device = torch.device('cuda')


def read_dataset():
    # lib-ver-fname to opt to func_name to emb
    mapping = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(f'{DB_PATH}/*.csv'):
        m = PATTERN.fullmatch(os.path.basename(path))
        assert m
        lib, version, opt, fname = m.groups()

        csv = pd.read_csv(path, converters={'embedding': lambda x: np.fromstring(x[1:-1], dtype=np.float32, sep=',')})
        for _, row in csv.iterrows():
            # FIXME
            mapping[(lib, fname)][row['Func_name']].append(row['embedding'])
    return mapping


def finetune_data_loader(mapping, batch_size, mode):
    select_key = 'coreutils' if mode == 'train' else 'diffutils'
    candidates = []
    for (key, _), name_dict in mapping.items():
        if key != select_key:
            continue
        for func_list in name_dict.values():
            if len(func_list) <= 1:
                continue
            candidates.append(func_list)
    r = random.Random(628)
    r.shuffle(candidates)

    output = []
    num_func_list = []

    for func_list in candidates:
        if len(output) >= batch_size or len(output) + len(func_list) > batch_size:
            data = np.vstack(output)
            mask = [np.ones((v, v), dtype=bool) for v in num_func_list]
            output.clear()
            num_func_list.clear()
            yield data, mask
        output.extend(func_list)
        num_func_list.append(len(func_list))
    if output:
        data = np.vstack(output)
        mask = [np.ones((v, v), dtype=bool) for v in num_func_list]
        output.clear()
        num_func_list.clear()
        yield data, mask


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.func_emb1 = nn.Linear(1536, 768)
        self.func_emb2 = nn.Linear(768, 512)

    def forward(self, x):
        x = self.func_emb1(x)
        x = F.gelu(x)
        x = self.func_emb2(x)
        return x


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


def finetune():
    lr = 1e-5
    model = MyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=10000, t_total=1000000)
    cur_step = 0
    mapping = read_dataset()
    for epoch in range(100):
        for data, mask in finetune_data_loader(mapping, 64, 'train'):
            data = torch.as_tensor(data, device=device)
            mask = torch.block_diag(*[torch.from_numpy(v) for v in mask]).to(device)

            optimizer.zero_grad()
            output = model(data)
            sim = cosine_similarity_torch(output) / 0.05
            sim = non_diagonal(sim)
            mask = non_diagonal(mask)

            test_sim = sim.clone()
            test_sim[~mask] = float('-inf')
            loss = torch.logsumexp(sim, dim=1) - torch.logsumexp(test_sim, dim=1)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
            cur_step += 1
            if cur_step % 500 == 0:
                print(f'{epoch=}, {cur_step=}, loss={loss.item()}')
                torch.save(model.state_dict(), 'finetune.dat')


if __name__ == '__main__':
    finetune()