import glob
import json
import os
import pickle
import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm

ignore_tokens = ['ptr', 'offset', 'xmmword', 'dword', 'qword', 'word', 'byte', 'short', ',']
dataset_path = 'E:/x86-64'


def yield_functions(shuffle=False, seed=None, split_ratio=None, mode=None):
    r = random.Random(seed)
    path_list = glob.glob(f'{dataset_path}/*/*.json')
    if shuffle:
        r.shuffle(path_list)
    if split_ratio is not None:
        split_point = int(len(path_list) * split_ratio)
        if mode == 'train':
            path_list = path_list[:split_point]
        elif mode == 'test':
            path_list = path_list[split_point:]
    for path in tqdm(path_list):
        with open(path, 'r') as f:
            v = json.load(f)
        v = list(v.items())
        if shuffle:
            r.shuffle(v)
        for func_name, func_dict in v:
            bbs = func_dict['BasicBlocks']
            edges = func_dict['Edges']
            edge_dict = defaultdict(set)
            for u, v in edges:
                edge_dict[u].add(v)
            bb_tokens = []
            for bb in bbs:
                tokens = []
                for inst in bb:
                    parts = inst.strip().split()
                    for i, p in enumerate(parts):
                        if not p:
                            continue
                        if p in ignore_tokens:
                            continue
                        if p.startswith('0x'):
                            if parts[i-1][0] == 'J':
                                tokens.append('<placeholder>')
                            elif parts[i-1] == 'CALL':
                                tokens.append('<function>')
                            else:
                                tokens.append('const')
                        elif p == '+':
                            if parts[i+1] != '-':
                                tokens.append(p)
                        else:
                            tokens.append(p)
                    tokens.append('<sep>')
                bb_tokens.append(tokens)
            bb_addrs = np.cumsum([0] + [len(v) for v in bb_tokens])[:-1]
            for u, v_set in edge_dict.items():
                if len(v_set) == 1:
                    v = next(iter(v_set))
                else:
                    v = [c for c in v_set if c != u + 1]
                    if len(v) > 1:
                        continue
                    v = v[0]
                if v >= len(bb_addrs):
                    continue
                addr = bb_addrs[v]
                if bb_tokens[u][-2] != '<placeholder>':
                    continue
                if addr < 511:
                    bb_tokens[u][-2] = f'JUMP_ADDR_{addr}'
                else:
                    bb_tokens[u][-2] = 'JUMP_ADDR_EXCEED'
            func_tokens = [v for l in bb_tokens for v in l]
            for i, v in enumerate(func_tokens):
                if v == '<placeholder>':
                    func_tokens[i] = 'JUMP_ADDR_UNK'
            yield path, func_name, func_tokens


def gen_dict():
    d = defaultdict(int)
    for _, _, func_tokens in yield_functions():
        for t in func_tokens:
            d[t] += 1
    v = list(d.items())
    v.sort(key=lambda x: x[1], reverse=True)
    d = {
        '<unk>': 0,
        '<mask>': 1,
        '<pad>': 2,
        '<cls>': 3,
        'JUMP_ADDR_EXCEED': 4,
        'JUMP_ADDR_UNK': 5,
    }
    for i in range(512):
        d[f'JUMP_ADDR_{i}'] = len(d)
    for t, _ in v[:5000]:
        if t not in d:
            d[t] = len(d)
    with open('dict.dat', 'wb') as f:
        pickle.dump(d, f)


def load_tokens():
    with open('dict.dat', 'rb') as f:
        d = pickle.load(f)
    assert '<loc>' not in d
    d['<loc>'] = len(d)
    return d


if __name__ == '__main__':
    v = load_tokens()
    print(v)
