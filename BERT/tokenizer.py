import glob
import json
import os
import pickle
import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm

ignore_tokens = ['ptr', 'offset', 'xmmword', 'dword', 'qword', 'word', 'byte', 'short', ',']
dataset_path = '/mnt/sata/lian/stateformer_dataset/output/x86-64'
type_dataset_path = '/mnt/sata/lian/stateformer_dataset/type/x86-64'


def yield_functions(shuffle=False, seed=None, split_ratio=None, mode=None):
    r = random.Random(seed)
    if os.name == 'nt':
        path_list = ['Z:\\diff3.json', 'Z:\\elfedit.json']
    else:
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
                            if parts[i-1] == 'CALL':
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
            func_tokens = [v for l in bb_tokens for v in l]
            yield path, func_name, func_tokens

def yield_functions_type(mode='train'):
    if mode == 'train':
        corpora_lines = open(type_dataset_path + '/train.static','r').readlines()
        relation_lines = open(type_dataset_path + '/train.label','r').readlines()
        instr_pos_lines = open(type_dataset_path + '/train.inst_pos_emb','r').readlines()
    elif mode == 'test':
        corpora_lines = open(type_dataset_path + '/valid.static','r').readlines()
        relation_lines = open(type_dataset_path + '/valid.label','r').readlines()
        instr_pos_lines = open(type_dataset_path + '/valid.inst_pos_emb','r').readlines()
    
    for i in range(len(corpora_lines)):
        parts = corpora_lines[i].strip().upper().split()
        labels = relation_lines[i].strip().split()
        instr_pos = instr_pos_lines[i].strip().split()
        func_tokens = []
        func_labels = []
        last_inst_id = '0'
        for j in range(len(parts)):
            p = parts[j]
            l = labels[j]
            if not p:
                continue
            if p.lower() in ignore_tokens:
                continue
            if p == 'HEXVAR':
                if parts[i-1] == 'CALL':
                    func_tokens.append('<function>')
                else:
                    func_tokens.append('const')
            elif p == 'NUM':
                func_tokens.append('const')
            elif p == '+':
                if parts[i+1] != '-':
                    func_tokens.append(p)
            else:
                func_tokens.append(p)
            func_labels.append(l)

            if j == len(parts)-1 or instr_pos[j+1] != last_inst_id:
                func_tokens.append('<sep>')
                func_labels.append(-1)
                if j <len(parts)-1:
                    last_inst_id = instr_pos[j+1]
        yield func_tokens, func_labels


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
        return pickle.load(f)


if __name__ == '__main__':
    v = load_tokens()
    print(v)
