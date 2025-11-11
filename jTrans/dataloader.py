import pickle
from collections import defaultdict
import os
import random

import numpy as np

import tokenizer


def data_loader(d, func_len, batch_size, mode):
    cls_id = d['<cls>']
    unk_id = d['<unk>']
    pad_id = d['<pad>']
    mask_id = d['<mask>']
    loc_id = d['<loc>']
    first_addr = d['JUMP_ADDR_0']
    last_addr = d['JUMP_ADDR_511']
    mask_percent = 0.1
    output = []
    mask_list = []
    padding_list = []
    jump_list = []
    for _, _, func_tokens in tokenizer.yield_functions(shuffle=True, seed=628, split_ratio=0.8, mode=mode):
        token_id_list = [cls_id]
        for t in func_tokens[:func_len - 1]:
            token_id_list.append(d.get(t, unk_id))
            if first_addr <= token_id_list[-1] <= last_addr:
                jump_list.append((len(output), len(token_id_list) - 1, token_id_list[-1] - first_addr + 1))
        mask_candidates = [i for i in range(len(token_id_list)) if i != 0 and token_id_list[i] != unk_id]
        masks = random.sample(mask_candidates, int(len(mask_candidates) * mask_percent) + 1)
        for m in masks:
            mask_list.append((len(output), m))
        padding_list.append([False] * len(token_id_list))
        if (l := len(token_id_list)) < func_len:
            token_id_list += [pad_id] * (func_len - l)
            padding_list[-1] += [True] * (func_len - l)
        output.append(token_id_list)
        if len(output) >= batch_size:
            if not jump_list:
                output.clear()
                mask_list.clear()
                padding_list.clear()
                jump_list.clear()
                continue
            data = np.asarray(output, dtype=np.int64)
            idx = np.asarray(mask_list, dtype=np.int64).T
            pd = np.asarray(padding_list, dtype=bool)
            sub_list = random.sample(jump_list, (len(jump_list) + 3) // 4)
            jump = np.asarray(sub_list, dtype=np.int64).T
            output.clear()
            mask_list.clear()
            padding_list.clear()
            jump_list.clear()

            labels = data[idx[0], idx[1]]
            jump_labels = data[jump[0], jump[1]]

            jump_data = data.copy()
            data[idx[0], idx[1]] = mask_id
            assert np.all((jump_labels >= first_addr) & (jump_labels <= last_addr))
            jump_data[jump[0], jump[1]] = loc_id
            yield data, labels, idx, pd, jump, jump_data, jump_labels


def finetune_data_loader(d, func_len, batch_size, mode):
    pad_id = d['<pad>']
    first_addr = d['JUMP_ADDR_0']
    last_addr = d['JUMP_ADDR_511']

    if mode == 'train':
        with open('finetune_data.dat', 'rb') as f:
            output_d = pickle.load(f)

    padding_list = []
    output = []
    num_func_list = []
    jump_list = []
    for func_dict in output_d.values():
        for func_list in func_dict.values():
            if len(func_list) <= 1:
                continue
            func_list = random.sample(func_list, min(len(func_list), 4))
            if len(output) >= batch_size or len(output) + len(func_list) > batch_size:
                assert len(output) != 0
                data = np.asarray(output, dtype=np.int64)
                pd = np.asarray(padding_list, dtype=bool)
                mask = [np.ones((v, v), dtype=bool) for v in num_func_list]
                jump = np.asarray(jump_list, dtype=np.int64).T
                output.clear()
                padding_list.clear()
                jump_list.clear()
                num_func_list.clear()
                yield data, pd, mask, jump
            for token_id_list in func_list:
                for i in range(len(token_id_list)):
                    if first_addr <= token_id_list[i] <= last_addr:
                        jump_list.append((len(output), i, token_id_list[i] - first_addr + 1))
                padding_list.append([False] * len(token_id_list))
                if (l := len(token_id_list)) < func_len:
                    token_id_list += [pad_id] * (func_len - l)
                    padding_list[-1] += [True] * (func_len - l)
                output.append(token_id_list)
            num_func_list.append(len(func_list))


def gen_finetune_data(d, func_len, mode, output_path):
    cls_id = d['<cls>']
    unk_id = d['<unk>']
    output_d = defaultdict(lambda: defaultdict(list))
    for path, func_name, func_tokens in tokenizer.yield_functions(shuffle=True, seed=628, split_ratio=0.8, mode=mode):
        token_id_list = [cls_id]
        for t in func_tokens[:func_len - 1]:
            token_id_list.append(d.get(t, unk_id))
        k = path.split(os.path.sep)[-2]
        k = k[:k.rindex('-')]
        k = k + '/' + os.path.basename(path)
        output_d[k][func_name].append(token_id_list)
    with open(output_path, 'wb') as f:
        pickle.dump(dict(output_d), f)


if __name__ == '__main__':
    gen_finetune_data(tokenizer.load_tokens(), 512, 'train', 'finetune_data.dat')
