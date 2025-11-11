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
    mask_percent = 0.1
    output = []
    mask_list = []
    padding_list = []
    for _, _, func_tokens in tokenizer.yield_functions(shuffle=True, seed=628, split_ratio=0.8, mode=mode):
        token_id_list = [cls_id]
        for t in func_tokens[:func_len - 1]:
            token_id_list.append(d.get(t, unk_id))
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
            data = np.asarray(output, dtype=np.int64)
            idx = np.asarray(mask_list, dtype=np.int64).T
            pd = np.asarray(padding_list, dtype=bool)
            output.clear()
            mask_list.clear()
            padding_list.clear()
            labels = data[idx[0], idx[1]]
            data[idx[0], idx[1]] = mask_id
            yield data, labels, idx, pd


def finetune_data_loader(d, func_len, batch_size, mode):
    cls_id = d['<cls>']
    unk_id = d['<unk>']
    pad_id = d['<pad>']
    output_d = defaultdict(lambda: defaultdict(list))
    for path, func_name, func_tokens in tokenizer.yield_functions(shuffle=True, seed=628, split_ratio=0.8, mode=mode):
        token_id_list = [cls_id]
        for t in func_tokens[:func_len - 1]:
            token_id_list.append(d.get(t, unk_id))
        k = path.split(os.path.sep)[-2]
        k = k[:k.rindex('-')]
        k = k + '/' + os.path.basename(path)
        output_d[k][func_name].append(token_id_list)
    
    padding_list = []
    output = []
    num_func_list = []
    for func_dict in output_d.values():
        for func_list in func_dict.values():
            if len(func_list) <= 1:
                continue
            func_list = random.sample(func_list, min(len(func_list), 4))
            if len(output) >= batch_size or len(output) + len(func_list) > batch_size:
                assert len(output) != 0, len(func_list)
                data = np.asarray(output, dtype=np.int64)
                pd = np.asarray(padding_list, dtype=bool)
                mask = [np.ones((v, v), dtype=bool) for v in num_func_list]
                output.clear()
                padding_list.clear()
                num_func_list.clear()
                yield data, pd, mask
            for token_id_list in func_list:
                padding_list.append([False] * len(token_id_list))
                if (l := len(token_id_list)) < func_len:
                    token_id_list += [pad_id] * (func_len - l)
                    padding_list[-1] += [True] * (func_len - l)
                output.append(token_id_list)
            num_func_list.append(len(func_list))


def finetune_type_data_loader(d, func_len, batch_size, mode):
    types_d = {'<notype>': -1,'no-access': 0, 'struct*': 1, 'signed_int': 2, 'signed_char*': 3, 'unsigned_long': 4, 'void*': 5, 'struct': 6, 'unsigned_char*': 7, 'unsigned_int': 8, 'array': 9, 'signed_long': 10, 'enum': 11, 'signed_int*': 12, 'union*': 13, 'unsigned_long*': 14, 'unsigned_char': 15, 'function*': 16, 'unsigned_int*': 17, 'unsigned_long_long*': 18, 'signed_long_long': 19, 'double': 20, 'unsigned_long_long': 21, 'signed_char': 22, 'union': 23, 'double*': 24, 'float': 25, 'unsigned_short': 26, 'signed_long*': 27, 'signed_long_long*': 28, 'float*': 29, 'long_double': 30, 'signed_short': 31, 'enum*': 32, 'unsigned_short*': 33, 'array*': 34, 'signed_short*': 35, 'long_double*': 36}
    cls_id = d['<cls>']
    unk_id = d['<unk>']
    pad_id = d['<pad>']
    notype_id = types_d['<notype>']
    padding_list = []
    output = []
    label = []
    for func_tokens, labels in tokenizer.yield_functions_type(mode=mode):
        token_id_list = [cls_id]
        label_id_list = [notype_id]
        for i in range(min(func_len, len(func_tokens)-1)):
            t = func_tokens[i]
            token_id_list.append(d.get(t, unk_id))
            label_id_list.append(types_d.get(labels[i], notype_id))

        if len(output) >= batch_size:
            assert len(output) != 0
            data = np.asarray(output, dtype=np.int64)
            pd = np.asarray(padding_list, dtype=bool)
            lab= np.asarray(label, dtype=np.int64)
            output.clear()
            padding_list.clear()
            label.clear()
            yield data, pd, lab
        padding_list.append([False] * len(token_id_list))
        if (l := len(token_id_list)) < func_len:
            token_id_list += [pad_id] * (func_len - l)
            label_id_list += [notype_id] * (func_len - l)
            padding_list[-1] += [True] * (func_len - l)
        output.append(token_id_list)
        label.append(label_id_list)

