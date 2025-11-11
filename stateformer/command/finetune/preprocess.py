# from command.params import fields
from multiprocessing import Pool
import subprocess
from itertools import product

import os

fields = ['static', 'inst_pos_emb', 'op_pos_emb', 'arch_emb', 'byte1', 'byte2', 'byte3', 'byte4']

def run(arch_opt, field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/pretrain/{field}/dict.txt', '--trainpref',
         f'data-src/finetune/{arch_opt}/train.{field}',
         '--validpref',
         f'data-src/finetune/{arch_opt}/valid.{field}', '--destdir', f'data-bin/finetune/{arch_opt}/{field}',
         '--workers',
         '40'])


#arch_opts = [name for name in os.listdir("data-src/finetune")]
arch_opts=['x86-64-cross-bin']
# out_dir='/mnt/sata/lian/BinKit_dataset/out'
# filenames = ['static', 'inst_pos_emb', 'op_pos_emb', 'arch_emb', 'byte1', 'byte2', 'byte3', 'byte4', 'label', 'arg_info']
# all_data = {}
# folders=os.listdir(out_dir)
# folders.sort()
# for folder in folders:
#     for field_name in filenames:
#         f=open(os.path.join(out_dir, folder, 'train.'+field_name), 'r')
#         if field_name not in all_data:
#             all_data[field_name]=[]
#         all_data[field_name].extend(f.readlines())

# dup = set()
# selected = []
# for i, line in enumerate(all_data['static']):
#     if line not in dup:
#         selected.append(i)
#         dup.add(line)

# for field_name in filenames:
#     f1=open(f'/mnt/sata/lian/stateformer/data-src/finetune/{arch_opts[0]}/train.{field_name}', 'w')
#     f2=open(f'/mnt/sata/lian/stateformer/data-src/finetune/{arch_opts[0]}/valid.{field_name}', 'w')
#     all_data[field_name] = [line for i, line in enumerate(all_data[field_name]) if i in selected]
#     length = len(all_data[field_name])
#     split_idx = length*9//10
#     f1.write(''.join(all_data[field_name][:split_idx]))
#     f2.write(''.join(all_data[field_name][split_idx:]))
#     f1.close()
#     f2.close()

with Pool() as pool:
    pool.starmap(run, product(arch_opts, fields))

for arch_opt in arch_opts:
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/label_dict.txt', '--trainpref',
         f'data-src/finetune/{arch_opt}/train.label',
         '--validpref',
         f'data-src/finetune/{arch_opt}/valid.label', '--destdir', f'data-bin/finetune/{arch_opt}/label',
         '--workers', '40'])

    subprocess.run(
        ['cp', '-r', f'data-bin/pretrain/cover', f'data-bin/finetune/{arch_opt}/'])
