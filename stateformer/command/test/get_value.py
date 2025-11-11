from os import stat
import sys
print(sys.path)
from command import params as configs
from fairseq.models.roberta_mf.model_nau import RobertaModelMFNAU
import json
import re
import tqdm
stateformer = RobertaModelMFNAU.from_pretrained(f'checkpoints/pretrain',
                                 checkpoint_file='checkpoint_best.pt',
                                 data_name_or_path=f'data-bin/finetune/x86')
stateformer = stateformer.cpu()
stateformer.eval()
loaded = stateformer.model

sample_path = '/mnt/sata/lian/stateformer_dataset/output/x86-64/coreutils-7.6-O1/kill.json'
arch_input = 'x86-64'
samples0 = {field: [] for field in configs.fields}

with open(sample_path,'r') as f:
    functionlist = json.load(f)

for func in tqdm.tqdm(functionlist.values()):
    basicblocks = func['BasicBlocks']
    static = []
    inst_pos = []
    op_pos = []
    arch = []
    inst_pos_counter = 0
    for block in basicblocks:
        for instr in block:
            tokens = instr.split()
            for i, token in enumerate(tokens):
                if '0x' in token.lower():
                    static.append('hexvar')

                elif token.lower().isdigit():
                    static.append('num')
                else:
                    static.append(token.lower())
                inst_pos.append(str(inst_pos_counter))
                op_pos.append(str(i))
                arch.append(arch_input)
            inst_pos_counter += 1
    if len(static) > 512:
        continue
    byte = ['##']*len(static)
    samples0['static'].append(' '.join(static))
    samples0['inst_pos_emb'].append(' '.join(inst_pos))
    samples0['op_pos_emb'].append(' '.join(op_pos))
    samples0['arch_emb'].append(' '.join(arch))
    samples0['byte1'].append(' '.join(byte))
    samples0['byte2'].append(' '.join(byte))
    samples0['byte3'].append(' '.join(byte))
    samples0['byte4'].append(' '.join(byte))

for sample_idx in range(len(samples0)):
    sample = {field: samples0[field][sample_idx] for field in configs.fields}
    sample0_tokens = stateformer.encode2(sample)
    sample0_emb = stateformer.process_token_dict(sample0_tokens)

    # run the model
    emb0_rep = loaded(sample0_tokens, features_only=True)[0]
    
