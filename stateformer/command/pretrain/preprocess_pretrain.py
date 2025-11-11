from command.params import fields, byte_start_pos, field_cf
from multiprocessing import Pool
import subprocess


def run(field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--trainpref', f'data-src/pretrain/train.{field}',
         '--validpref',
         f'data-src/pretrain/valid.{field}', '--destdir', f'data-bin/pretrain/{field}', '--workers',
         '40'])


def run_byte_with_dict(field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', 'data-bin/byte_dict.txt', '--trainpref',
         f'data-src/pretrain/train.{field}',
         '--validpref',
         f'data-src/pretrain/valid.{field}', '--destdir', f'data-bin/pretrain/{field}',
         '--workers',
         '40'])
#python prepare_finetune_complete.py --output_dir out --input_dir diffutils-2.8-O2 --stack_dir stacks/ --arch x86-64
# /home/administrator/Downloads/Lian/ghidra_9.2.2_PUBLIC/support/analyzeHeadless /home/administrator/ghidra utils -import /home/administrator/Downloads/Lian/PseudocodeDiffing/Dataset_for_BinDiff/diffutils/binaries/diffutils-3.4-O2 -scriptPath /mnt/sata/lian/stateformer/command/finetune -postScript get_var_loc_complete.py /mnt/sata/lian/stateformer/command/finetune/stacks/
with Pool() as pool:
    pool.map(run_byte_with_dict, fields[byte_start_pos:])

with Pool() as pool:
    pool.map(run, fields[:byte_start_pos] + [field_cf])
