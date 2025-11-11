import os
import sys
import json
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.vocab import load_asmfile, init_tokenizer, generate_embeddings, check_vocab
from utils.model import load_weights_uniasm
from utils.asmfunction import create_functions
import tensorflow as tf


def load_uniasm(model_path, vocab_path, config_json,):
    model = load_weights_uniasm(model_path, config_json)
    tokenizer = init_tokenizer(vocab_path)

    return model, tokenizer




def get_embedding(tokenizer, model, funcs):
    # the input is the function dict
    funcs_objs = create_functions(funcs, bin_file=None, func_file = None)
    generate_embeddings(tokenizer, model, funcs_objs, func_type='liner', max_seq=512, b_num=False)
    funcs_embs = [f.embedding for f in funcs_objs]
    return funcs_embs





if __name__=='__main__':
    config_json = """
        {
            "hidden_act": "gelu",
            "hidden_size": 768,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_hidden_layers": 8,
            "vocab_size": 21000,
            "num_attention_heads": 12
        }
    """
    model_path = "C:\\Users\\Zixiang\\UniASM\\out\\best_model.h5"
    vocab_path = "C:\\Users\\Zixiang\\UniASM\\out\\vocab.txt"
    model, tokenizer = load_uniasm(model_path, vocab_path, config_json)
    with open("C:\\Users\\Zixiang\\jTrans\\x86-64\\binutils-2.30-O0\\elfedit.json", "r") as f:
        funcs_objs = json.load(f)
    # funcs_objs = [f for f in funcs_objs.values()]
    funcs_embs = get_embedding(tokenizer, model, funcs_objs)
    print(funcs_embs)

