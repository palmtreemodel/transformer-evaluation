# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
import glob
import random
from tqdm import tqdm


def files(path):
    file = []
    for i in glob.iglob(path):
        file.append(i)
    return file

def process_code(file):
    code = []
    with open(file, "r") as f:
        func_objs = json.load(f)
        for key, obj in func_objs.items():
            if not key.startswith("__"):
                code.extend(obj["BasicBlocks"])
    return code

for fold in range(10):
    split_point = fold*5
    train_range = (split_point-40,split_point)
    valid_range = (split_point-10,split_point)
    test_range = (split_point,split_point+10)

    train_list = list(i%50 for i in range(train_range[0]+50,train_range[1]+50))
    valid_list = list(i%50 for i in range(valid_range[0]+50,valid_range[1]+50))
    test_list = list(i%50 for i in range(test_range[0]+50,test_range[1]+50))

    os.mkdir("data/{}".format(fold))
    folder = "data/{}".format(fold)
    cont=0
    with open("{}/train.jsonl".format(folder),'w') as f:
        for i in tqdm(train_list):
            items=files("/home/xli287_ucr_edu/Obf_detector_GPT/json/benign/{}-*".format(i))
            items += files("/home/xli287_ucr_edu/Obf_detector_GPT/json/ollvm_bcf/{}-*".format(i))
            items += files("/home/xli287_ucr_edu/Obf_detector_GPT/json/ollvm_fla/{}-*".format(i))
            items += files("/home/xli287_ucr_edu/Obf_detector_GPT/json/ollvm_sub/{}-*".format(i))
            random.shuffle(items)
            # print(items)
            for item in items:
                filename = item.split('/')[-1]    
                js={}
                js['label']= filename.split("-")[0]
                js['index']=str(cont)
                js['code']=process_code(item)
                if (js['code']):
                    f.write(json.dumps(js)+'\n')
                    cont+=1
            
    with open("{}/valid.jsonl".format(folder),'w') as f:
        for j in tqdm(valid_list):
            items=files("/home/xli287_ucr_edu/Obf_detector_GPT/json/benign/{}-*".format(j))
            items += files("/home/xli287_ucr_edu/Obf_detector_GPT/json/ollvm_bcf/{}-*".format(j))
            items += files("/home/xli287_ucr_edu/Obf_detector_GPT/json/ollvm_fla/{}-*".format(j))
            items += files("/home/xli287_ucr_edu/Obf_detector_GPT/json/ollvm_sub/{}-*".format(j))
            random.shuffle(items)
            for item in items:
                filename = item.split('/')[-1] 
                js={}
                js['label']= filename.split("-")[0]
                js['index']=str(cont)
                js['code']=process_code(item)
                f.write(json.dumps(js)+'\n')
                cont+=1
                
    with open("{}/test.jsonl".format(folder),'w') as f:
        for k in tqdm(test_list):
            items=files("/home/xli287_ucr_edu/Obf_detector_GPT/json/benign/{}-*".format(k))
            items += files("/home/xli287_ucr_edu/Obf_detector_GPT/json/ollvm_bcf/{}-*".format(k))
            items += files("/home/xli287_ucr_edu/Obf_detector_GPT/json/ollvm_fla/{}-*".format(k))
            items += files("/home/xli287_ucr_edu/Obf_detector_GPT/json/ollvm_sub/{}-*".format(k))
            random.shuffle(items)
            for item in items:
                filename = item.split('/')[-1] 
                js={}
                js['label']= filename.split("-")[0]
                js['index']=str(cont)
                js['code']=process_code(item)
                f.write(json.dumps(js)+'\n')
                cont+=1