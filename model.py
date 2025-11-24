# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

def create_input_for_stateformer(input, model):
    static = [t[0]for t in input[0]]
    inst_pos_emb = [t[0]for t in input[1]]
    op_pos_emb = [t[0]for t in input[2]]
    samples0 = {field:"" for field in ['static', 'inst_pos_emb', 'op_pos_emb', 'arch_emb', 'byte1', 'byte2', 'byte3', 'byte4']}
    arch = ['x86-64']*len(static)
    byte = ['##']*len(static)
    samples0['static']=' '.join(static)
    samples0['inst_pos_emb'] = ' '.join(inst_pos_emb)
    samples0['op_pos_emb'] = ' '.join(op_pos_emb)
    samples0['arch_emb'] = ' '.join(arch)
    samples0['byte1'] = ' '.join(byte)
    samples0['byte2'] = ' '.join(byte)
    samples0['byte3'] = ' '.join(byte)
    samples0['byte4'] = ' '.join(byte)
    sample0_tokens = model.encode2(samples0)
    model.process_token_dict(sample0_tokens)
    return sample0_tokens
    
class Model(nn.Module):   
    def __init__(self, encoder,config=None,tokenizer=None,args=None):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
    def forward(self, input_ids=None,p_input_ids=None,n_input_ids=None,labels=None, type=None):       
        if type == "jTrans":
            bs,_=input_ids.size()
            input_ids=torch.cat((input_ids,p_input_ids,n_input_ids),0)
            outputs=self.encoder.forward(input_ids.t(), input_ids.eq(2), None, ft=True) # attention_mask <pad> = 2
            outputs=outputs.split(bs,0)
        elif type == "stateformer":
            input = create_input_for_stateformer(input_ids, self.encoder)
            p_input = create_input_for_stateformer(p_input_ids, self.encoder)
            n_input = create_input_for_stateformer(n_input_ids, self.encoder)
            output = self.encoder.model(input, features_only=True)[0].mean(dim=1)
            p_output = self.encoder.model(p_input, features_only=True)[0].mean(dim=1)
            n_output = self.encoder.model(n_input, features_only=True)[0].mean(dim=1)
            outputs = [output, p_output, n_output]

        
        prob_1=(outputs[0]*outputs[1]).sum(-1)
        prob_2=(outputs[0]*outputs[2]).sum(-1)
        temp=torch.cat((outputs[0],outputs[1]),0)
        temp_labels=torch.cat((labels,labels),0)
        prob_3= torch.mm(outputs[0],temp.t())
        mask=labels[:,None]==temp_labels[None,:]
        prob_3=prob_3*(1-mask.float().to("cuda:0"))-1e9*mask.float().to("cuda:0")  
        prob=torch.softmax(torch.cat((prob_1[:,None],prob_2[:,None],prob_3),-1),-1)
        loss=torch.log(prob[:,0]+1e-10)
        loss=-loss.mean()
        return loss,outputs[0]