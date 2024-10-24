import numpy as np
import torch
import torch.nn as nn
from .layers import *
from .modules import *

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x





class RepeatedModule(nn.Module):
    
    def __init__(self, n_layers, d_model,
                 n_head, d_k, d_v, d_inner, dropout=0.1):
        
        super().__init__()
        
        self.linear = nn.Linear(1024, d_model)
        self.sequence_embedding = nn.Embedding(20, d_model)
        self.d_model = d_model 
        
        self.reciprocal_layer_stack = nn.ModuleList([
                ReciprocalLayer(d_model,  d_inner,  n_head, d_k, d_v) 
                for _ in range(n_layers)])
    
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
       
    
    def _positional_embedding(self, batches, number):
        
        result = torch.exp(torch.arange(0, self.d_model,2,dtype=torch.float32)*-1*(np.log(10000)/self.d_model))
        
        numbers = torch.arange(0, number, dtype=torch.float32)
        
        numbers = numbers.unsqueeze(0)
        
        numbers = numbers.unsqueeze(2)
       
        result = numbers*result
        
        result = torch.cat((torch.sin(result), torch.cos(result)),2)
       
        return result
    
    def forward(self, peptide_sequence, protein_sequence):
        
        
        sequence_attention_list = []
        
        prot_attention_list = []
        
        prot_seq_attention_list = []
        
        seq_prot_attention_list = []
        
        sequence_enc = self.sequence_embedding(peptide_sequence)
        
        sequence_enc += to_var(self._positional_embedding(peptide_sequence.shape[0],
                                                           peptide_sequence.shape[1]))    
        sequence_enc = self.dropout(sequence_enc)
        
        
        
        
        
        prot_enc = self.dropout_2(self.linear(protein_sequence))
        
    
        

        for reciprocal_layer in self.reciprocal_layer_stack:
            
            prot_enc, sequence_enc, prot_attention, sequence_attention, prot_seq_attention, seq_prot_attention =\
                reciprocal_layer(sequence_enc, prot_enc)
            
            sequence_attention_list.append(sequence_attention)
            
            prot_attention_list.append(prot_attention)
            
            prot_seq_attention_list.append(prot_seq_attention)
            
            seq_prot_attention_list.append(seq_prot_attention)
            
        
        
        return prot_enc, sequence_enc, sequence_attention_list, prot_attention_list,\
            seq_prot_attention_list, seq_prot_attention_list
    

class FullModel(nn.Module):
    
    def __init__(self, n_layers, d_model, n_head,
                 d_k, d_v, d_inner, return_attention=False, dropout=0.2):
        
        super().__init__()
        self.repeated_module = RepeatedModule(n_layers, d_model,
                               n_head, d_k, d_v, d_inner, dropout=dropout)
        
        self.final_attention_layer = MultiHeadAttentionSequence(n_head, d_model,
                                                                d_k, d_v, dropout=dropout)
        
        self.final_ffn = FFN(d_model, d_inner, dropout=dropout) 
        self.output_projection_prot = nn.Linear(d_model, 2)
        
        
        
        self.softmax_prot =nn.LogSoftmax(dim=-1)
   
                
        self.return_attention = return_attention
        
    def forward(self, peptide_sequence, protein_sequence):
        
      
        prot_enc, sequence_enc, sequence_attention_list, prot_attention_list,\
            seq_prot_attention_list, seq_prot_attention_list = self.repeated_module(peptide_sequence,
                                                                                    protein_sequence)
            
        
        
        prot_enc, final_prot_seq_attention  = self.final_attention_layer(prot_enc, sequence_enc, sequence_enc)
        
        prot_enc = self.final_ffn(prot_enc)

        prot_enc = self.softmax_prot(self.output_projection_prot(prot_enc))
        
        
        
        
        
        if not self.return_attention:
            return prot_enc
        else:
            return prot_enc, sequence_attention_list, prot_attention_list,\
            seq_prot_attention_list, seq_prot_attention_list
        
