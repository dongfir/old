from torch import nn
from .modules import *

class ReciprocalLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v):
        
        super().__init__()
        
        self.sequence_attention_layer = MultiHeadAttentionSequence(n_head, d_model,
                                                                d_k, d_v)
        
        self.protein_attention_layer = MultiHeadAttentionSequence(n_head, d_model,
                                                               d_k, d_v)
        
        self.reciprocal_attention_layer = MultiHeadAttentionReciprocal(n_head, d_model,
                                                                           d_k, d_v)
        
        
        
        self.ffn_seq = FFN(d_model, d_inner)
        
        self.ffn_protein = FFN(d_model, d_inner)

    def forward(self, sequence_enc, protein_seq_enc):
        prot_enc, prot_attention = self.protein_attention_layer(protein_seq_enc, protein_seq_enc, protein_seq_enc)
        
        seq_enc, sequence_attention = self.sequence_attention_layer(sequence_enc, sequence_enc, sequence_enc)
        
        
        prot_enc, seq_enc, prot_seq_attention, seq_prot_attention = self.reciprocal_attention_layer(prot_enc,
                                                                                   seq_enc,
                                                                                   seq_enc,
                                                                                   prot_enc)
        prot_enc = self.ffn_protein(prot_enc)
        
        seq_enc = self.ffn_seq(seq_enc)
        
        
        
        return prot_enc, seq_enc, prot_attention, sequence_attention, prot_seq_attention, seq_prot_attention
    

    
