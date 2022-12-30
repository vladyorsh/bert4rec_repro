import numpy as np
from tensorflow.keras import Model
import tensorflow as tf

from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel
from membert import TModel as MEMBERT
from membert import BERTLoss

class MEMBERT4Rec(SequentialRecsysModel):
    def __init__(self,
                 hidden_size = 256,
                 max_history_len = 100,
                 dropout_prob=0.2
                 num_attention_heads = 16,
                 num_hidden_layers = 3,
                 shared_embs = False
                ):
        super().__init__('linear', hidden_size, max_history_len)
        self.hidden_size = hidden_size
        self.max_history_length = max_history_len
        self.dropout_prob = dropout_prob
        self.num_attention_heads = num_attention_heads 
        self.num_hidden_layers = num_hidden_layers
        self.shared_embs = shared_embs

    def get_model(self):
        return MEMBERT4RecModel(self.num_items + 2, self.max_history_length, self.hidden_size, self.num_attention_heads, self.dropout_prob, self.num_hidden_layers, self.shared_embs)


class MEMBERT4RecModel(Model):
    def __init__(self, vocab_size, seq_len, hidden_dim, num_heads, dropout_rate, num_blocks, shared_embs, 
                        *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert = MEMBERT(vocab_size, seq_len, hidden_dim, num_heads, dropout_rate, num_blocks, shared_embs)
        self.loss = BERTLoss()

    def call(self, inputs, **kwargs):
        sequences = inputs[0]
        labels = inputs[1]
        positions = inputs[2]
        
        out  = self.bert(sequences)
        loss = self.loss(labels, out)
        
        return loss

    def score_all_items(self, inputs):
        sequence = inputs[0] 
        result = self.bert(sequence)[:,-1,:-2]
        return result