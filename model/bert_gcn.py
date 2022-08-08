import torch
from torch_geometric.nn import GCNConv, GATConv
from transformers import DistilBertModel
from utils import LABELS
import torch.nn.functional as F
from torch import nn

class BERTxGCN(torch.nn.Module):

    def __init__(self, n_classes=len(LABELS), hidden_size=768, dropout_rate=0.2, bert_model="distilbert-base-multilingual-cased"):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.BERT = DistilBertModel.from_pretrained(
            bert_model)
        self.hidden_size = hidden_size
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()
        
        self.conv1 = GCNConv(self.hidden_size + 2, 128, improved=True)
        self.conv2 = GCNConv(128,  self.n_classes, improved=True)
        
    def forward(self, data):
        # print(data)
        # for transductive setting with full-batch update
        edge_index, edge_weight = data.edge_index, data.edge_attr
        bert_output = self.BERT(attention_mask=data.attention_mask,
                      input_ids=data.input_ids)
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py
        first_token_tensor = bert_output['last_hidden_state'][:, 0]
        pooled_output = self.dense(first_token_tensor)
        x = self.activation(pooled_output)
        x = torch.cat((x, data.p_num, data.text_len), dim=1)
        x = F.dropout(F.relu(self.conv1(x, edge_index, edge_weight)),
                    p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
    

        return F.log_softmax(x, dim=1)
