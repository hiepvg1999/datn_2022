import torch
from torch_geometric.nn import GATv2Conv
from transformers import RobertaModel
from utils import LABELS
import torch.nn.functional as F
from torch import nn
import random
import warnings
warnings.filterwarnings("ignore")
class BERTxGAT(torch.nn.Module):

    def __init__(self, n_classes=len(LABELS), hidden_size=768, dropout_rate=0.2, bert_model="roberta-base"):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.BERT = RobertaModel.from_pretrained(
            bert_model)
        self.hidden_size = hidden_size
        self.dense1 = nn.Linear(self.hidden_size, self.hidden_size*2)  #update
        self.activation = nn.Tanh()
        self.conv1 = GATv2Conv(self.hidden_size + 2,512)
        self.conv2 = GATv2Conv(512,256)
        # update
        self.post_process_layers = nn.Sequential(
            nn.BatchNorm1d(256, affine= False),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(256, n_classes),
            nn.GELU()
        )
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
        pooled_output = self.dense1(first_token_tensor)
        x = self.activation(pooled_output)
        x = torch.add(torch.mul(x[:,self.hidden_size:],random.uniform(0,1)),x[:,:self.hidden_size]) #update
        x = torch.cat((x, data.p_num, data.text_len), dim=1)
        x = F.dropout(F.relu(self.conv1(x, edge_index, edge_weight)),
                    p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.post_process_layers(x)

        return F.log_softmax(x, dim=1)
    def __repr__(self):
        return "BERTxGAT"