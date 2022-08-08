import torch
from torch_geometric.nn import SAGEConv
from transformers import RobertaModel
from utils import LABELS
import torch.nn.functional as F
from torch import nn
import random
import warnings
import numpy as np
warnings.filterwarnings("ignore")
class BERTxSAGE(torch.nn.Module):

    def __init__(self, n_classes=len(LABELS), hidden_size=768, dropout_rate=0.2, bert_model="roberta-base"):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.BERT = RobertaModel.from_pretrained(
            bert_model)
        self.hidden_size = hidden_size
        self.dense1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense2 = nn.Linear(self.hidden_size, 256)
        self.activation = nn.Tanh()
        self.conv1 = SAGEConv(in_channels= self.hidden_size + 6, out_channels = 512, normalize = True)
        self.conv2 = SAGEConv(in_channels = 512, out_channels= 256, normalize = True)
        self.post_process_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, affine= False),
            nn.Linear(256, n_classes)
        )
        
    def forward(self, data):
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
        text_embedding = F.relu(self.dense2(x))
        x = torch.cat((x, data.p_num, data.text_len,data.bbox), dim=1)  # update
        x = F.dropout(F.relu(self.conv1(x, edge_index, edge_weight)),
                    p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        fusion_feature = torch.cat((x, text_embedding), dim= 1)
        # graph_feature = x
        # out1 = torch.matmul(F.softmax(torch.matmul(graph_feature, text_embedding.T))/np.sqrt(text_embedding.shape[1]), text_embedding)
        # out1 = torch.add(0.1* out1, graph_feature)
        # out2 = torch.matmul(F.softmax(torch.matmul(text_embedding, graph_feature.T))/np.sqrt(graph_feature.shape[1]), graph_feature)
        # out2 = torch.add(0.1* out2, graph_feature)
        # fusion_feature = torch.cat((out1, out2), dim =1)
        x = self.post_process_layers(fusion_feature)

        return F.log_softmax(x, dim=1)
    def __repr__(self) -> str:
        return "BERTxSAGE"
