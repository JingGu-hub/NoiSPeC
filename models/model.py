
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn.neural_network import MLPClassifier

from models.MLP import MLP


# define Encoder
class Encoder(nn.Module):
    def __init__(self, input_length, input_dimension, embedding_size, feature_size, num_layers, num_heads=4, dropout=0.1):
        super(Encoder, self).__init__()
        self.input_length = input_length
        self.input_dimension = input_dimension
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        new_input_dimension = math.ceil(input_length/num_heads)* num_heads
        padding_length = new_input_dimension-input_length
        self.padding_length = padding_length

        self.embedding = nn.Linear(input_dimension, embedding_size)
        self.time_attention = nn.TransformerEncoderLayer(batch_first=True, d_model=embedding_size, nhead=num_heads)
        self.spatial_attention = nn.TransformerEncoderLayer(batch_first=True, d_model=new_input_dimension, nhead=num_heads)
        self.norm1 = nn.LayerNorm([new_input_dimension, embedding_size])
        self.norm2 = nn.LayerNorm([new_input_dimension, embedding_size])
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embedding_size*new_input_dimension, feature_size)


    def forward(self, x):
        pd = (0, 0, 0, self.padding_length, 0, 0) # pad -2 dim by padding_length behind
        x = F.pad(x, pd, "constant", 0)
        x = self.embedding(x)
        for attn_layer in range(self.num_layers):
            x = self.time_attention(x)
            x = F.relu(x)
            x = self.norm1(x) 
            x = self.dropout(x)
            x = x.transpose(1,2)
            x = self.spatial_attention(x)
            x = x.transpose(1,2)
            x = F.relu(x) 
            x = self.norm2(x) 
            x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.projection(x)

        return x

# define Decoder
class Decoder(nn.Module):
    def __init__(self, feature_size, seq_length, input_dimension):
        super(Decoder, self).__init__()
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.input_dimension = input_dimension
        self.linear_decoder = nn.Linear(feature_size, seq_length*input_dimension)
        
    def forward(self, x):
        x = self.linear_decoder(x)
        x = x.view(x.shape[0],self.seq_length, self.input_dimension)
        return x

# define classifier
class Classifier(nn.Module):
    def __init__(self, feature_size, class_number):
        super(Classifier, self).__init__()
        self.feature_size = feature_size
        self.class_number = class_number
        self.linear_decoder = nn.Linear(feature_size, class_number)

        self.projection_head = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, class_number),
        )

        self.mlp = MLP(input_size=feature_size, hidden_size=64, output_size=class_number)
        
    def forward(self, x):
        x = self.mlp(x)
        x = torch.sigmoid(x)
        return x


