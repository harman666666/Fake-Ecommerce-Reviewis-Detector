import torch
from torch import nn
import numpy as np
from amazon_review_data import *

class FakeReviewDetector(nn.Module):
    def __init__(self, configuration, vocabulary_size, word_embeddings):
        super(FakeReviewDetector, self).__init__()
        
        self.CONFIGURATION = configuration
        
        # Embedding Layer
        self.embeddings = nn.Embedding(vocabulary_size, self.CONFIGURATION["embed_size"])

        # Do not train the embeddings layer because we are using pretrained embeddings
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        
        self.lstm = nn.LSTM(input_size = self.CONFIGURATION["embed_size"],
                            hidden_size = self.CONFIGURATION["hidden_size"],
                            num_layers = 1,
                            bidirectional = True)
        
        self.fc = nn.Linear(
             self.CONFIGURATION["hidden_size"] * 2 + 6,
             self.CONFIGURATION["output_size"]
        )

        self.softmax = nn.Softmax()
                   
    def forward(self, x):
        
        text, product_cat, rating, length, exclamations, title_sent, text_sent = x
        text_embedding = self.embeddings(text)
        
        # Run LSTM over text embedding sequence
        output, (h_n, _) = self.lstm(text_embedding)
        
        # hidden state below has multiple layers due to forward and backward direction from
        # bidirectional lstm
        lstm_final_hidden_state = h_n 
        
        # Concatenate the forward and backward hidden state outputs from the bidirectional lstm. 
        concatenated_hidden_state = torch.cat([lstm_final_hidden_state[i,:,:] for i in range(lstm_final_hidden_state.shape[0])], dim=1)
                
        all_features_concatenated = torch.cat((product_cat.unsqueeze(1).type(torch.FloatTensor), 
                               rating.unsqueeze(1).type(torch.FloatTensor),
                               length.unsqueeze(1).type(torch.FloatTensor),
                               exclamations.unsqueeze(1).type(torch.FloatTensor),
                               title_sent.unsqueeze(1).type(torch.FloatTensor),
                               text_sent.unsqueeze(1).type(torch.FloatTensor),
                               concatenated_hidden_state), dim=1) 
        

        output_from_fc_net = self.fc(all_features_concatenated)
        return self.softmax(output_from_fc_net) 
