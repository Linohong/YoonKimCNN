import torch
import torch.nn as nn
import torch.nn.functional as F
import varPack as D

torch.manual_seed(1)

'''
README : Network.py
It defines CNN Network for processing (training, developing, testing) sentiment analysis data
'''

class CnnNetwork(nn.Module) :
    def __init__(self) :
        super(CnnNetwork, self).__init__()
        # Embedding layer
        self.embeddings = nn.Embedding(D.vocab_size + 2, D.EMBEDDING_DIM)
        # Kernel Operation
        self.conv1 = nn.Conv2d(1, D.FEATURE_SIZE, (5, D.EMBEDDING_DIM) )
        # Affine Operation
        self.fc1 = nn.Linear(D.FEATURE_SIZE, 2)

    def forward(self, x) :
        # Max pooling over a (2, 2) window
        x = self.embeddings(x).view((1, 1, D.max_sent_len, -1))
        x = F.max_pool2d(F.relu(self.conv1(x)), (D.max_sent_len-4, 1) )
        # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        prob = F.softmax(x, dim=1)
        return prob

    def num_flat_features(self, x) :
        size = x.size()[1:]
        num_features = 1
        for s in size :
            num_features *= s
        return num_features
