import torch
import torch.nn as nn
import torch.nn.functional as F
import varPack as D

torch.manual_seed(1)

'''
README : Network_wEmbeddingLayer.py
It defines CNN Network for processing (training, developing, testing) sentiment analysis data
This network is based on the paper from Yoon Kim, 2014. 
'''

class CnnNetwork(nn.Module) :
    def __init__(self) :
        super(CnnNetwork, self).__init__()
        # embedding layer
        self.embeddings = nn.Embedding(D.vocab_size, D.EMBEDDING_DIM)
        self.embeddings.weight.data.copy_(torch.from_numpy(D.wordEmbedding))

        # Kernel Operation
        self.conv1 = nn.Conv2d(1, D.FEATURE_SIZE, (3, D.EMBEDDING_DIM) )
        self.conv2 = nn.Conv2d(1, D.FEATURE_SIZE, (4, D.EMBEDDING_DIM) )
        self.conv3 = nn.Conv2d(1, D.FEATURE_SIZE, (5, D.EMBEDDING_DIM) )
        # Dropout & Linear
        self.drp = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(D.FEATURE_SIZE * 3, 2)

    def forward(self, x) :
        # Get embeddings first then conv.
        x = self.embeddings(x).view(1, 1, D.max_sent_len, -1)

        # Max pooling over a (2, 2) window
        x_3 = F.max_pool2d(F.relu(self.conv1(x)), (D.max_sent_len-2, 1) ) # second argument for the max-pooling size
        x_4 = F.max_pool2d(F.relu(self.conv2(x)), (D.max_sent_len-3, 1) )
        x_5 = F.max_pool2d(F.relu(self.conv3(x)), (D.max_sent_len-4, 1) )

        # If the size is a square you can only specify a single number
        x_3 = x_3.view(-1, self.num_flat_features(x_3))
        x_4 = x_4.view(-1, self.num_flat_features(x_4))
        x_5 = x_5.view(-1, self.num_flat_features(x_5))

        x = torch.cat((x_3, x_4, x_5), 1)
        # Fully connected layer with dropout and softmax output
        x = self.drp(x)
        x = self.fc1(x)
        prob = F.softmax(x, dim=1)

        return prob

    def num_flat_features(self, x) :
        size = x.size()[1:]
        num_features = 1
        for s in size :
            num_features *= s
        return num_features

