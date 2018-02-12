import torch
import torch.nn as nn
import torch.optim as optim
import Network_wEmbeddingLayer as Net
import varPack as D
import numpy as np
from random import shuffle
from torch.autograd import Variable
from sklearn.model_selection import KFold


shuffled = D.text_and_label
shuffled = [ [D.text_and_label[0][i], D.text_and_label[1][i]] for i in range(len(D.text_and_label[0]))]
shuffle(shuffled)

# definition for K-Folds


kf = KFold(n_splits=10)
kf.get_n_splits(shuffled)

scores = []
k=0
print('Training Start ...')
for train_index, test_index in kf.split(shuffled) :
    print('now : [%d] Fold' % k)
    CNN = Net.CnnNetwork()
    optimizer = optim.SGD(CNN.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.BCELoss()
    CNN.cuda()

    k=k+1
    # @ Cross Validation : Train Part
    for epoch in range(10):
        for i, index in enumerate(train_index) :
            label = shuffled[index][1]
            sent = shuffled[index][0]

            # make the input
            input = D.makeInput(sent)
            if ( input == -1 ) :
                continue

            # Wrap Input inside the Variable
            input = Variable(torch.cuda.LongTensor(input))
            label = Variable(torch.cuda.FloatTensor(label))

            # Clear the Buffer
            optimizer.zero_grad()

            # Forward, Backward, Optimize
            output = CNN(input)
            loss = criterion(output, label) # label must be the list of one integer of target index
            loss.backward()
            optimizer.step()

    # @ Cross Validation : Test Part
    got_right = 0
    for i, index in enumerate(test_index) :
        label = shuffled[index][1]
        sent = shuffled[index][0]

        # make the input
        input = D.makeInput(sent)
        if (input==-1) :
            continue

        input = Variable(torch.cuda.LongTensor(input))
        label = Variable(torch.cuda.FloatTensor(label))
        optimizer.zero_grad()

        output = CNN(input)
        loss = criterion(output, label)

        if ((output.data[0][0] >= output.data[0][1] and label.data[0] == 1) or (output.data[0][0] < output.data[0][1] and label.data[1] == 1)) :
            got_right = got_right + 1

    cur_score = float(got_right)/len(test_index)
    scores.append(cur_score)
    print('[%d] Cross Validation - Accuracy : %.3f' % (k, cur_score))


print('*****************************************')
print('*****************FINALLY*****************')
print('Accuracy cross validated : %.3f (+/- %0.2f)' % (np.mean(scores), np.std(scores) * 2))
print('*****************************************')
print('Done Training !')

# Writing to File
print('Saving the Model...')
torch.save(CNN.state_dict(), './savedCNN')
torch.save(CNN, './saveEntireCNN')





