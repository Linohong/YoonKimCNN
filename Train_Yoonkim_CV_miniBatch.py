import torch
import torch.nn as nn
import torch.optim as optim
import Network_wEmbeddingLayer as Net
import varPack as D
import numpy as np
import torch.utils.data as data
from random import shuffle
from torch.autograd import Variable
from sklearn.model_selection import KFold

torch.manual_seed(1)

shuffled = D.text_and_label
shuffled = [ [D.text_and_label[0][i], D.text_and_label[1][i]] for i in range(len(D.text_and_label[0])) ]
shuffle(shuffled)

# definition for K-Folds
kf = KFold(n_splits=10)
kf.get_n_splits(shuffled)

scores = []
k=0
print('Training Start ...')
# for loop for k-fold
for train_index, test_index in kf.split(shuffled) :
    print('now : [%d] Fold' % (k+1))
    CNN = Net.CnnNetwork()
    optimizer = optim.Adadelta(CNN.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    CNN.cuda()

    # added to mini-batches
    trainloader = data.DataLoader(train_index, D.batch_size, shuffle=False, num_workers=8)

    k=k+1
    # @ Cross Validation : Train Part using mini-batches
    for epoch in range(10):
        for batch in trainloader :
            label = []
            sent = []
            for index in batch :
                label.append(shuffled[index][1])
                sent.append(shuffled[index][0])

            # make the input
            input = []
            for i in range(len(label)) :
                input.append(D.makeInput(sent[i]))

            # Wrap Input inside the Variable
            input = Variable(torch.cuda.LongTensor(D.batch_size, input))
            label = Variable(torch.cuda.FloatTensor(D.batch_size, label))

            # Clear the Buffer
            optimizer.zero_grad()

            # Forward, Backward, Optimize
            output = CNN(input)
            loss = criterion(output, label) # label must be the list of one integer of target index
            loss.backward()
            optimizer.step()

    # @ Cross Validation : Test Part using mini-batches
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

        output = CNN(input)
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





