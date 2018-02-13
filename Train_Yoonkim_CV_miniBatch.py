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
shuffled = [ [D.text_and_label[0][i], D.text_and_label[1][i]] for i in range(len(D.text_and_label[0])) ] # text_and_label[0][i] are sentence composed of indices of words
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
    #optimizer = optim.Adadelta(CNN.parameters(), lr=0.001)
    optimizer = optim.Adam(CNN.parameters()

                           )
    criterion = nn.BCELoss()
    CNN.cuda()

    # added to mini-batches
    x_train, y_train = np.array([lst[0] for lst in shuffled]), np.array([lst[1] for lst in shuffled])
    x_train, y_train = torch.from_numpy(x_train[train_index]).long(), torch.from_numpy(y_train[train_index]).float()
    k_train_data = torch.utils.data.TensorDataset(x_train, y_train)
    trainloader = data.DataLoader(k_train_data, batch_size=D.batch_size, shuffle=False, num_workers=8) # check if trainloader set right

    k=k+1
    # @ Cross Validation : Train Part using mini-batches
    for epoch in range(10):
        for (batch_sent, batch_label) in trainloader :
            # Wrap Input inside the Variable
            input = Variable(batch_sent.cuda())
            label = Variable(batch_label.cuda())
            #input = Variable(torch.cuda.LongTensor(batch_sent))
            #label = Variable(torch.cuda.FloatTensor(batch_label))

            # Clear the Buffer
            optimizer.zero_grad()

            # Forward, Backward, Optimize
            output = CNN(input)
            loss = criterion(output, label) # label must be the list of one integer of target index
            loss.backward()
            optimizer.step()

        # @ Cross Validation : Test Part using mini-batches

        k_test_data = np.array(shuffled)[test_index]
        k_test_data = k_test_data.tolist()
        got_right = 0
        for i, batch in enumerate(k_test_data) :
            input = Variable(torch.cuda.LongTensor(batch[0]))
            label = Variable(torch.cuda.FloatTensor(batch[1]))

            output = CNN(input)
            if ((output.data[0][0] >= output.data[0][1] and label.data[0] == 1) or (output.data[0][0] < output.data[0][1] and label.data[1] == 1)) :
                got_right = got_right + 1

        cur_score = float(got_right)/len(k_test_data)
        scores.append(cur_score)
        print('[%d] Cross Validation [%d] epoch - Accuracy : %.3f' % (k, epoch, cur_score))


print('*****************************************')
print('*****************FINALLY*****************')
print('Accuracy cross validated : %.3f (+/- %0.2f)' % (np.mean(scores), np.std(scores) * 2))
print('*****************************************')
print('Done Training !')

# Writing to File
print('Saving the Model...')
torch.save(CNN.state_dict(), './savedCNN')
torch.save(CNN, './saveEntireCNN')





