import torch
import torch.nn as nn
import torch.optim as optim
import Network_with_Embedding as Net
import varPack as D
from random import shuffle
from torch.autograd import Variable

CNN = Net.CnnNetwork()
optimizer = optim.SGD(CNN.parameters(), lr=0.001, momentum=0.9)
criterion = nn.BCELoss()

# run on GPU
CNN.cuda()
shuffled = D.text_and_label
shuffled = [ [D.text_and_label[0][i], D.text_and_label[1][i]] for i in range(len(D.text_and_label[0]))]
shuffle(shuffled)

for epoch in range(30) :
    running_loss  = 0.0
    for i, curData in enumerate(shuffled) :
        # Input
        label = curData[1]
        sent = curData[0]
        words_in_sent = []
        words_in_sent.extend(sent.split())
        s = len(words_in_sent)
        if ( s > D.max_sent_len ) :
            print("Longer sentence more than max_sent_len, so skipped")
            continue

        input = []
        for word in words_in_sent :
            try :
                input.extend([D.word_to_ix[word]])
            except KeyError :
                print("No Such Word in dictionary, zero padding")
                input.extend([D.word_to_ix['HAS_TO_BE_UNK']])
        for _ in range(D.max_sent_len - s) :
            input.extend([D.word_to_ix['HAS_TO_ZERO_PAD']])

        # Wrap Input inside the Variable

        input = Variable(torch.cuda.LongTensor(input))
        label = Variable(torch.cuda.FloatTensor(label))

        # Clear the Buffer
        optimizer.zero_grad()

        # Forward, Backward, Optimize
        output = CNN(input)
        m = nn.Sigmoid()
        loss = criterion(m(output), label) # label must be the list of one integer of target index
        loss.backward()
        optimizer.step()

        # Print the Results (Statistics)
        running_loss += loss.data[0]
        if i % 2000 == 0 :
            # print average loss every 2000 steps
            print('[%d]epoch, [%5d] Step - loss : %.3f' % (epoch, i, running_loss/2000))
            running_loss = 0



print('Done Training !')

# Writing to File
print('Saving the Model...')
torch.save(CNN.state_dict(), './savedCNN')
torch.save(CNN, './saveEntireCNN')





