import torch
import varPack as D
import preprocessing
from torch.autograd import Variable


CNN = torch.load('./saveEntireCNN')
test_data = list(open('../data/testData.txt', "r").readlines())
test_data = [preprocessing.clean_str(sent) for sent in test_data]

f = open('result.txt', 'w')
got_right=0
for i, sent in enumerate(test_data):
    # Input
    words_in_sent = []
    words_in_sent.extend(sent.split())
    s = len(words_in_sent)
    if (s > D.max_sent_len):
        continue

    input = []
    for word in words_in_sent:
        try:
            input.extend([D.word_to_ix[word]])
        except KeyError:
            print("No Such Word in dictionary, zero padding")
            input.extend([D.word_to_ix['_UNSEEN_']])
    for _ in range(D.max_sent_len - s):
        input.extend([D.word_to_ix['_ZEROS_']])

    # Wrap Input inside the Variable
    input = Variable(torch.cuda.LongTensor(input))

    # Forward, Backward, Optimize
    output = CNN(input)


    if (output.data[0][0] > output.data[0][1]) :
        f.write("positive" + "\n")
        if ( i%2 == 0 ) :
            got_right = got_right+1
    else :
        f.write("negative" + "\n")
        if ( i%2 != 0 ) :
            got_right = got_right+1

f.write('Accuracy : ' + str(got_right/1000.0))