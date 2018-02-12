import DataRead
import gensim
import numpy as np
import preprocessing

'''
# README : varPack.py
This file is like a header file in C programming.
It defines and holds variables that are used in several different 
states of the whole process. 
'''

# RAW TEXT DATA
pos = "../data/rt-polarity.pos.txt"
neg = "../data/rt-polarity.neg.txt"
text_and_label = DataRead.load_data(pos, neg)
text = text_and_label[0]
label = text_and_label[1]

# VOCABULARY
vocab = []
for sent in text :
    vocab.extend(sent.split())
vocab.extend(['_UNSEEN_'])
vocab.extend(['_ZEROS_'])
vocab = set(vocab)

# Read WordEmbedding from file
EMBEDDING_DIM = 300
zeros = [0] * EMBEDDING_DIM
print('Started loading word2vec ...')
readEmbedding = gensim.models.KeyedVectors.load_word2vec_format('../../Word2Vec/GoogleNews-vectors-negative300.bin', binary=True)
wordEmbedding = []
word_to_ix = {word:i for i,word in enumerate(vocab)}
for key in word_to_ix :
    try :
        wordEmbedding.append(readEmbedding[key])
    except KeyError :
        if ( key == '_ZEROS_') :
            wordEmbedding.append(zeros)
        wordEmbedding.append(np.random.rand(EMBEDDING_DIM))
wordEmbedding = np.array(wordEmbedding)
del readEmbedding
print('Done word2vec loading !')

# HYPER-PARAMETERS
CONTEXT_SIZE = 2
vocab_size = len(wordEmbedding)
max_sent_len = 66
FEATURE_SIZE = 100

# etc. Funtions
def makeInput(sent) :
    words_in_sent = []
    words_in_sent.extend(sent.split())
    s = len(words_in_sent)
    if (s > max_sent_len):
        print("Longer sentence more than max_sent_len, so skipped")
        return -1

    input = []
    for word in words_in_sent:
        try:
            input.append(word_to_ix[word])
        except KeyError:
            print("No Such Word in dictionary, Give back unknown index")
            input.append(word_to_ix['_UNSEEN_'])
    for _ in range(max_sent_len - s):
        input.append(word_to_ix['_ZEROS_'])

    return input