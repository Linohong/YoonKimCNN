import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('../../Word2Vec/GoogleNews-vectors-negative300.bin', binary=True)

print(model.wv["n't"])