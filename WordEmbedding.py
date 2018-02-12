import varPack as D # stands for data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(1)

trigrams = []
for sent in D.text_for_trigram :
    tmp = [ ([sent[i], sent[i+1]], sent[i+2]) for i in range(len(sent)-2) ]
    trigrams.extend(tmp)

class NGramLanguageModeler(nn.Module) :
    def __init__ (self, vocab_size, embedding_dim, context_size) :
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs) :
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=0)
        return log_probs

loss_function = nn.NLLLoss()
model = NGramLanguageModeler(D.vocab_size, D.EMBEDDING_DIM, D.CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.01)

losses = []

for epoch in range(10) :
    total_loss = torch.Tensor([0])
    print('epoch : #%d' % epoch)
    for context, target in trigrams :
        context_idxs = [D.word_to_ix[w] for w in context]
        context_var = Variable(torch.LongTensor(context_idxs))

        model.zero_grad()
        log_probs = model(context_var)
        loss = loss_function(log_probs, Variable(torch.LongTensor([D.word_to_ix[target]])))
        loss.backward()
        optimizer.step()

        total_loss += loss.data
        print(target)
    losses.append(total_loss)
print(losses)
print('Done training !')
print()


print('Writing to file...')
file = open("embedding.txt", "w")
for word in D.vocab :
    word_idx = [D.word_to_ix[word]]
    word_var = Variable(torch.LongTensor(word_idx))
    embeds = model.embeddings(word_var).view((1, -1))

    file.write(word + ":")
    for i in range(D.EMBEDDING_DIM) :
        file.write(str(embeds.data[0][i]) + " ")
    file.write('\n')
print('Done Writing !')
print('Finish !')