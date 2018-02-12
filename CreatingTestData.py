import numpy as np
import preprocessing

''' 
README : CreatingTestData.py
read 1,000 sentences from the training data by turn, 
one from the positive sentences and
one from the negative sentences.
'''

def create_test_data(positive_data_file, negative_data_file):
   positive_examples = list(open(positive_data_file, "r").readlines())
   positive_examples = [s.strip() for s in positive_examples]
   negative_examples = list(open(negative_data_file, "r").readlines())
   negative_examples = [s.strip() for s in negative_examples]

   # process with rules
   x_text = []
   for i in range(500) :
       x_text.extend([positive_examples[i]])
       x_text.extend([negative_examples[i]])

   print('Writing to file (TestData)')
   file = open("../data/testData.txt", "w")
   for sent in x_text :
       file.write(sent + "\n")

   print('Done Writing !')
   print('Finish !')

pos = "../data/rt-polarity.pos.txt"
neg = "../data/rt-polarity.neg.txt"
create_test_data(pos, neg)

