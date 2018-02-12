import numpy as np
import preprocessing

''' 
README : DataRead.py
This file is dedicated to refine and organize every un-organized data.
Data is comprised of positive and negative mood of sentences.
Every sentences including pos. and neg. is concatenated into 
one variable called x_text and its label as y. 
'''

def load_data(positive_data_file, negative_data_file):
   positive_examples = list(open(positive_data_file, "r").readlines())
   positive_examples = [s.strip() for s in positive_examples]
   negative_examples = list(open(negative_data_file, "r").readlines())
   negative_examples = [s.strip() for s in negative_examples]

   # process with rules
   x_text = positive_examples + negative_examples
   x_text = [preprocessing.clean_str(sent) for sent in x_text]
   # x_text = [sent.split(' ') for sent in x_text]

   # Generate labels
   positive_labels = [[1, 0] for _ in positive_examples]
   negative_labels = [[0, 1] for _ in negative_examples]
   y = np.concatenate([positive_labels, negative_labels], 0)

   return [x_text, y]