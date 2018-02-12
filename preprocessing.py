import re

'''
## README : preprocessing.py
This file preprocesses unrefined data as a refined data with 
restrictions coded below.
'''

def clean_str(string, remove_dot=True):
  """
  Tokenization/string cleaning for all datasets except for SST.
  Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
  """
  if remove_dot:
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  else:
    string = re.sub(r"[^A-Za-z0-9().,!?\'\`]", " ", string)

  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  # string = re.sub(r"\(", " \( ", string)
  # string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()