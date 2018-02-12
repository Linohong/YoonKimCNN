import torch
import torch.functional as F
from torch.autograd import Variable
import torch.nn as nn

m = nn.Dropout(p=0.2)
input = Variable(torch.randn(5,5))
output = m(input)
print(output)

