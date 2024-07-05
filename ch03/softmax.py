import numpy as np

def softmax(a): # a는 array형
  c = np.max(a) # 오버플로우 문제 해결하기 위해
  exp_a = np.exp(a-c) #exp_a는 array형
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y