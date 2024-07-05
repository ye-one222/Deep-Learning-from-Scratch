# 활성화 함수들 모음
import numpy as np

def sigmoid(x):             
  return 1/(1+np.exp(-x))

def softmax(a): # a는 array형
  c = np.max(a) # 오버플로우 문제 해결하기 위해
  exp_a = np.exp(a-c) #exp_a는 array형
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y

def relu(x):
  return np.maximum(0,x)