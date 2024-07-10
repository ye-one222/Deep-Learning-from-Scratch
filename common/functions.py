'''활성화 함수들'''
import numpy as np

def sigmoid(x):             
  return 1/(1+np.exp(-x))

def softmax(x): # x는 array형
  '''
  c = np.max(a) # 오버플로우 문제 해결하기 위해
  exp_a = np.exp(a-c) #exp_a는 array형
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y
  '''
  if x.ndim == 2: #?????
    x = x.T
    x = x - np.max(x, axis=0)
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T 

  x = x - np.max(x) # 오버플로 대책
  return np.exp(x) / np.sum(np.exp(x))

def relu(x):
  return np.maximum(0,x)


'''손실 함수들'''
def cross_entropy_error(y,t):
  delta = 1e-7

  ## batch 사용
  if y.ndim == 1: # 원래처럼 하나의 데이터가 들어오는 경우도 지원하기 위해
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  batch_size = y.shape[0]
  
  # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
  if t.size == y.size:
    t = t.argmax(axis=1)

  ## 만약 one-hot 인코딩이 false인 경우라면
  return -np.sum(np.log(y[np.arange(batch_size),t] + delta)) / batch_size # 정답 레이블에 해당하는 출력만 사용하면 되기 때문


# 이건뭐지
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)