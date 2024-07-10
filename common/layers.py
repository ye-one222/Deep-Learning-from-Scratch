import numpy as np
from common.functions import *

# 활성화 함수 계층 - ReLU 계층
class Relu:
  def __init__(self):
    self.mask = None  # True/False로 구성된 넘파이 배열

  def forward(self, x):
    self.mask = (x <= 0)  # x<=0이면 True
    out = x.copy()
    out[self.mask] = 0

    return out

  def backward(self, dout):
    dout[self.mask] = 0 # mask의 원소가 True이면 dout를 0으로
    dx = dout

    return dx
  
# 활성화 함수 계층 - sigmoid 계층
class Sigmoid:
  def __init__(self):
    self.out = None # 역전파는 순전파의 출력만으로 계산할 수 있으므로, 순천파의 출력을 out에 저장해놓자

  def forward(self, x):
    out = sigmoid(x)
    self.out = out
    return out

  def backward(self, dout):
    dx = dout * (1.0 - self.out) * self.out

    return dx
  

# 어파인 계층
class Affine:
  def __init__(self, W, b):
    self.W = W
    self.b = b
    self.x = None
    self.dW = None
    self.db = None

  def forward(self, x):
    self.x = x
    out = np.dot(x, self.W) + self.b

    return out
  
  def backward(self, dout):
    dx = np.dot(dout, self.W.T)
    self.dW = np.dot(self.x.T, dout)
    self.db = np.sum(dout, axis=0)

    return dx
  

# softmax-with-loss 계층
class SoftmaxWithLoss:
  def __init__(self):
    self.loss = None # 손실함수
    self.y = None    # softmax의 출력
    self.t = None    # 정답 레이블(원-핫 인코딩 형태)
      
  def forward(self, x, t):
    self.t = t
    self.y = softmax(x)
    self.loss = cross_entropy_error(self.y, self.t)
    
    return self.loss

  def backward(self, dout=1):
    batch_size = self.t.shape[0]
    if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
        dx = (self.y - self.t) / batch_size
    else:
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx = dx / batch_size
    
    return dx