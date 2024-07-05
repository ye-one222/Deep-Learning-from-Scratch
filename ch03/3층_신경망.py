# 입력층 뉴런 2개
# 1층 뉴런 3개
# 2층 뉴런 2개
# 출력층 뉴런 2개
## 1,2층의 활성화 함수 : 시그모이드 함수
## 출력층의 활성화 함수 : 항등 함수

import numpy as np
from sigmoid import sigmoid

# 가중치, 편향 초기화
def init_network():
  network = {}    # 딕셔너리 변수 network 선언
  network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  network['b1'] = np.array([0.1, 0.2, 0.3])
  network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
  network['b2'] = np.array([0.1, 0.2])
  network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
  network['b3'] = np.array([0.1, 0.2])

  return network

# 신경망의 순방향 구현
def forward(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1)+b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2)+b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3)+b3
  y = a3 # 항등함수
  
  return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)