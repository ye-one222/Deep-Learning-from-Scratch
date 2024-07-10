# 경사 하강법을 간단하게 구현해보자

import numpy as np
from numerical_gradient import numerical_gradient

def gradient_descent(f, init_x, lr=0.01, step_num=100):
  '''
  f : 최적화하려는 함수 = 목적 함수 (손실 함수)
  init_x : 초깃값
  lr : 학습률
  step_num : 경사법에 따른 반복 횟수
  '''
  x = init_x

  for i in range(step_num):
    grad = numerical_gradient(f,x)
    x -= lr * grad

  return x

def function_2(x):
  return x[0]**2 + x[1]**2

# lr = 0.1
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))

# lr = 0.01 -> 0.1일 때보다는 정답에 덜 근접
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.01, step_num=100))

# lr이 너무 큰 경우 -> 발산
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))

# lr이 너무 작은 경우 -> 갱신 거의 안 되고 끝남
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))