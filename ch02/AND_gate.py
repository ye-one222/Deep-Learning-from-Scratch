### 편향 X ###
"""
def AND(x1, x2):
  w1, w2, theta = 0.5, 0.5, 0.7
  tmp = x1*w1 + x2*w2
  if tmp <= theta:
    return 0
  elif tmp > theta:
    return 1
"""

### 편향 O ###
import numpy as np

def AND(x1, x2):
  x = np.array([x1, x2])      # 입력
  w = np.array([0.5, 0.5])    # 가중치
  b = -0.7                    # 편향
  tmp = np.sum(x*w) + b
  if tmp <= 0:
    return 0
  else:
    return 1