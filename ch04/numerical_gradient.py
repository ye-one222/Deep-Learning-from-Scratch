# 기울기를 구현해 보자

import numpy as np

def numerical_gradient(f, x):
  h = 1e-4
  grad = np.zeros_like(x) # x와 형상이 같은, 원소가 모두 0인 배열을 생성

  for idx in range(x.size):
    tmp_val = x[idx]

    # f(x+h) 계산
    x[idx] = tmp_val + h
    fxh1 = f(x) # 변경된 x 값을 함수 f에 입력하여 함숫값을 계산함

    # f(x-h) 계산
    x[idx] = tmp_val - h
    fxh2 = f(x)

    grad[idx] = (fxh1 - fxh2) / (2*h) # 중심 차분 이용
    x[idx] = tmp_val  # 값 복원

  return grad