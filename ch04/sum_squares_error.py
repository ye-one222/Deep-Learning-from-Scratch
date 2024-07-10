import numpy as np

def sum_sqaures_error(y,t):
  return 0.5 * np.sum((y-t)**2) # **2 = 각 원소에 대해 제곱

# 정답 == 추정
t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
print("정답:2, 추정:2")
print(sum_sqaures_error(np.array(y), np.array(t)))

# 정답 != 추정
y = [0.1,0.1,0.2,0.0,0,0.1,0.1,0.21,0.09,0.1]
print(np.sum(np.array(y))," / ",np.argmax(np.array(y)))
print("정답:2, 추정:7")
print(sum_sqaures_error(np.array(y), np.array(t)))

#정답 == 추정 but SSE 결과값은 2번째 경우보다 더 큼
y = [0.1,0.125,0.15,0.05,0.1,0.1,0.1,0.1,0.1,0.075]
print(np.sum(np.array(y))," / ",np.argmax(np.array(y)))
print("정답:2, 추정:2")
print(sum_sqaures_error(np.array(y), np.array(t)))