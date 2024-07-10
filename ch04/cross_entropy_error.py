import numpy as np

def cross_entropy_error(y,t):
  delta = 1e-7  # log 내부가 0이 되지 않도록 하기 위해 아주 작은 값(10^-7)을 더함`
  ## batch 사용 X
  # return -np.sum(t * np.log(y+delta))

  ## batch 사용
  if y.ndim == 1: # 원래처럼 하나의 데이터가 들어오는 경우도 지원하기 위해
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  batch_size = y.shape[0]
  return -np.sum(t * np.log(y+delta)) / batch_size
  ## 만약 one-hot 인코딩이 false인 경우라면
  # return -np.sum(np.log(y[np.arange(batch_size),t] + delta)) / batch_size # 정답 레이블에 해당하는 출력만 사용하면 되기 때문


# 정답 == 추정
t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
print("정답:2, 추정:2")
print(cross_entropy_error(np.array(y), np.array(t)))

# 정답 != 추정
y = [0.1,0.1,0.2,0.0,0,0.1,0.1,0.21,0.09,0.1]
print(np.sum(np.array(y))," / ",np.argmax(np.array(y)))
print("정답:2, 추정:7")
print(cross_entropy_error(np.array(y), np.array(t)))

#정답 == 추정 but CEE 결과값은 2번째 경우보다 더 큼
y = [0.1,0.125,0.15,0.05,0.1,0.1,0.1,0.1,0.1,0.075]
print(np.sum(np.array(y))," / ",np.argmax(np.array(y)))
print("정답:2, 추정:2")
print(cross_entropy_error(np.array(y), np.array(t)))