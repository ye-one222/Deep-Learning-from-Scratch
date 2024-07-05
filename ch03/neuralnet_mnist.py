# MNIST 데이터셋을 이용하여 추론을 진행하는 신경망을 구현해보자

# 이 신경망은 입력층:784개, 출력층:10개, 1층:50개, 2층:100개 뉴런을 가짐
## 이미지 크기가 28*28이므로 입력층 뉴런이 784개
## 0~9 숫자를 구별하는 문제이므로 출력층 뉴린이 10개
## 은닉층 뉴런 개수는 그냥 정한 것임
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax, relu


def get_data():
  (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
  return x_test, t_test


# sample_weight.pkl에 저장된 학습된 가중치 매개변수 읽어옴 (가중치, 편향 매개변수가 딕셔너리 변수로 저장되어 있음)
def init_network():
  file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_weight.pkl") # 디렉토리 명시해야 오류 안남
  with open(file_path, 'rb') as f:
    network = pickle.load(f)
  return network


def predict(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  # 활성화 함수로 시그모이드 사용
  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = softmax(a3)

  '''
  # 활성화 함수로 ReLU 사용 -> 0.8415 => 왜 더 낮게 나오지..? => "렐루 함수가 일반적으로 더 높은 정확도를 제공하는 경향이 있음, 일반적인 답변은 할 수 없음"
  a1 = np.dot(x, W1) + b1
  z1 = relu(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = relu(a2)
  a3 = np.dot(z2, W3) + b3
  y = softmax(a3)
  '''

  return y


x, t = get_data()
network = init_network()

batch_size = 100 # batch(묶음) 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size):  # len(x)=10000
  x_batch = x[i:i+batch_size] # 100장씩 묶어서 꺼냄
  y_batch = predict(network, x_batch)  # y는 numpy array형 (각 label의 확률을 나타냄) -> y_batch : y의 100개 묶음
  p = np.argmax(y_batch, axis=1)  # (axis=1) 행 축을 따라 확률이 가장 높은 원소의 인덱스를 얻음 => 예측 결과

  accuracy_cnt += np.sum(p == t[i:i+batch_size])  # () 내부 -> bool 배열, np.sum() -> True가 몇 개인지 합산

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # 데이터 배치 처리하면 더 효율적으로/빠르게 추론 처리