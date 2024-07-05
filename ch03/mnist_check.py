# MNIST 데이터를 가져와보자
import sys
import os
import pickle
import numpy as np

# 현재 파일이 속한 디렉토리의 부모 디렉토리를 모듈 검색 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataset.mnist import load_mnist

# MNIST 데이터셋 로드
# 이미지를 numpy 배열로 저장
## flatten : 입력 이미지를 평탄화(1차원 배열로)
## normalize : 입력 이미지를 정규화(0.0~1.0 사이의 값으로)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 각 데이터의 형상 출력
print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000,)
print(x_test.shape)  # (10000, 784)
print(t_test.shape)  # (10000,)
