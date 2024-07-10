# 신경망에서의 기울기를 구해보자

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화, W는 형상이 2X3인 가중치 매개변수

    def predict(self, x): # 예측을 수행하는 함수
        return np.dot(x, self.W)

    def loss(self, x, t): # 손실함수의 값 구하는 함수
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)