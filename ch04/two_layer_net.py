# 2층 신경망 클래스 구현

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        '''
        초기화 수행 (인수 차례대로 : 입력층 뉴런 수, 은닉층 뉴런 수, 출력층 뉴런 수)
        '''
        # 가중치 초기화
        self.params = {}                                                                # 가중치 매개변수를 저장하는 변수
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)  # 1번째 층의 가중치
        self.params['b1'] = np.zeros(hidden_size)                                       # 1번째 층의 바이어스
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) # 2번째 층의 가중치
        self.params['b2'] = np.zeros(output_size)                                       # 2번째 층의 바이어스

    def predict(self, x):
        '''
        추론 수행 (x는 이미지 데이터)
        '''
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    def loss(self, x, t):
        '''
        손실 함수의 값 구함 (x : 입력 (이미지)데이터, t : 정답 레이블)
        '''
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        '''
        정확도 구함
        '''
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def numerical_gradient(self, x, t):
        '''
        가중치 매개변수의 기울기를 구함 (x : 입력 데이터, t : 정답 레이블)
        '''
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}                                                  # params 변수에 대응하는 각 매개변수의 기울기 저장
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
       
    def gradient(self, x, t):
        '''
        가중치 매개변수의 기울기를 구함 (numerical_gradient의 성능 개선ver) -> ch5에서 배울거임
        '''
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
        