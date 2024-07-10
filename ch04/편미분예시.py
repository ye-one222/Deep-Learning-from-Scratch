import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 수치 미분 함수
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    for idx in range(x.size):
        temp_val = x[idx]
        
        # f(x + h)
        x[idx] = float(temp_val) + h
        fxh1 = f(x)
        
        # f(x - h)
        x[idx] = temp_val - h 
        fxh2 = f(x) 
        
        # 편미분 계산
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = temp_val  # 값 복원
        
    return grad

# 변수 2개 -> 편미분 함수
def function_2(x):
    return np.sum(x**2, axis=0)  # x[0]**2 + x[1]**2

# x1, x2 데이터 생성
x1 = np.arange(-3.0, 3.0, 0.1)
x2 = np.arange(-3.0, 3.0, 0.1)
X1, X2 = np.meshgrid(x1, x2)
Z = function_2(np.array([X1, X2]))

# 3D 그래프 그리기
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
plt.show()
