'''
## 기본 ver 수치미분
def numerical_diff(f, x):
  h = 1e-50 # =10^(-50)
  return (f(x+h)-f(x))/h
'''

## 위의 문제점 개선한 ver 수치미분
def numerical_diff(f, x):
  h = 1e-4 # =0.0001
  return (f(x+h)-f(x-h)) / (2*h)