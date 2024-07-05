import numpy as np
from draw_function import draw_function

def step_function(x):
  y = x > 0                 # y의 type은 bool
  return y.astype(np.int64) # y를 bool에서 int로 변환

x=np.arange(-5.0,5.0,0.1)
y=step_function(x)
draw_function(x,y)