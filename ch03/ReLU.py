import numpy as np
from draw_function import draw_function

def relu(x):
  return np.maximum(0,x)

x=np.arange(-5.0,5.0,0.1)
y=relu(x)
draw_function(x,y)