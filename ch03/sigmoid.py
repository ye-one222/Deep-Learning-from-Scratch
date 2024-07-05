import numpy as np
from draw_function import draw_function

def sigmoid(x):             
  return 1/(1+np.exp(-x))  

x=np.arange(-5.0,5.0,0.1)
y=sigmoid(x)
draw_function(x,y)