from NAND_gate import NAND
from OR_gate import OR
from AND_gate import AND

def XOR(x1, x2):
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)
  return y