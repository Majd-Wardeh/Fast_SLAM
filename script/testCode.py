import numpy as np
from numpy import pi, abs

# ratio = 0.1
# n = 4
# x = ratio*pi + n*pi
# x = -x
# print(x)

# y = x%(2*pi) if x >= 0 else -(-x%(2*pi)) 
# print(y)
# print(ratio*pi)


phai = -4.42790375558
print(phai)

# if abs(phai) > pi:
#     phai = -(phai%(0.5*pi)) if phai >= 0 else (-phai%(0.5*pi))

if phai > pi:
    phai = phai - 2*pi

elif phai < -pi:
    phai = phai + 2*pi 

print(phai)