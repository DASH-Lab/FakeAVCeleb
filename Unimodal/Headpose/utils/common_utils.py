import numpy as np
import math



def cosine_sim(x, y):
    divisor = x[0]*y[0]+x[1]*y[1]+x[2]*y[2]
    divider = math.sqrt(x[0]**2+x[1]**2 + x[2]**2) * math.sqrt(y[0]**2+y[1]**2+y[2]**2)
    return divisor/divider


def cosine_dist(x, y):
    return 1-cosine_sim(x,y)