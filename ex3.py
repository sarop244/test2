import numpy as np


def ucle(a,b):
    

    A = np.array(a)
    B = np.array(b)

    dist = np.sqrt(np.sum(np.square(A-B)))

    print(dist/68)

