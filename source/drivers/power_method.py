import math
import numpy as np
from numpy.linalg import norm

def max_eig(A):
    assert(len(A.shape) == 2)
    v = np.random.rand(A.shape[1])

    for _ in range(1, 20):
        u = np.copy(v)
        v = A.dot(v)
        print(math.acos(u.dot(v) / (norm(u) * norm(v))))
        v = v / norm(v)
        print(v)
    return v

def main():
    A = np.array([[2, -12], [1, -5]])
    max_eig(A)
    print(np.linalg.eig(A))

if __name__ == "__main__":
    main()
