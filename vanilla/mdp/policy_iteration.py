import numpy as np

R = np.array([[-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
              [-np.inf, 50,      0,       0,       500,     -np.inf],
              [-np.inf, 0,       -np.inf, 0,       -50,     -np.inf],
              [-np.inf, 0,       0,       0,       0,       -np.inf],
              [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]])

gamma = 0.5

U = np.zeros((5, 6))
U[1, 1] = -np.inf

actions = [[0, -1], [1, 0], [0, 1], [-1, 0]]

def policy(s):
    sj = s + actions
    return s + actions[np.argmax(U[tuple(sj.T)])]

for _ in range(100):
    for i in range(1, 4):
        for j in range(1, 5):
            state = np.array([i, j])
            if R[tuple(state)] != 0:
                U[tuple(state)] = R[tuple(state)]
            else:
                U[tuple(state)] = gamma * U[tuple(policy(state))]

print(U)