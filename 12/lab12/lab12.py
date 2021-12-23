import numpy as np


B = np.array([[1,10],[1,10],[10,1],[1,10],[10,1]])
Z = np.array([[1],[1],[5],[1],[5]])

A = np.linalg.inv(np.dot(np.transpose(B),B) - np.identity(2))

Amy = np.dot(A, np.dot(np.transpose(B),Z))

print(np.dot(Amy.T, np.array([[10],[1]])))