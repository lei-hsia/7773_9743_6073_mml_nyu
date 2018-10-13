import numpy as np 
import matplotlib.pyplot as plt 

N = 4
lambdas = [4,4,4,4,4]
# X_T: is ONE increment of Poisson process, i.e. #events happened in one time unit
X_T = [np.random.poisson(lam, size=N) for lam in lambdas]
S = [[np.sum(X[0:i]) for i in range(N)] for X in X_T]
k = np.linspace(0,N,N)
'''X 就是给定一个lambda & 总的steps poisson process'''

# Plot the graph

graphs = [plt.step(k, S[i], label="Lambda = %d"%lambdas[i])[0] for i in range(len(lambdas))]
plt.legend(handles=graphs, loc=2)
plt.title("Poisson Process", y=1.03)
plt.ylim(0)
plt.xlim(0)
plt.show()