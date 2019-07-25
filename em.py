
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
from mpl_toolkits.mplot3d import Axes3D
#K:the number of class<-given!!
K = 4
x = np.loadtxt("x.csv",delimiter = ",")

#N:number of data
N = len(x)
#D:dimension of each data
D = len(x[0])

#initialize paremeters

mu_mean = x.mean(0)
mu_cov = (x-mu_mean).T.dot(x-mu_mean)/len(x)
mu = np.random.multivariate_normal(mu_mean,mu_cov,size = K)
print(mu)
#sigma = 10.0*np.array([np.identity(D)]*K)+np.array([U[k].T.dot(U[k]) for k in range(K)])
sigma = np.array([mu_cov]*K)
#sigma = 1.0*np.array([np.identity(D)]*K)
pi = np.random.rand(K)
pi = pi/sum(pi)

#define likelihood
def Gamma(x,mu,sigma,pi):
	K = len(pi)
	gamma = np.zeros((len(x),K))#initialize
	N = len(x)
	for k in range(K):
		gamma[:,k] = [pi[k]*st.multivariate_normal.pdf(d,mu[k],sigma[k]) for d in x]
	for n in range(N):#normalize
		gamma[n] = gamma[n]/sum(gamma[n])
	return gamma#N*K

def likelihood(x,mu,sigma,pi):
	K = len(pi)
	N = len(x)
	L = 0.0
	for n in range(N):
		p_n = 0.0
		for k in range(K):
			p_n += pi[k]*st.multivariate_normal.pdf(x[n],mu[k],sigma[k])
		L += np.log(p_n)
	return L


#EM algorithm

iter_end = 100
L = []

for iter in range(iter_end):
	#Estimation
	L.append(likelihood(x,mu,sigma,pi))
	print("iter step = ",iter,"Likelihood=",L[-1])
	gamma = Gamma(x,mu,sigma,pi)
	N_k = gamma.sum(0)
	
	#Maximization
	pi = N_k/np.array([N]*K)
	mu = gamma.T.dot(x)#K*D
	mu = np.array([mu[k]/N_k[k] for k in range(K)])
	S = np.array([np.tensordot(x[n],x[n],axes = 0) for n in range(N)])
	S_k = np.tensordot(gamma,S,([0],[0]))
	S_k = np.array([S_k[k]/N_k[k] for k in range(K)])
	mu_mat = np.array([np.tensordot(mu[k],mu[k],axes =0) for k in range(K)])
	sigma = S_k-mu_mat#K*D*D
print("pi",pi)
print("mu",mu)
print("sigma",sigma)

np.savetxt("z.csv",gamma,delimiter = ",")

z_pred = np.argmax(gamma,axis = 1)
#scatter plot of data
"""
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
color = ["g","b","r","y"]
x_colored = []
for k in range(K):
	x_colored.append(x[z_pred == k])
	ax.plot(x_colored[k][:,0],x_colored[k][:,1],x_colored[k][:,2],"o",color = color[k])

ax.view_init(elev = 30,azim = 45)
fig.savefig("pred_em.jpg")
"""
with open("params.dat",mode = "w") as f_params:
	f_params.write("pi\n")
	np.savetxt(f_params,pi)
	f_params.write("\nmu\n")
	np.savetxt(f_params,mu)
	f_params.write("\nsigma\n")
	for k in range(K):
		np.savetxt(f_params,sigma[k])
		f_params.write("\n")

with open("Likelihood_em.txt",mode = "w") as f_likelihood:
	np.savetxt(f_likelihood,L)
