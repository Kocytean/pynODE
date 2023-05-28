import numpy as np

# u_ = sum(v*sigmoid(wx + kt +b))

class System:
	pre_sigmoid_max = 5000
	N_x = 10
	N_t = 10
	T = 1
	lr = 0.001
	alpha = 0.001

def sigmoid(x, derivative = 0):
	sigm = 1./(1. + np.exp(-x.clip(min = -System.pre_sigmoid_max, max = System.pre_sigmoid_max)))
	if derivative==1:
		return np.multiply(sigm,(1 - sigm))
	elif derivative==2:
		return np.multiply(np.multiply(sigm , 1 - sigm),(1 - 2*sigm))
	elif derivative==3:
		return np.multiply(np.multiply(sigm, 1-sigm), 1 + 6*np.multiply(sigm, sigm) - 6*sigm)
	return sigm

class Net:
	def __init__(self, fanout):
		self.m = fanout
		self.w = np.random.ranf(fanout)/np.sqrt(fanout)
		self.k = np.random.ranf(fanout)/np.sqrt(fanout)
		self.v = np.random.ranf(fanout)/np.sqrt(fanout)
		self.b = np.random.ranf(fanout)
		self.dw = np.zeros(fanout)
		self.dk = np.zeros(fanout)
		self.dv = np.zeros(fanout)
		self.db = np.zeros(fanout)


	def run(self, x, t, derivative = 0):
		if derivative:
			if np.isscalar(x):
				return np.matmul(sigmoid((self.w*x + (self.k*t + self.b)), derivative)*np.power(self.w, derivative), self.v)
			else:
				return np.matmul(sigmoid((np.outer(x, self.w) + (self.k*t + self.b)), derivative)*np.power(self.w, derivative), self.v)

		if np.isscalar(x):
			return np.matmul(sigmoid((self.w*x + (self.k*t + self.b))), self.v)
		else:
			return np.matmul(sigmoid((np.outer(x, self.w) + (self.k*t + self.b))),self.v)

	def train(self, x, t, f, tau=5, update=True):
		h = System.T/System.N_t
		Ei = np.zeros(self.m)
		scale = 2*System.lr/(System.N_t*System.N_x*self.m)
		for i in range(self.m):
			w = self.w[i]
			v = self.v[i]
			b = self.b[i]
			k = self.k[i]

			#t = 0
			Z = w*x + b
			A = sigmoid(Z)
			A_1 = A*(1-A)
			N = self.run(x, 0)
			s = tau*scale*System.N_t*N*v
			self.dw[i] -=sum(s*x*A_1)
			self.dv[i] -=sum(s*A/v)
			self.db[i] -=sum(s*A_1)
			Ei[i] +=sum(N*N)
			
			for t_ in t:
				#x = 0
				Z = k*t_ + b
				A = sigmoid(Z)
				A_1 = A*(1-A)
				N = self.run(0, t_)
				s = tau*scale*N*v
				self.dk[i] -=s*t_*A_1
				self.dv[i] -=s*A/v
				self.db[i] -=s*A_1
				Ei[i] +=N*N
				#x = 1
				Z = w + k*t_ + b
				A = sigmoid(Z)
				A_1 = A*(1-A)
				N = self.run(1, t_)
				s = tau*scale*N*v
				self.dw[i] -=s*A_1
				self.dk[i] -=s*t_*A_1
				self.dv[i] -=s*A/v
				self.db[i] -=s*A_1
				Ei[i] +=N*N
			
			Z = np.array([[w*x_ + k*t_ + b for x_ in x] for t_ in t])
			A = sigmoid(Z)
			A_1 = A*(1-A)
			A_2 = A_1*(1-2*A)
			A_3 = A_1*(1-6*A_1)
			dE_dv = A_1*k - A_2*w*w
			E = dE_dv*v - f
			dE_dw = v*(A_2*x*k-2*w*A_2 - w*w*x*A_3)
			dE_dk = v*(A_1 + (t*((k*A_2 + w*w*A_3).transpose() )).transpose())
			dE_db = v*(k*A_2 + w*w*A_3)
			scale = 2*System.lr/(System.N_x*System.N_t*self.m)
			Ei[i]+= sum(sum(E*E))

			self.dv[i]-=scale*sum(sum(dE_dv*E))
			self.dw[i]-=scale*sum(sum(dE_dw*E))
			self.dk[i]-=scale*sum(sum(dE_dk*E))
			self.db[i]-=scale*sum(sum(dE_db*E))

		if update:
			scale = System.alpha/(System.N_x*System.N_t*self.m)
			self.w += self.dw - self.w*scale
			self.k += self.dk - self.k*scale
			self.v += self.dv - self.v*scale
			self.b += self.db 
			self.dw = np.zeros(self.m)
			self.dk = np.zeros(self.m)
			self.dv = np.zeros(self.m)
			self.db = np.zeros(self.m)
		return Ei

	def test(self, x, t, u):
		errors = []
		for i, t_ in enumerate(t):
			u_ = self.run(x, t_)
			e= u[i] - u_
			errors.append(e*e)
		return np.array(errors)



