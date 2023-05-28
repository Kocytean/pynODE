import numpy as np

def mul_x(T, v):
	return np.transpose(np.transpose(T,[0,2,1])*v,[0,2,1])

def mul_t(T, v):
	return np.transpose(np.transpose(T)*v)

def sigma(z):
	return 1/(1+np.exp(-z))

def sigmas(z, third_derivative = True):
		s = 1/(1+np.exp(-z))
		s1 = s*(1-s)
		s2 = s1*(1-2*s)
		if third_derivative:
			s3 = s1*(1-6*s1)
			return s, s1, s2, s3
		return s, s1, s2

class System:

	nx = 11
	nt = 11
	tau=1
	lr= .0001

class Net():
	def __init__(self, fanout):
		self.fanout = fanout
		self.v = np.random.rand(self.fanout)
		self.w = np.random.rand(self.fanout)
		self.k = np.random.rand(self.fanout)
		self.b = np.random.rand(self.fanout)

	def run(self, x, t):
		z = np.array([[[w_*x_ + k_*t_ + b_ for w_, k_, b_ in zip(self.w, self.k, self.b)] for x_ in x] for t_ in t])
		s = sigma(z)
		return np.sum(self.v*s, axis=2)

	def train(self, x, t, u_act, f, test = False):

		z = np.array([[[w_*x_ + k_*t_ + b_ for w_, k_, b_ in zip(self.w, self.k, self.b)] for x_ in x] for t_ in t])
		zd = np.array([[[w_*x_ + k_*t_ + b_ for w_, k_, b_ in zip(self.w, self.k, self.b)] for x_ in x[1:-1]] for t_ in t[1:]])
		zx = np.array([[w_*x_ + b_ for w_,  b_ in zip(self.w, self.b)] for x_ in x[1:-1]])
		zt = np.array([[k_*t_ + b_ for k_, b_ in zip(self.k, self.b)] for t_ in t])
		z1 = np.array([[w_+k_*t_ + b_ for w_, k_, b_ in zip(self.w, self.k, self.b)] for t_ in t])

		vk= self.v*self.k
		vw=self.v*self.w
		w2 = self.w*self.w
		vw2=self.v*w2

		s, s1, s2, s3 = sigmas(z)
		sd, sd1, sd2, sd3 = sigmas(zd)
		sx, sx1, sx2 = sigmas(zx, False)
		st, st1, st2 = sigmas(zt, False)
		sxe, sxe1, sxe2 = sigmas(z1, False)

		f_ = np.sum(vk*sd1-vw2*sd2)-f*(System.nt-1)
		g1 = np.sum(self.v*sx)
		g2 = np.sum(self.v*st)
		g3 = np.sum(self.v*sxe)

		E = (f_*f_)/((2*(System.nx-2)*(System.nt-1))) + System.tau*(g1*g1)/(2*(System.nx-2)) + System.tau*((g2*g2)+(g3*g3))/((2*System.nt))
		
		

		dvf = (f_/((System.nx-2)*(System.nt-1)))*np.sum(np.sum((self.k*sd1-w2*sd2), axis=0), axis=0)
		dvg = (g1/((System.nx-2)))*np.sum(sx, axis=0)+(np.sum(g2*st+g3*sxe, axis=0)/System.nt)

		dwf = f_*np.sum(np.sum(vk*mul_x(sd2,x[1:-1])-2*vw*sd2-vw2*mul_x(sd3,x[1:-1]), axis =0), axis=0)/(((System.nx-2)*(System.nt-1)))
		dwg = (g1/((System.nx-2)))*np.sum(self.v*mul_t(sx1,x[1:-1]), axis=0)+(g3/(System.nt))*System.tau*np.sum(self.v*sxe1, axis=0)

		dkf = (f_/(((System.nx-2)*(System.nt-1))))*np.sum(np.sum(self.v*sd1+vk*mul_t(sd2,t[1:])-vw2*mul_t(sd3,t[1:]), axis=1), axis=0)
		dkg = (g2*np.sum(self.v*mul_t(st1,t), axis=0) + g3*np.sum(self.v*mul_t(sxe1,t), axis=0))/(System.nt)

		dbf = (f_/(((System.nx-2)*(System.nt-1))))*np.sum(vk*sd2-vw2*sd3)
		dbg = (System.tau*g1*np.sum(self.v*sx1, axis=0)/((System.nx-2))) + (System.tau*np.sum(g2*self.v*st1+g3*self.v*sxe1, axis=0)/(System.nt))

		dv = dvf+System.tau*dvg
		dw = dwf+System.tau*dwg
		dk = dkf+System.tau*dkg
		db = dvf+System.tau*dvg

		self.v -= System.lr*dv
		self.w -= System.lr*dw
		self.k -= System.lr*dk
		self.b -= System.lr*db
		if test:
			unn = np.sum(self.v*s, axis=2)
			e = np.sum((u_act-unn)*(u_act-unn))/((System.nx*System.nt))
			return E, e
		return E
		

