import numpy as np

def mul_x(T, v):
	return np.transpose(np.transpose(T,[0,2,1])*v,[0,2,1])

def mul_t(T, v):
	return np.transpose(np.transpose(T)*v)

def sum_t(T, v):
	return np.transpose(np.transpose(T)+v)

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
	lr= .0001

class Net():
	def __init__(self, fanout):
		self.fanout = fanout
		self.v = np.random.rand(self.fanout)/np.sqrt(self.fanout)
		self.w = np.random.rand(self.fanout)/np.sqrt(self.fanout)
		self.k = np.random.rand(self.fanout)/np.sqrt(self.fanout)
		self.l = np.random.rand(self.fanout)/np.sqrt(self.fanout)
		self.b = np.random.rand(self.fanout)
		self.a = 1

	def run(self, x, t, z_ = None):
		u_ = np.zeros(x.shape[0])
		u = [u_.copy()]
		a2 = self.a*self.a
		for i, t_ in enumerate(t[1:]):
			if not z_ is None:
				z = z_[i+1]
			else:
				z = np.array([[w_*x_ + k_*t_ + l_*u__ + b_ for w_, k_, l_, b_ in zip(self.w, self.k, self.l, self.b)] for x_, u__ in zip(x, u_)])

			s = sigma(z)
			u.append((1-np.exp(-a2*t_))*np.sum(mul_t(self.v*s,x*(x-1)), axis=1))
			u_ = u[-1].copy()
		return np.array(u)

	def train(self, x, t, f, u_act=None):
		xl = x.shape[0]-2
		u_ = np.zeros((4,xl)) # u_, ut_, ux_, uxx_
		uv_ = np.zeros((4, xl, self.fanout)) # du_dv, d2u_dvdt, d2u_dvdx, d3u_dvdx2 
		uw_ = np.zeros((4, xl, self.fanout)) 
		uk_ = np.zeros((4, xl, self.fanout)) 
		ul_ = np.zeros((4, xl, self.fanout)) 
		ub_ = np.zeros((4, xl, self.fanout)) 
		ua_ = np.zeros((4, xl))  

		dv = np.zeros(self.fanout)
		dw = np.zeros(self.fanout)
		dk = np.zeros(self.fanout)
		dl = np.zeros(self.fanout)
		db = np.zeros(self.fanout)
		da = 0
		a2 = self.a*self.a
		E = 0
		xd = x[1:-1]
		scale = xl*(t.shape[0]-1)
		for t_index, t_ in enumerate(t[1:]):
			a = (1-np.exp(-a2*t_))
			a1 = a2*(1-a)
			b = xd*(xd-1)
			b1 = 2*xd - 1
			ab = a*b
			a1b = a1*b
			ab1 = a*b1
			z = np.array([[w_*x_ + k_*t_ + l_*u__ + b_ for w_, k_, l_, b_ in zip(self.w, self.k, self.l, self.b)] for x_, u__ in zip(x, u_[0])])
			zd = np.array([[w_*x_ + k_*t_ + l_*u__ + b_ for w_, k_, l_, b_ in zip(self.w, self.k, self.l, self.b)] for x_, u__ in zip(xd, u_[0])])


			s, s1, s2, s3 = sigmas(z)
			sd, sd1, sd2, sd3 = sigmas(zd)
			
			vs = self.v*sd
			vs1 = self.v*sd1
			vs2 = self.v*sd2
			vs3 = self.v*sd3

			u=[ab*np.sum(vs, axis = 1)]
			u.append(a1b*np.sum(vs, axis=1) + ab*np.sum(vs1*(self.k + np.outer(u_[1],self.l)), axis=1) )
			u.append(ab1*np.sum(vs, axis=1) + ab*np.sum(vs1*(self.w + np.outer(u_[2],self.l)), axis=1) )
			u.append(2*a*np.sum(vs, axis=1) + 2*ab1*np.sum(vs1*(self.w + np.outer(u_[1],self.l)), axis=1) + ab*np.sum(np.outer(u_[2],self.l)*vs1,axis=1) + ab*np.sum((np.outer(u_[3], self.l) + self.w)*vs2, axis=1))

			f_ = u[1] - u[3] - f[1:-1]
			E+=np.sum(f_*f_/scale)
			
			uv = [mul_t(sd+vs1*uv_[0]*self.l, ab)]
			uv.append(mul_t(sd, a1b) + mul_t(vs1*uv_[0]*self.l, a1b) + mul_t(self.k + np.outer(u_[1],self.l), ab)*sd1 + vs1*mul_t(uv_[1]*self.l, ab) + vs2*mul_t(self.l*(self.k+np.outer(u_[1], self.l))*uv_[0], ab))
			uv.append(mul_t(sd, ab1) + mul_t(vs1*uv_[0]*self.l, ab1) + mul_t(self.w + np.outer(u_[2],self.l), ab)*sd1 + vs1*mul_t(uv_[2]*self.l, ab) + vs2*mul_t(self.l*(self.w+np.outer(u_[2], self.l))*uv_[0], ab))
			d3u_dvdx2 = 2*s + 2*self.l*uv_[0]*vs1 + 2*mul_t(self.w+np.outer(u_[2], self.l), b1)*sd1 + mul_t(2*self.l*uv_[2]*vs1, b1) + mul_t(2*self.l*(self.w + np.outer(u_[2], self.l)), b1)*uv_[0]*vs2 + np.outer(b*u_[3], self.l)*sd1
			d3u_dvdx2+= mul_t(uv_[3], b)*self.l*vs1 + np.outer(b*u_[3], self.l*self.l)*uv_[0]*vs2 + mul_t((self.w + np.outer(u_[2], self.l))**2, b)*sd2 + 2*self.l*mul_t(self.w+np.outer(u_[2], self.l),b)*uv_[0]*vs2 + vs3*self.l*mul_t((self.w + np.outer(u_[2], self.l))**2, b)*uv_[0]
			uv.append(d3u_dvdx2*a)

			uw = [vs1*mul_t(sum_t(self.l*uw_[0], xd), ab)]
			uw.append(vs1*mul_t(sum_t(self.l*uw_[0], xd),a1b) + vs1*mul_t(self.l*uw_[1],ab) + vs2*(self.k+np.outer(u_[1], self.l))*mul_t(sum_t(uw_[0]*self.l,xd),ab))
			uw.append(vs1*mul_t(sum_t(self.l*uw_[0], xd),ab1) + vs1*mul_t(1+self.l*uw_[2],ab) + vs2*(self.k+np.outer(u_[1], self.l))*mul_t(sum_t(uw_[0]*self.l,xd),ab))
			d3u_dwdx2 = 2*vs1*sum_t(self.l*uw_[0], xd) + 2*vs1*mul_t(1+self.l*uw_[2], b1) + 2*vs2*sum_t(self.l*uw_[0], xd)*mul_t(self.w + np.outer(u_[2],self.l), b1) + vs1*self.l*mul_t(uw_[3],b) + vs2*np.outer(b*u_[3],self.l)*sum_t(self.l*uw_[0],xd)
			d3u_dwdx2+= 2*vs2*(self.w + np.outer(u_[2], self.l))*mul_t(1+self.l*uw_[2],b) + vs3*mul_t(sum_t(self.l*uw_[0],xd),b)*((self.w + np.outer(u_[2], self.l))**2)
			uw.append(d3u_dwdx2*a)

			uk = [vs1*mul_t(self.l*uk_[0]+ t_, ab)]
			uk.append(vs1*mul_t(t_+uk_[0]*self.l,a1b) + vs1*mul_t(1+self.l*uk_[1], ab) + vs2*mul_t(self.k + np.outer(u_[1], self.l), ab)*(t_+self.l*uk_[0]))
			uk.append(vs1*mul_t(t_+uk_[0]*self.l,ab1) + vs1*mul_t(self.l*uk_[2], ab) + vs2*mul_t(self.w + np.outer(u_[2], self.l), ab)*(t_+self.l*uk_[0]))
			d3u_dkdx2 = 2*vs1*(t_ + uk_[0]*self.l) + 2*vs1*self.l*mul_t(uk_[2], b1) + 2*vs2*mul_t(self.w+np.outer(u_[2], self.l), b1)*(t_ + self.l*uk_[0]) + vs1*self.l*mul_t(uk_[3], b) + vs2*self.l*mul_t(t_ + self.l*uk_[0], b*u_[3])
			d3u_dkdx2+= 2*vs2*self.l*mul_t(self.w + np.outer(u_[2], self.l), b)*uk_[2] + vs3*mul_t((self.w+np.outer(u_[2], self.l))**2, b)*(t_ + self.l*uk_[0])
			uk.append(d3u_dkdx2*a)

			ul = [vs1*mul_t(sum_t(self.l*ul_[0], u_[0]), ab)]
			ul.append(vs1*mul_t(sum_t(self.l*ul_[0], u_[0]), a1b) + vs1*mul_t(sum_t(self.l*ul_[1],u_[1]), ab) + vs2*mul_t(self.k + np.outer(u_[1], self.l), ab)*sum_t(self.l*ul_[0], u_[0]))
			ul.append(vs1*mul_t(sum_t(self.l*ul_[0], u_[0]), ab1) + vs1*mul_t(sum_t(self.l*ul_[2],u_[2]), ab) + vs2*mul_t(self.w + np.outer(u_[2], self.l), ab)*sum_t(self.l*ul_[0], u_[0]))
			d3u_dldx2 = 2*vs1*sum_t(self.l*ul_[0], u_[0]) + 2*vs1*mul_t(sum_t(self.l*ul_[2], u_[2]), b1) + 2*vs2*mul_t(sum_t(self.l*ul_[0], u_[0]), b1)*(self.w+np.outer(u_[2], self.l)) + vs1*self.l*mul_t(sum_t(self.l*ul_[3], u_[3]), b) + vs2*self.l*mul_t(sum_t(self.l*ul_[0], u_[0]), b*u_[3])
			d3u_dldx2+= 2*vs2*mul_t(sum_t(self.l*ul_[2], u_[2]),b)*(self.w+np.outer(u_[2], self.l)) + vs3*mul_t((self.w+np.outer(u_[2], self.l))**2, b)*sum_t(self.l*ul_[0], u_[0])
			ul.append(d3u_dldx2*a)

			ub = [vs1*mul_t(self.l*ub_[0]+ 1, ab)]
			ub.append(vs1*mul_t(1+self.l*ub_[0], a1b) + vs1*self.l*mul_t(ub_[1],ab) + vs2*mul_t(self.k + np.outer(u_[1],self.l), ab)*(1 + self.l*ub_[0]))
			ub.append(vs1*mul_t(1+self.l*ub_[0], ab1) + vs1*self.l*mul_t(ub_[2],ab) + vs2*mul_t(self.w + np.outer(u_[2],self.l), ab)*(1 + self.l*ub_[0]))
			d3u_dbdx2 = 2*vs1*(1 + ub_[0]*self.l) + 2*vs1*self.l*mul_t(ub_[2], b1) + 2*vs2*mul_t(self.w + np.outer(u_[2], self.l), b1)*(1 + self.l*ub_[0]) + vs1*self.l*mul_t(ub_[3], b) + vs2*self.l*mul_t(1 + self.l*ub_[0], b*u_[3])
			d3u_dbdx2+= 2*vs2*self.l*mul_t(self.w + np.outer(u_[2], self.l), b)*ub_[2] + vs3*mul_t((self.w+np.outer(u_[2], self.l))**2, b)*(1 + self.l*ub_[0])
			ub.append(d3u_dbdx2*a)

			ua = [np.sum(2*self.a*t_*(1-a)*mul_t(vs, b) + mul_t(vs1*self.l, ab*ua_[0]), axis = 1)]
			ua.append(np.sum(2*self.a*(1-a)*(1-a2*t_)*mul_t(vs, b) + mul_t(vs1, a1b*ua_[0])*self.l + 2*vs1*self.a*t_*(1-a)*mul_t(self.k + np.outer(u_[1], self.l), b) + self.l*mul_t(vs1, ua_[1]*ab) + vs2*self.l*mul_t(self.k + np.outer(u_[1], self.l), ab*ua_[0]), axis = 1))
			ua.append(np.sum(2*self.a*t_*(1-a)*(1-a2*t_)*mul_t(vs, b1) + mul_t(vs1, ab1*ua_[0])*self.l + 2*vs1*self.a*t_*(1-a)*mul_t(self.w+ np.outer(u_[2], self.l), b) + self.l*mul_t(vs1, ua_[2]*ab) + vs2*self.l*mul_t(self.w + np.outer(u_[2], self.l), ab*ua_[0]), axis = 1))
			d3u_dadx2 = 4*self.a*t_*(1-a)*vs + 2*a*mul_t(vs1,ua_[0])*self.l + 4*self.a*t_*(1-a)*vs1*mul_t(self.w + np.outer(u_[2], self.l), b1) + 2*self.l*mul_t(vs1, ua_[2]*ab1) + 2*vs2*mul_t(self.w+ np.outer(u_[2], self.l), ab1*ua_[0]) + 2*self.a*t_*(1-a)*self.l*mul_t(vs1,u_[3]*b) + self.l*mul_t(vs1, ua_[3]*ab)
			d3u_dadx2+= self.l*self.l*mul_t(vs2, ua_[0]*u_[3]*ab) + 2*self.a*t_*(1-a)*vs2*mul_t((self.w + np.outer(u_[2], self.l))**2, ab) + 2*vs2*self.l*mul_t(self.w+ np.outer(u_[2], self.l), ab*ua_[2]) + vs3*self.l*mul_t((self.w+ np.outer(u_[2], self.l))**2, ab*ua_[0])
			ua.append(np.sum(d3u_dadx2, axis = 1))

			delta = lambda x: np.sum(f_*(x[1] - x[3]).transpose(), axis = 1)
			dv+= delta(uv)
			dw+= delta(uw)
			dk+= delta(uk)
			dl+= delta(ul)
			db+= delta(ub)
			da+= np.sum(f_*(ua[1] - ua[3]))


		scale/=System.lr
		self.v -= dv/scale
		self.w -= dw/scale
		self.k -= dk/scale
		self.l -= dl/scale
		self.b -= db/scale
		self.a -= da/scale 
		if not u_act is None:
			unn = self.run(x, t)
			e = np.sum((u_act-unn)*(u_act-unn))/(System.nx*System.nt)
			return E, e
		return E
		

