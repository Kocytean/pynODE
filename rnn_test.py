from rnn import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

System.nx = 15
System.nt = 10
fanout = 50
num_iterations = 100000
min_sum_error = 100000
total_error_trend = []
test = Net(fanout)
x = np.linspace(0, 1, System.nx)
t = np.linspace(0, 1, System.nt)
u_act = np.array([[(1-np.exp(-np.pi*t_))*np.sin(np.pi*x_) for x_ in x] for t_ in t])
f =(np.pi*np.pi)*np.sin(np.pi*x)
for i in range(num_iterations):
	
	if i%100 == 99:
		E, e = test.train(x, t, f, u_act)
		total_error_trend.append(E)
		min_sum_error = min(min_sum_error, e)
		print('Iter ' +str(i+1)+':')
		print([E, e, min_sum_error, test.a])
	else:
		test.train(x, t, f, u_act)

print(e)
print(total_error_trend)
fig = plt.figure(figsize = (12,6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
Z = test.run(x,t)
X, T = np.meshgrid(x, t)
surf1 = ax1.plot_surface(X, T, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
surf2 = ax2.plot_surface(X, T, u_act, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()