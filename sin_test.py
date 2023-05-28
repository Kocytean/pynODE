from net import *

System.N_x = 10
System.N_t = 10
fanout = 10

num_iterations = 500000
min_sum_error = 100000
total_error_trend = []
test = Net(fanout)
x = np.linspace(0,1, System.N_x)
t = np.linspace(0,1, System.N_t)
f = np.array([np.pi*np.pi*np.sin(np.pi*x) for _ in range(System.N_t)])
u0 = np.zeros(System.N_x)
u = np.outer(1-np.exp(-np.pi*np.pi*t),np.sin(np.pi*x))
for i in range(num_iterations):
	errors = test.train(x, t, f,20)
	total_error_trend.append(sum(errors))
	if i%100 == 99:
		print('Iteration ' +str(i+1))
		result = test.test(x,t,u)
		# print(sum(result.transpose()))
		sum_error = sum(sum(result))
		print(sum_error)
		min_sum_error = min(min_sum_error, sum_error)
		print([sum_error, min_sum_error])
		print(sum(errors))
final_error = test.test(x,t,u)
print(final_error)
print([total_error_trend[999 + 1000*i ] for i in range(500)])