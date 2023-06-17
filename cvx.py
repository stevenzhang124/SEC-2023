import cvxpy as cp
import numpy as np

# # Problem data.
# m = 30
# n = 20
# np.random.seed(1)
# A = np.random.randn(m, n)
# b = np.random.randn(m)

# # Construct the problem.
# x = cp.Variable(n)
# objective = cp.Minimize(cp.sum_squares(A @ x - b))
# constraints = [0 <= x, x <= 1]
# prob = cp.Problem(objective, constraints)

# # The optimal objective value is returned by `prob.solve()`.
# result = prob.solve()
# # The optimal value for x is stored in `x.value`.
# print("status:", prob.status)
# print("optimal value", prob.value)
# print("optimal var", x.value)

# np.random.seed(1)
# m = 3
# B = 1
# a = np.random.randn(m, 1)
# A = np.eye(m)
# print(A)
# C = np.random.randn(m, 1)

# x = cp.Variable((m, 1))
# T = cp.Variable((1, 1))
# objective = cp.Minimize(T)
# constraints = [0 <= x, cp.sum(x) <= B, A@x-x@T <= C]
# prob = cp.Problem(objective, constraints)
# # The optimal objective value is returned by `prob.solve()`.
# result = prob.solve()
# # The optimal value for x is stored in `x.value`.
# print("status:", prob.status)
# print("optimal value", prob.value)
# print("optimal var", x.value)

# -----------------------------------------------------------------
# A = [10, 20, 30]
# C = [5000, 7000, 10000]
# B = 1000
# x1 = cp.Variable(integer=True)
# x2 = cp.Variable(integer=True)
# x3 = cp.Variable(integer=True)

# # cost = cp.maximum(A[0]+C[0]/x1, A[1]+C[1]/x2, A[2]+C[2]/x3)
# # objective = cp.Minimize(cost)
# constraints = [x1+x2+x3 <= B, 0 <= x1, x1<= B, 0<=x2, x2<=B, 0<=x3, x3<=B]
# prob = cp.Problem(cp.Minimize(cp.maximum(A[0]+C[0]/x1, A[1]+C[1]/x2, A[2]+C[2]/x3)), constraints)
# print(prob)
# result = prob.solve(qcp=True, solver=cp.MOSEK, low=40, high=50,verbose=True)
# # The optimal value for x is stored in `x.value`.
# print("status:", prob.status)
# print("optimal value", prob.value)
# print("optimal var", x1.value)
# print("optimal var", x2.value)
# print("optimal var", x3.value)


# the formulation should be OK, problem is about the coding.
# A = [5.5036, 0.28246, 0.21598, 0.119655]
# C = [177704, 177704, 177704, 177704]
# B = 655360
# x1 = cp.Variable(integer=True)
# x2 = cp.Variable(integer=True)
# x3 = cp.Variable(integer=True)
# x4 = cp.Variable(integer=True)

# # cost = cp.maximum(A[0]+C[0]/x1, A[1]+C[1]/x2, A[2]+C[2]/x3)
# # objective = cp.Minimize(cost)
# constraints = [x1+x2+x3+x4 <= B, 0 <= x1, x1<= B, 0<=x2, x2<=B, 0<=x3, x3<=B, 0<=x4, x4<=B]
# prob = cp.Problem(cp.Minimize(cp.maximum(A[0]+C[0]/x1, A[1]+C[1]/x2, A[2]+C[2]/x3, A[3]+C[3]/x4)), constraints)
# print(prob)
# result = prob.solve(qcp=True, solver=cp.MOSEK, low=5.7747, high=5.8239,verbose=True)
# # The optimal value for x is stored in `x.value`.
# print("status:", prob.status)
# print("optimal value", prob.value)
# print("optimal var", x1.value)
# print("optimal var", x2.value)
# print("optimal var", x3.value)
# print("optimal var", x4.value)


def convex(A, C, B):
	print(A)
	print(C)
	print(B)
	# X = cp.Variable((len(A), 1), integer=True)
	# print(X.shape())
	# expr = np.array(A).reshape(-1,1) + np.array(C).reshape(-1,1) / X

	X = []
	for i in range(len(A)):
		X.append(cp.Variable()) #integer=True
	# constraints = [cp.sum(X) <= B, 0 <= X]
	constraints = [cp.sum(X) <= B]
	for i in range(len(X)):
		constraints.append(0 <= X[i])

	expr = ()
	low_constraints = []
	for i in range(len(A)):
		expr += (A[i]+C[i]/X[i],)
		low_constraints.append(A[i]+C[i]/B)
	
	print(expr)
	cost = cp.maximum(*expr)

	prob = cp.Problem(cp.Minimize(cost), constraints)

	low_bound = max(low_constraints)
	print(low_bound)
	# calculate high bound
	straggler = max(A)
	B_tmp = []
	for i in range(len(A)):
		if A[i] != straggler:
			B_tmp.append(C[i]/(straggler-A[i]))
		else:
			straggler_index = i

	print(B_tmp)
	high_bound = straggler + C[straggler_index]/(B-sum(B_tmp)) + 80
	print(high_bound)
	print(prob)
	result = prob.solve(qcp=True, solver=cp.MOSEK, low=low_bound, high=high_bound*2,verbose=True)
	# The optimal value for x is stored in `x.value`.
	print("status:", prob.status)
	print("optimal value", prob.value)
	for x in X:
		print("optimal var", x.value)
	# print("optimal var", x1.value)
	# print("optimal var", x2.value)
	# print("optimal var", x3.value)

	train_time = prob.value
	
	
	return train_time