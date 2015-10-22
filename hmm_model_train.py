'''
'''

def forward_alg(matrix_a, matrix_b, obs):
	"""
	DOESNT WORK
	"""
	# Create data structures for going back and forth between tokens in string
	# form and integer form.
	num_to_token = []
	num_to_token.append('<START>')
	for token in matrix_a:
		# print(token)
		if token not in ['<START>', '<END>', 0, '<SEEN>', '<NOT_SEEN_P>']:
			# print(token)
			num_to_token.append(token)
	num_to_token.append('<END>')

	# init
	alpha = init_forward(matrix_a, matrix_b, obs, num_to_token)

	# recursion
	alpha = recursion_forward(matrix_a, matrix_b, obs, num_to_token, alpha)

	# termination
	result = 0
	for i in range(1, N + 1):
		transition = matrix_a[num_to_token[i]]['<END>']
		forward = alpha[T - 1][i]

		result += math.log(transition) + forward


	return result

def recursion_forward(matrix_a, matrix_b, obs, num_to_token, alpha):
	"""
	DOESNT WORK
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	for t in range(1, T):
		for j in range(1, N + 1):
			sigma = 0
			for i in range(1, N + 1):
				transition = matrix_a[num_to_token[i]][num_to_token[j]]
				emission = matrix_b[num_to_token[j]][obs[t]]
				forward = alpha[t - 1][i]

				sigma = sigma + math.log(transition) \
							  + math.log(emission) \
							  + forward

			alpha[t][j] = sigma


	return alpha

def init_forward(matrix_a, matrix_b, obs, num_to_token):
	"""
	DOESNT WORK
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	# T rows, N columns
	alpha = [[1 for i in range(N + 2)] for j in range(T)]

	for j in range(1, N + 1):
		transition = matrix_a['<START>'][num_to_token[j]]
		emission = matrix_b[num_to_token[j]][obs[0]]

		alpha[1][j] = math.log(transition) + math.log(emission)

	return alpha

def backward_alg(matrix_a, matrix_b, obs):
	"""
	DOESNT WORK
	"""
	# Create data structures for going back and forth between tokens in string
	# form and integer form.
	num_to_token = []
	num_to_token.append('<START>')
	for token in matrix_a:
		# print(token)
		if token not in ['<START>', '<END>', 0, '<SEEN>', '<NOT_SEEN_P>']:
			# print(token)
			num_to_token.append(token)
	num_to_token.append('<END>')

	# init
	beta = init_backward(matrix_a, matrix_b, obs, num_to_token)

	# recursion
	beta = recursion_backward(matrix_a, matrix_b, obs, num_to_token, beta)

	# termination
	result = 0
	for j in range(1, N + 1):
		transition = matrix_a['<START>'][num_to_token[j]]
		emission = matrix_b[num_to_token[j]][obs[0]]
		backward = beta[0][j]

		result += math.log(transition) + \
				  math.log(emission) + \
				  backward

	return result

def recursion_backward(matrix_a, matrix_b, obs, num_to_token, beta):
	"""
	DOESNT WORK
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	for t in range(T - 1, -1, -1): # not sure about this
		for i in range(1, N + 1):
			sigma = 0
			for j in range(1, N + 1):
				transition = matrix_a[num_to_token[i]][num_to_token[j]]
				emission = matrix_b[num_to_token[j]][obs[t + 1]]
				backward = beta[t + 1][j]

				sigma = sigma + math.log(transition) + \
						 		math.log(emission) + \
						 		backward

			beta[t][i] = sigma

	return beta

def init_backward(matrix_a, matrix_b, obs, num_to_token):
	"""
	DOESNT WORK
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	beta = [[1 for i in range(N + 2)] for j in range(T)]

	for j in range(1, N + 1):
		beta[T - 1][j] = math.log(matrix_a[num_to_token[i]]['<END>'])

	return beta

def forward-backward():
	"""
	DOESNT WORK
	"""
	print('Training HMM matrices A and B...')

	# initialize A, B
	matrix_a = {}
	matrix_b = {}
	Xx

	# iterate until convergence
		# E-step
		Xx

		# M-step
		Xx

	model = {}
	model['a'] = matrix_a
	model['b'] = matrix_b

	pickle.dump(model, open('trainmodel.dat', 'wb'))
	print('Saving to trainmodel.dat')

	return A, B

def main():
	print()
	# forward-backward()


if __name__ == '__main__':
	main()