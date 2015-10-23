'''
'''

import pickle
import sys

def printError():
	"""
	WORKS
	"""
	print('Error.')
	print('Usage: $ python3 hmm_model_train.py <input file>')
	sys.exit()

def forwardAlg(matrix_a, matrix_b, obs, num_to_token):
	"""
	SHOULD WORK
	"""
	# init
	alpha = initForward(matrix_a, matrix_b, obs, num_to_token)

	# recursion
	alpha = recursionForward(matrix_a, matrix_b, obs, num_to_token, alpha)

	# termination
	result = 0
	for i in range(1, N + 1):
		transition = matrix_a[num_to_token[i]]['<END>']
		forward = alpha[T - 1][i]

		result += math.log(transition) + forward


	return result, alpha

def recursionForward(matrix_a, matrix_b, obs, num_to_token, alpha):
	"""
	SHOULD WORK
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

def initForward(matrix_a, matrix_b, obs, num_to_token):
	"""
	SHOULD WORK
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	# T rows, N columns
	alpha = [[1 for i in range(N + 2)] for j in range(T)]

	for j in range(1, N + 1):
		transition = matrix_a['<START>'][num_to_token[j]]
		emission = matrix_b[num_to_token[j]][obs[0]]

		alpha[0][j] = math.log(transition) + math.log(emission)

	return alpha

def backwardAlg(matrix_a, matrix_b, obs, num_to_token):
	"""
	SHOULD WORK
	"""
	# init
	beta = initBackward(matrix_a, matrix_b, obs, num_to_token)

	# recursion
	beta = recursionBackward(matrix_a, matrix_b, obs, num_to_token, beta)

	# termination
	result = 0
	for j in range(1, N + 1):
		transition = matrix_a['<START>'][num_to_token[j]]
		emission = matrix_b[num_to_token[j]][obs[0]]
		backward = beta[0][j]

		result += math.log(transition) + \
				  math.log(emission) + \
				  backward

	return result, beta

def recursionBackward(matrix_a, matrix_b, obs, num_to_token, beta):
	"""
	SHOULD WORK
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	for t in range(T - 2, -1, -1): # not sure about this
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

def initBackward(matrix_a, matrix_b, obs, num_to_token):
	"""
	SHOULD WORK
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	beta = [[1 for i in range(N + 2)] for j in range(T)]

	for j in range(1, N + 1):
		beta[T - 1][j] = math.log(matrix_a[num_to_token[j]]['<END>'])

	return beta

def forwardBackward(states, sent_list, Xx):
	"""
	DOESNT WORK
	"""
	# initialize A, B
	matrix_a = {}
	matrix_b = {}

	words_dict = {}
	for line in sent_list:
		for word in line.split(' '):
			word = word.split('/')[0]

			if word not in words_dict:
				words_dict[word] = 1
			else:
				words_dict[word] += 1

	for i in states:
		matrix_a[i] = {}
		matrix_b[i] = {}
		for j in states:
			matrix_a[i][j] = 1 / len(states) # what prob to enter here? XX

		for line in sent_list:
			for word in line.split(' '):
				word = word.split('/')[0]

				if word not in matrix_b[i]:
					matrix_b[i][word] = 1 / len(words_dict) # what prob to enter here? XX

	# iterate until convergence
	for repeat in range(1000): # change this to check convergence later on XX
		sentence = sent_list[0] # XXXXXXXXXXX

		N = len(states) - 2
		T = len(sentence)

		alpha, final_alpha = forwardAlg(matrix_a, matrix_b, sentence, states) # XX
		beta, final_beta = backwardAlg(matrix_a, matrix_b, sentence, states) # XX

		# E-step
		gamma = {}
		for t in range(0, T):
			gamma[t] = {}
			for j in range(0, N + 1):
				gamma[t][j] = (alpha[t][j] * beta[t][j]) / final_alpha

		xi = {}
		for t in range(0, T - 1): # should this be 'T - 1'? XX
			xi[t] = {}
			for i in range(0, N + 1):
				xi[t][i] = {}
				for j in range(0, N + 1):
					xi[t][i][j] = alpha[t][i] + \
								  math.log(matrix_a[states[i]][states[j]]) + \
								  math.log(matrix_b[states[j]][sentence[t + 1]]) + \
								  beta[t + 1][j] # what do we do here when T for beta and alpha? XX

		# Setup a_hat and b_hat
		a_hat = {}
		b_hat = {}
		for state in states:
			a_hat[state] = {}
			b_hat[state] = {}

			for s in states:
				a_hat[state][s] = False

			for sent in sent_list:
				for word in sent.split(' '):
					b_hat[state][word] = False

		# M-step
		for i in range(0, N + 1):
			for j in range(0, N + 1):
				a_num_sum = 0
				for t in range(0, T - 1):
					a_num_sum += xi[t][i][j]

				a_denom_sum = 0
				for t in range(0, T - 1):
					for k in range(0, N + 1):
						a_denom_sum += xi[t][i][k]

				a_hat[states[i]][states[j]] = a_num_sum / a_denom_sum

		for j in range(0, N + 1): # XX
			for v_k in b_hat[states[j]]: # XX
				for w in sentence: # XX
					b_num_sum = 0
					for t in range(0, T):
						if w == v_k: # such that XX
							b_num_sum += gamma[t][j]

					b_denom_sum = 0
					for t in range(0, T):
						b_denom_sum += gamma[t][j]

					b_hat[states[j]][v_k] = b_num_sum / b_denom_sum

		Xx

	return matrix_a, matrix_b

def hmmTrainer(f):
	"""
	DOESNT WORK
	"""
	print('Training HMM matrices A and B...')

	states = ['<START>', 'DET', '.', 'ADJ', 'PRT', 'VERB', 'NUM', 'X', \
			  'CONJ', 'PRON', 'ADV', 'ADP', 'NOUN', '<END>']

	sent_list = []
	for line in f:
		sent_list.append(line.lower()) # lower case? XX

	matrix_a, matrix_b = forwardBackward(states, sent_list, Xx)

	model = {}
	model['a'] = matrix_a
	model['b'] = matrix_b

	pickle.dump(model, open('trainmodel.dat', 'wb'))
	print('Saving to trainmodel.dat')

def main():
	if len(sys.argv) != 2:
		printError()

	try:
		f = open(sys.argv[1])
	except:
		printError()

	hmmTrainer(f)
	# forward-backward()


if __name__ == '__main__':
	main()