'''
hmm_model_train.py
Frederik Roenn Stensaeth
10.23.15

A Python program that creates a model from the pos-tagged training data. 
It takes as a command-line argument a file to use for training, and output 
the model to a file named trainmodel.dat.
'''

import pickle
import sys
import math
import os.path
import re
import hmm_model_build

def printError():
	"""
	printError() prints out a generic error message and a usage statement.

	@params: n/a.
	@return: n/a.
	"""
	print('Error.')
	print('Usage: $ python3 hmm_model_train.py <input file>')
	sys.exit()

def forwardAlg(matrix_a, matrix_b, obs, num_to_token):
	"""
	forwardAlg() takes a pair of A and B matrices, a list of observations and
	a list containing all states. This information is used to find the
	forward probabilities of the observation.

	@params: A and B matrices,
			 list of observations,
			 list of states (with <START> and <END> at index 0 and -1 
			 				respectively).
	@return: forward probability of the entire observation sequence,
			 alpha matrix.
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	# Init
	alpha = initForward(matrix_a, matrix_b, obs, num_to_token)

	# Recursion
	alpha = recursionForward(matrix_a, matrix_b, obs, num_to_token, alpha)

	# Termination
	# Loops over all the different states to find the forward probability of
	# the entire observation sequence.
	result = 0
	for i in range(1, N + 1):
		transition = matrix_a[num_to_token[i]]['<END>']
		forward = alpha[T - 1][i]

		result = result + math.exp(math.log(transition) + forward)
	
	return math.log(result), alpha

def recursionForward(matrix_a, matrix_b, obs, num_to_token, alpha):
	"""
	recursionForward() takes a pair of A and B matrices, a list of
	observations, a list of states and an alpha matrix, and finds the
	remaining forward probabilities. The completed alpha matrix is returned.

	@params: A and B matrices,
			 list of observations,
			 list of states (with <START> and <END> at index 0 and -1 
			 				respectively),
			 alpha matrix.
	@return: completed alpha matrix.
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	# Loops over all the times and state/state combinations in order to
	# calculate the forward probabilities.
	for t in range(1, T):
		for j in range(1, N + 1):
			sigma = 0 #0.00000000001
			for i in range(1, N + 1):
				# Gets the transition, emission and forward probability (t - 1)
				# which we can use to find the forward probability at t.
				transition = matrix_a[num_to_token[i]][num_to_token[j]]
				emission = matrix_b[num_to_token[j]][obs[t].split('/')[0].lower()]
				forward = alpha[t - 1][i]

				new_unit = math.log(transition) + \
						   math.log(emission) + \
						   forward
				sigma = sigma + math.exp(new_unit)

			# If the forward probability is 0, set it to be something really
			# small.
			if sigma == 0:
				alpha[t][j] = math.log(0.00000000001)
			else:
				alpha[t][j] = math.log(sigma)

	return alpha

def initForward(matrix_a, matrix_b, obs, num_to_token):
	"""
	initForward() takes a pair of A and B matrices, a list of observations
	and a list of states, and finds the initial forward
	probabilities (<START> --> state). The alpha matrix is returned.

	@params: A and B matrices,
			 list of observations,
			 list of states (with <START> and <END> at index 0 and -1 
			 				respectively).
	@return: alpha matrix.
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	# T rows, N columns
	alpha = [[1 for i in range(N + 2)] for j in range(T)]
	# Loops over the states and finds the probabilities of going from <START>
	# to that state for that perticular word.
	for j in range(1, N + 1):
		transition = matrix_a['<START>'][num_to_token[j]]

		emission = matrix_b[num_to_token[j]][obs[0].split('/')[0].lower()]

		alpha[0][j] = math.log(transition) + math.log(emission)

	return alpha

def backwardAlg(matrix_a, matrix_b, obs, num_to_token):
	"""
	backwardAlg() takes a pair of A and B matrices, a list of observations and
	a list containing all states. This information is used to find the
	backward probabilities of the observation. 

	@params: A and B matrices,
			 list of observations,
			 list of states (with <START> and <END> at index 0 and -1 
			 				respectively),
			 beta matrix.
	@return: completed beta matrix.
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	# init
	beta = initBackward(matrix_a, matrix_b, obs, num_to_token)

	# recursion
	beta = recursionBackward(matrix_a, matrix_b, obs, num_to_token, beta)

	# termination
	result = 0#0.00000000001
	for j in range(1, N + 1):
		transition = matrix_a['<START>'][num_to_token[j]]
		emission = matrix_b[num_to_token[j]][obs[0].split('/')[0].lower()]
		backward = beta[0][j]

		new_unit = math.log(transition) + \
				   math.log(emission) + \
				   backward
		result = result + math.exp(new_unit)

	return math.log(result), beta

def recursionBackward(matrix_a, matrix_b, obs, num_to_token, beta):
	"""
	recursionBackward() takes a pair of A and B matrices, a list of
	observations, a list of states and a beta matrix, and finds the
	remaining backward probabilities. The completed beta matrix is returned.

	@params: A and B matrices,
			 list of observations,
			 list of states (with <START> and <END> at index 0 and -1 
			 				respectively),
			 beta matrix.
	@return: completed beta matrix.
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	for t in range(T - 2, -1, -1):
		for i in range(1, N + 1):
			sigma = 0 #0.00000000001
			for j in range(1, N + 1):
				transition = matrix_a[num_to_token[i]][num_to_token[j]]
				emission = matrix_b[num_to_token[j]][obs[t + 1].split('/')[0].lower()]
				backward = beta[t + 1][j]

				new_unit = math.log(transition) + \
						   math.log(emission) + \
						   backward
				sigma = sigma + math.exp(new_unit)

			if sigma == 0:
				beta[t][i] = math.log(0.00000000001)
			else:
				beta[t][i] = math.log(sigma)

	return beta

def initBackward(matrix_a, matrix_b, obs, num_to_token):
	"""
	initBackward() takes a pair of A and B matrices, a list of observations
	and a list of states, and finds the initial backward
	probabilities (<START> --> state). The beta matrix is returned.

	@params: A and B matrices,
			 list of observations,
			 list of states (with <START> and <END> at index 0 and -1 
			 				respectively).
	@return: beta matrix.
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	beta = [[1 for i in range(N + 2)] for j in range(T)]

	for j in range(1, N + 1):
		beta[T - 1][j] = math.log(matrix_a[num_to_token[j]]['<END>'])

	return beta

def getZeta(matrix_a, matrix_b, alpha, beta, sentence, states):
	"""
	getZeta() takes a pair of A and B matrices, an alpha and a beta matrix,
	a sentence (list of words (strings)) and a list of states, and finds
	the probabilities of having a particular tag sequence pass through a
	particular point for the sentence given.

	@params: A and B matrices,
			 alpha matrix,
			 beta matrix,
			 sentence (list of strings),
			 list of states.
	@return: zeta matrix.
	"""
	N = len(matrix_a) - 2
	T = len(sentence)

	zeta = {}
	# Loops over the times and different state/state combinations in order to
	# find the probability XX
	for t in range(0, T - 1):
		zeta[t] = {}
		for i in range(1, N + 1):
			zeta[t][i] = {}
			for j in range(1, N + 1):
				if t == (T - 1):
					# Last word in the sentence.
					zeta[t][i][j] = alpha[t][i] + math.log(matrix_a[states[i]]['<END>'])
				else:
					zeta[t][i][j] = alpha[t][i] + \
								    math.log(matrix_a[states[i]][states[j]]) + \
								    math.log(matrix_b[states[j]][sentence[t + 1].split('/')[0].lower()]) + \
								    beta[t + 1][j]
					zeta[t][i][j] = math.exp(zeta[t][i][j])

	return zeta

def getGamma(matrix_a, alpha, beta, final_alpha, sentence):
	"""
	getGamma() xx

	@params: xx
	@return: xx
	"""
	N = len(matrix_a) - 2
	T = len(sentence)

	gamma = {}
	for t in range(0, T):
		gamma[t] = {}
		for j in range(1, N + 1):
			gamma[t][j] = math.exp(alpha[t][j] + beta[t][j]) / math.exp(final_alpha)

	return gamma

def initAB(sent_list_list, states):
	"""
	initAB() xx

	@params: xx
	@return: xx
	"""
	# initialize A, B
	matrix_a = {}
	matrix_b = {}

	words_dict = {}
	for line in sent_list_list:
		for word in line:
			if word != '':
				word = word.split('/')[0].lower()

				if word not in words_dict:
					words_dict[word] = 1
				else:
					words_dict[word] += 1

	sent_list_list_cheat = sent_list_list[:int(len(sent_list_list) / 4)]
	for i, sent_list in enumerate(sent_list_list_cheat):
		sent_list_list_cheat[i] = ' '.join(sent_list)

	counts_a = hmm_model_build.getTransitionMatrix(sent_list_list_cheat)
	counts_b = hmm_model_build.getEmissionMatrix(sent_list_list_cheat)


	for line in sent_list_list:
		for word in line:
			word = word.split('/')[0].lower()

			# other probs? XX
			for state in states:
				if state in counts_b:
					if word not in counts_b[state]:
						counts_b[state][word] = 0.00000000001 # e
					else:
						counts_b[state][word] += 0.00000000001 # e
				else:
					counts_b[state] = {}
					counts_b[state][word] = 0.00000000001 # e

				for next_state in states:
					if state not in counts_a:
						counts_a[state] = {}
					
					if next_state not in counts_a[state]:
						counts_a[state][next_state] = 0.00000000001 # e
					else:
						counts_a[state][next_state] += 0.00000000001 # e


	del counts_a['<NOT_SEEN_P>']

	matrix_a = counts_a
	matrix_b = counts_b

	return matrix_a, matrix_b

def forwardBackward(states, sent_list):
	"""
	forwardBackward() xx

	@params: xx
	@return: xx
	"""
	sent_list_list = []
	for line in sent_list:
		new_sent = []
		for word in line.split(' '):
			if word != '':
				new_sent.append(word)

		if len(new_sent) != 0:
			sent_list_list.append(new_sent)

	# initialize A, B
	matrix_a, matrix_b = initAB(sent_list_list, states)

	# iterate until convergence
	for repeat in range(10): # change this to check convergence later on XX
		print('Repeat: ' + str(repeat))
		# Crate a_hat and b_hat.
		a_hat = {}
		b_hat = {}

		for state in states:
			a_hat[state] = {}
			b_hat[state] = {}

			for s in states:
				a_hat[state][s] = 1

			for sent in sent_list:
				for r_word in sent.split(' '):
					word = r_word.split('/')[0].lower()
					b_hat[state][word] = 1
		count = 0
		for sentence in sent_list_list:
			print(count)
			count += 1
			N = len(states) - 2
			T = len(sentence)

			# Get alpha and beta for the observation.
			final_alpha, alpha = forwardAlg(matrix_a, matrix_b, sentence, states) # XX
			final_beta, beta = backwardAlg(matrix_a, matrix_b, sentence, states) # XX

			# print(final_alpha)
			# print(final_beta)
			# print()

			# Get gamma and zeta for the observation.
			zeta = getZeta(matrix_a, matrix_b, alpha, beta, sentence, states)
			gamma = getGamma(matrix_a, alpha, beta, final_alpha, sentence)

			for j in range(1, N + 1):
				b_hat_denom = sum([gamma[t][j] for t in range(T)])
				if b_hat_denom == 0:
					b_hat_denom = 0.00000000001

				for v_k in sentence:
					b_hat_num = 0
					for t in range(T):
						if v_k.split('/')[0].lower() == sentence[t].split('/')[0].lower():
							b_hat_num += gamma[t][j]
					b_hat[states[j]][v_k.split('/')[0].lower()] += b_hat_num / b_hat_denom

			for i in range(1, N + 1):
				for j in range(1, N + 1):
					a_hat_num = 0
					a_hat_denom = 0.00000000001
					for t in range(0, T - 1):
						a_hat_num += zeta[t][i][j]

						a_hat_denom += sum(zeta[t][i][k] for k in range(1, N + 1))
					a_hat[states[i]][states[j]] += a_hat_num / a_hat_denom

		# print(gamma)
		# print()
		# print(zeta)
		# sys.exit()

		# Normalize counts.
		a_hat, b_hat = normalizeCounts(a_hat, b_hat, states)

		# Set A and B to a_hat and b_hat.
		matrix_a = a_hat
		matrix_b = b_hat

	# sys.exit()
	for i in range(10):
		print()
	print(matrix_a)
	for i in range(5):
		print()
	print(matrix_b)

	return matrix_a, matrix_b

def normalizeCounts(a_hat, b_hat, states):
	"""
	normalizeCounts() xx

	@params: xx
	@return: xx
	"""
	totalTransition = {}
	totalSubs = {}
	for state in states:
		totalTransition[state] = 0
		totalSubs[state] = 0

	for i in states:
		totalTransition[i] = sum([a_hat[i][j] for j in states])
		# for j in states:
		# 	totalTransition[i] += a_hat[i][j]
		totalSubs[i] = sum([b_hat[i][j] for j in b_hat[i]])
		# for j in lemmas:
		# 	t

	for state in states:
		for next_s in states:
			a_hat[state][next_s] = a_hat[state][next_s] / totalTransition[state]
		for word in b_hat[state]:
			b_hat[state][word] = b_hat[state][word] / totalSubs[state]


	return a_hat, b_hat

def hmmTrainer(f):
	"""
	hmmTrainer() takes the content of a file and builds A and B matrices
	from that file. The resulting A and B matrices are dumped using pickle.

	@params: content of file.
	@return: n/a (results are dumped using pickle to trainmodel.dat).
	"""
	print('Training HMM matrices A and B...')

	states = ['<START>', 'DET', '.', 'ADJ', 'PRT', 'VERB', 'NUM', 'X', \
			  'CONJ', 'PRON', 'ADV', 'ADP', 'NOUN', '<END>']

	# Cleans the input.
	sent_list = []
	for line in f:
		new_line = re.sub('(\n)+', ' ', line)
		sent_list.append(new_line) # lower case? XX

	matrix_a, matrix_b = forwardBackward(states, sent_list)

	model = {}
	model['a'] = matrix_a
	model['b'] = matrix_b

	pickle.dump(model, open('trainmodel2.dat', 'wb'))
	print('Saving to trainmodel.dat')

def main():
	if len(sys.argv) != 2:
		printError()

	try:
		f = open(sys.argv[1])
	except:
		printError()

	hmmTrainer(f)


if __name__ == '__main__':
	main()