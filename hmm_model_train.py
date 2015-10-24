'''
'''

import pickle
import sys
import math
import os.path
import re
import hmm_model_build

def printError():
	"""
	WORKS
	"""
	print('Error.')
	print('Usage: $ python3 hmm_model_train.py <input file>')
	sys.ezetat()

def forwardAlg(matrix_a, matrix_b, obs, num_to_token):
	"""
	WORKS
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	# init
	alpha = initForward(matrix_a, matrix_b, obs, num_to_token)

	# recursion
	alpha = recursionForward(matrix_a, matrix_b, obs, num_to_token, alpha)

	# termination
	result = 0
	for i in range(1, N + 1):
		transition = matrix_a[num_to_token[i]]['<END>']
		forward = alpha[T - 1][i]

		result = result + math.exp(math.log(transition) + forward)

		# print(forward)
		# print(transition)
	
	# print(matrix_a)	
	# print(result)
	# print(obs)
	return math.log(result), alpha

def recursionForward(matrix_a, matrix_b, obs, num_to_token, alpha):
	"""
	WORKS
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	for t in range(1, T):
		for j in range(1, N + 1):
			sigma = 0.00000000001
			for i in range(1, N + 1):
				transition = matrix_a[num_to_token[i]][num_to_token[j]]
				emission = matrix_b[num_to_token[j]][obs[t].split('/')[0].lower()]
				forward = alpha[t - 1][i]

				new_unit = math.log(transition) + \
						   math.log(emission) + \
						   forward
				sigma = sigma + math.exp(new_unit)

			alpha[t][j] = math.log(sigma)

	return alpha

def initForward(matrix_a, matrix_b, obs, num_to_token):
	"""
	WORKS
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	# T rows, N columns
	alpha = [[1 for i in range(N + 2)] for j in range(T)]
	for j in range(1, N + 1):
		# print(num_to_token)
		# print(N)
		# for i in matrix_a:
		# 	print(i)

		# print(j)
		# print(len(num_to_token))
		transition = matrix_a['<START>'][num_to_token[j]]

		# print(obs[0])
		# print(matrix_b[num_to_token[j]])
		emission = matrix_b[num_to_token[j]][obs[0].split('/')[0].lower()]

		alpha[0][j] = math.log(transition) + math.log(emission)

	return alpha

def backwardAlg(matrix_a, matrix_b, obs, num_to_token):
	"""
	WORKS
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	# init
	beta = initBackward(matrix_a, matrix_b, obs, num_to_token)

	# recursion
	beta = recursionBackward(matrix_a, matrix_b, obs, num_to_token, beta)

	# termination
	result = 0.00000000001
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
	WORKS
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	for t in range(T - 2, -1, -1):
		for i in range(1, N + 1):
			sigma = 0.00000000001
			for j in range(1, N + 1):
				transition = matrix_a[num_to_token[i]][num_to_token[j]]
				emission = matrix_b[num_to_token[j]][obs[t + 1].split('/')[0].lower()]
				backward = beta[t + 1][j]

				new_unit = math.log(transition) + \
						   math.log(emission) + \
						   backward
				sigma = sigma + math.exp(new_unit)

			beta[t][i] = math.log(sigma)

	return beta

def initBackward(matrix_a, matrix_b, obs, num_to_token):
	"""
	WORKS
	"""
	N = len(matrix_a) - 2
	T = len(obs)

	beta = [[1 for i in range(N + 2)] for j in range(T)]

	for j in range(1, N + 1):
		beta[T - 1][j] = math.log(matrix_a[num_to_token[j]]['<END>'])

	return beta

def getZeta(matrix_a, matrix_b, alpha, beta, sentence, states):
	"""
	DOESNT WORK
	"""
	N = len(matrix_a) - 2
	T = len(sentence)

	zeta = {}
	for t in range(0, T - 1): # should this be 'T - 1'? XX
		zeta[t] = {}
		for i in range(1, N + 1):
			zeta[t][i] = {}
			for j in range(1, N + 1):
				if t == (T - 1):
					# last word in the sentence.
					zeta[t][i][j] = alpha[t][i] + math.log(matrix_a[states[i]]['<END>'])
				else:
					# print([sentence[t + 1].split('/')[0]])
					zeta[t][i][j] = alpha[t][i] + \
								    math.log(matrix_a[states[i]][states[j]]) + \
								    math.log(matrix_b[states[j]][sentence[t + 1].split('/')[0].lower()]) + \
								    beta[t + 1][j]
			# print(alpha[t][i])
			# print(zeta[t][i])
	# print(zeta)
	# print()
	return zeta

def getGamma(matrix_a, alpha, beta, final_alpha, sentence):
	"""
	DOESNT WORK
	"""
	N = len(matrix_a) - 2
	T = len(sentence)

	gamma = {}
	for t in range(0, T):
		gamma[t] = {}
		for j in range(1, N + 1):
			gamma[t][j] = alpha[t][j] + beta[t][j] - final_alpha

	# print(gamma)
	# print()
	return gamma

def initAB(sent_list_list, states):
	"""
	DOESNT WORK
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

			for state in states:
				if state in counts_b:
					if word not in counts_b[state]:
						counts_b[state][word] = 0.00000000001 # e
				else:
					counts_b[state] = {}
					counts_b[state][word] = 0.00000000001 # e

				for next_state in states:
					if state not in counts_a:
						counts_a[state] = {}
					
					if next_state not in counts_a[state]:
						counts_a[state][next_state] = 0.00000000001 # e


	del counts_a['<NOT_SEEN_P>']

	matrix_a = counts_a
	matrix_b = counts_b

	# print(counts_a)
	# print()
	# print(counts_b)
	# sys.exit()

	# for i in states:
	# 	matrix_a[i] = {}
	# 	matrix_b[i] = {}
	# 	for j in states:
	# 		matrix_a[i][j] = 1 / len(states)

	# 	for line in sent_list_list:
	# 		for word in line:
	# 			word = word.split('/')[0]

	# 			if (word not in matrix_b[i]) and (word != ''):
	# 				matrix_b[i][word] = 1 / len(words_dict)

	return matrix_a, matrix_b

def forwardBackward(states, sent_list):
	"""
	DOESNT WORK
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
	for repeat in range(5): # change this to check convergence later on XX
		print(repeat)
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

			# print(sentence)

			# Get alpha and beta for the observation.
			final_alpha, alpha = forwardAlg(matrix_a, matrix_b, sentence, states) # XX
			final_beta, beta = backwardAlg(matrix_a, matrix_b, sentence, states) # XX

			# print(final_alpha)
			# print(final_beta)
			# print()

			# print(final_alpha)
			# print(final_beta)

			# Get gamma and zeta for the observation.
			zeta = getZeta(matrix_a, matrix_b, alpha, beta, sentence, states)
			gamma = getGamma(matrix_a, alpha, beta, final_alpha, sentence)
			# print(gamma)
			# print()
			# Find new a_hat and b_hat counts/ probs.
			# M-step.
			# print('ajanfnlkeanlkfaenlfanklklnankank1')

			for j in range(1, N + 1):
				b_hat_denom = sum([math.exp(gamma[t][j]) for t in range(T)])

				for v_k in sentence:
					b_hat_num = 0
					for t in range(T):
						if v_k.split('/')[0].lower() == sentence[t].split('/')[0].lower():
							b_hat_num += math.exp(gamma[t][j])
					# print(b_hat_denom)
					# print(b_hat_num)
					# print(v_k.split('/')[0])
					# print(math.log(b_hat_num))
					# print(math.log(b_hat_denom))
					b_hat[states[j]][v_k.split('/')[0]] = math.exp(math.log(b_hat_num) - math.log(b_hat_denom))

			# print('zeta')
			for i in range(1, N + 1):
				for j in range(1, N + 1):
					a_hat_num = 0
					a_hat_denom = 0
					for t in range(0, T - 1):
						# print(sentence)
						# print(t)
						# print(zeta[t])
						a_hat_num += math.exp(zeta[t][i][j])

						a_hat_denom += sum(math.exp(zeta[t][i][k]) for k in range(1, N + 1))
					print(a_hat_num)
					print(a_hat_denom)
					if a_hat_denom != 0:
						a_hat[states[i]][states[j]] = math.exp(math.log(a_hat_num) - math.log(a_hat_denom))


		# Normalize a_hat
		for s_f in a_hat:
			total = sum([a_hat[s_f][s_s] for s_s in a_hat[s_f]])

			for s_s in a_hat[s_f]:
				a_hat[s_f][s_s] = (a_hat[s_f][s_s] / total)

		# Normalize b_hat
		for s_f in b_hat:
			total = sum([b_hat[s_f][state] for state in b_hat[s_f]])

			for state in b_hat[s_f]:
				b_hat[s_f][state] = b_hat[s_f][state] / total


		# for s_f in a_hat:
		# 	total = 0
		# 	for s_s in a_hat[s_f]:
		# 		total += a_hat[s_f][s_s]
		# 	# print(s_f)
		# 	# print(a_hat[s_f])
		# 	for s_s in a_hat[s_f]:
		# 		if s_f != '<END>':
		# 			print(a_hat[s_f][s_s])
		# 			a_hat[s_f][s_s] = (a_hat[s_f][s_s] / total)
		# 		else:
		# 			a_hat[s_f][s_s] = 0

		# # Normalize b_hat
		# for state in b_hat:
		# 	total = 0
		# 	for word in b_hat[state]:
		# 		total += b_hat[state][word]

		# 	for word in b_hat[state]:
		# 		if state != '<END>' and state != '<START>':
		# 			# print(word)
		# 			# print(state)
		# 			# print(b_hat[state])
		# 			b_hat[state][word] = (b_hat[state][word] / total)
		# 		else:
		# 			b_hat[state][word] = 0

		# Set A and B to a_hat and b_hat.
		matrix_a = a_hat
		matrix_b = b_hat

	# sys.exit()
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
		new_line = re.sub('(\n)+', ' ', line)
		sent_list.append(new_line) # lower case? XX

	matrix_a, matrix_b = forwardBackward(states, sent_list)

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