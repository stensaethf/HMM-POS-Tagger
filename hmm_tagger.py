'''
hmm_tagger.py
Frederik Roenn Stensaeth
10.18.15

A Python program that uses the Viterbi algorithm and the model file that 
was created to determine the most probable tag sequence given a sequence 
of words/tokens.
'''

import sys
import pickle
import os.path

def printError():
	"""
	"""
	print('Error.')
	print('Usage: $ echo <string to be tagged> | python hmm_tagger.py',
	      '[countmodel.dat | trainmodel.dat]')
	sys.exit()

def viterbi(matrix_a, matrix_b, obs):
	"""
	"""
	# Create data structures for going back and forth between tokens in string
	# form and integer form.
	token_to_num = {}
	token_to_num['<START>'] = 0
	num_to_token = []
	num_to_token.append('<START>')
	for token in matrix_a:
		if token not in ['<START>', '<END>', 0]:
			num_to_token.append(token)
			token_to_num[token] = len(num_to_token) - 1
	num_to_token.append('<END>')
	token_to_num['<END>'] = len(num_to_token) - 1

	n = len(matrix_a) - 2 # Do not want to count <START> and <END>
	T = len(obs)

	# Setups the vit and backpoitner matrices that we will use for computing
	# the most probable tag sequence and for keeping track of what tags that
	# tag sequence actually consists of.
	# [[0 for i in range(column)] for j in range(row)] 
	vit = [[0 for i in range(T)] for j in range(n + 2)]
	backpointer = [[0 for i in range(T)] for j in range(n + 2)]

	# NOTE:
	# From here on we need to be defensive. The words may not have been seen
	# before, so need to alter the code beneath so that it can handle that.

	for state in range(1, n):
		s = num_to_token[state]
		print(matrix_a['<START>'][s])
		print(s)

		vit[state][0] = matrix_a['<START>'][s] * matrix_b[s][obs[0]]
		backpointer[state][0] = 0

	for t in range(2, T):
		for state in range(1, n):
			vit[state][t] = float('-inf')
			s = num_to_token[state]
			for sprev in range(1, n):
				sp = num_to_token[sprev]
				vtj = vit[sprev][t - 1] * matrix_a[sp][s] * matrix_b[s][obs[t]]
				if vtj > vit[state][t]:
					vit[state][t] = vtj
					backpointer[state][t] = num_to_token[sprev]

	# Find sequence of tags.

	return None

def hmmTagger(f, std_in_raw):
	"""
	"""
	std_input = ['<START>'] + std_in_raw.read().strip().split(' ') + ['<END>']

	model = pickle.load(open(f, 'rb'))
	matrix_a = model['a']
	matrix_b = model['b']

	result = ' '.join(viterbi(matrix_a, matrix_b, std_input))

	print(result)

def main():
	if len(sys.argv) != 2:
		printError()

	if not os.path.isfile(sys.argv[1]):
		printError()

	if sys.argv[1] not in ['countmodel.dat', 'trainmodel.dat']:
		printError()

	hmmTagger(sys.argv[1], sys.stdin)


if __name__ == '__main__':
	main()