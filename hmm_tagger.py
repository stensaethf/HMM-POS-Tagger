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
import math

def printError():
	"""
	WORKS
	"""
	print('Error.')
	print('Usage: $ echo <string to be tagged> | python3 hmm_tagger.py',
	      '[countmodel.dat | trainmodel.dat]')
	sys.exit()

def viterbi(matrix_a, matrix_b, obs):
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
	# print(num_to_token)
	# print(num_to_token)

	N = len(num_to_token) - 2 # Do not want to count <START> and <END>
	T = len(obs)

	# Setups the vit and backpointer matrices that we will use for computing
	# the most probable tag sequence and for keeping track of what tags that
	# tag sequence actually consists of.
	# [[0 for i in range(column)] for j in range(row)]
	vit = [[1 for i in range(T)] for j in range(N + 2)]
	backpointer = [[1 for i in range(T)] for j in range(N + 2)]

	# NOTE:
	# From here on we need to be defensive. The words may not have been seen
	# before, so need to alter the code beneath so that it can handle that.

	# initialization
	for state in range(1, N + 1):
		s = num_to_token[state]
		# print(s)
		# print(matrix_a['<START>'][s])
		# print(s)
		# print(obs[0])

		if s in matrix_a['<START>']:
			# Everything is good.
			transition = matrix_a['<START>'][s]
		else:
			# Havent seen that particular state after <START>.
			transition = 0.00001 # gt

		if s in matrix_b:
			if obs[0].lower() in matrix_b[s]:
				# Everything is good.
				emission = matrix_b[s][obs[0].lower()]
			else:
				# Havent seen that word with that state before.
				emission = 0.00001
		else:
			# Havent seen that particular state in training.
			emission = 0.00001



		vit[state][0] = math.log(transition) + math.log(emission)
		backpointer[state][0] = 0

	# print(backpointer)
	# print()
	# print(vit)

	####
	# WORKS TILL HERE
	####

	# recursion
	for t in range(1, T):
		for state in range(1, N + 1):
			vit[state][t] = float('-inf')
			# print()
			# print(vit)
			# print()

			s = num_to_token[state]
			for sprev in range(1, N + 1):
				sp = num_to_token[sprev]
				if sp in matrix_a:
					if s in matrix_a[sp]:
						# everything is good.
						transition = matrix_a[sp][s]
					else:
						# havent seen that state after sp.
						transition = 0.00001 # gt
				else:
					# havent seen that state.
					transition = 0.00001

				if s in matrix_b:
					if obs[t].lower() in matrix_b[s]:
						# everything is good
						emission = matrix_b[s][obs[t].lower()]
					else:
						# havent seen that combination.
						emission = 0.00001
				else:
					# havent seen that state.
					emission = 0.00001

				vtj = vit[sprev][t - 1] + math.log(transition) + math.log(emission)
				# print(vtj)
				# print(vit[state][t])
				if vtj > vit[state][t]:
					vit[state][t] = vtj
					backpointer[state][t] = num_to_token[sprev]
	# print(vit)
	# print(len(vit))
	# print(len(vit[0]))
	# sys.exit()
	# print(backpointer)
	# sys.exit()

	####
	# SHOULD WORK TILL HERE
	####

	# termination
	maximum = float('-inf')
	final_state = None
	for state in range(1, N + 1):
		s = num_to_token[state]

		vit_num = vit[state][T - 1]
		# print(s)
		# print(matrix_a[s])
		if s in matrix_a:
			if '<END>' in matrix_a[s]:
				# we're good
				transition = matrix_a[s]['<END>']
			else:
				# havent seen <END> after that state
				transition = 0.00001
		else:
			# havent seen that state
			transition = 0.00001

		temp = vit_num + math.log(transition)

		# print('temp: ' + str(temp))
		# print('max: ' + str(maximum))

		if temp > maximum:
			# '<END>'
			# print(transition)
			# print(vit_num)
			# print('entered entered entered entered entered entered')
			# for index in range(1, N):
			# 	vit[index][T] = vit_num + math.log(transition)
			# 	# print(s)
			# 	# print(temp)
			# 	backpointer[index][T] = s
			maximum = temp
			final_state = s

	# print(maximum)
	# print(s)
	# print(backpointer[num_to_token.index(s)][T - 1])
	# print(backpointer[num_to_token.index(backpointer[num_to_token.index(s)][T - 1])][T - 1])
	# sys.exit()

	# Find sequence of tags via the backpointer matrix.

	# print()
	# print(vit)
	# print()
	# print(backpointer)
	# print('BACKTRACE')

	# Trace our steps backwards in time to find the complete tag sequence.
	final = final_state
	backtrace = [final]
	# print(final)
	for t in range(T - 1, 0, -1):
		# print(t)
		# print(final)
		index = num_to_token.index(final)
		# print(index)
		# print(backpointer[index][t])
		final = backpointer[index][t]
		backtrace = [final] + backtrace

	# print(backtrace)

	# Construct the result.
	result = []
	for i, word in enumerate(obs):
		result.append(word + '/' + backtrace[i])
	# print(result)

	return result

def hmmTagger(f, std_in_raw):
	"""
	DOESNT WORK
	"""
	std_input = std_in_raw.read().strip().split(' ')

	model = pickle.load(open(f, 'rb'))

	try:
		matrix_a = model['a']
		matrix_b = model['b']
	except:
		printError()

	result = ' '.join(viterbi(matrix_a, matrix_b, std_input))

	# print(result)
	# send/ print to standard output.
	sys.stdout.write(result)

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