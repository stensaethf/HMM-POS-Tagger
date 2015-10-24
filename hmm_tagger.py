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

def viterbi(matrix_a, matrix_b, obs, prob_unk_state):
	"""
	DOESNT WORK
	"""
	# Create data structures for going back and forth between tokens in string
	# form and integer form.
	num_to_token = []
	num_to_token.append('<START>')
	for token in matrix_a:
		if token not in ['<START>', '<END>', 0, '<SEEN>', '<NOT_SEEN_P>']:
			num_to_token.append(token)
	num_to_token.append('<END>')

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

		if s in matrix_a['<START>']:
			# Everything is good.
			transition = math.log(matrix_a['<START>'][s])
		else:
			# Havent seen that particular state after <START>.
			transition = float('-inf')

		if s in matrix_b:
			if obs[0].lower() in matrix_b[s]:
				# Everything is good.
				emission = math.log(matrix_b[s][obs[0].lower()])
			else:
				# Havent seen that word with that state before.
				emission = math.log(prob_unk_state)
		else:
			# Havent seen that particular state in training.
			emission = float('-inf')



		vit[state][0] = transition + emission
		backpointer[state][0] = 0

	# recursion
	for t in range(1, T):
		for state in range(1, N + 1):
			vit[state][t] = float('-inf')

			s = num_to_token[state]
			for sprev in range(1, N + 1):
				sp = num_to_token[sprev]
				if sp in matrix_a:
					if s in matrix_a[sp]:
						# everything is good.
						transition = math.log(matrix_a[sp][s])
					else:
						# havent seen that state after sp.
						transition = float('-inf')
				else:
					# havent seen that state.
					transition = float('-inf')

				if s in matrix_b:
					if obs[t].lower() in matrix_b[s]:
						# everything is good
						emission = math.log(matrix_b[s][obs[t].lower()])
					else:
						# havent seen that combination.
						emission = math.log(prob_unk_state)
				else:
					# havent seen that state.
					emission = float('-inf')

				vtj = vit[sprev][t - 1] + transition + emission
				if vtj > vit[state][t]:
					vit[state][t] = vtj
					backpointer[state][t] = num_to_token[sprev]

	# termination
	maximum = float('-inf')
	final_state = None
	for state in range(1, N + 1):
		s = num_to_token[state]

		vit_num = vit[state][T - 1]
		if s in matrix_a:
			if '<END>' in matrix_a[s]:
				# we're good
				print(s)
				print(matrix_a[s])
				transition = math.log(matrix_a[s]['<END>'])
			else:
				# havent seen <END> after that state
				transition = float('-inf')
		else:
			# havent seen that state
			transition = float('-inf')

		temp = vit_num + transition

		if temp > maximum:
			maximum = temp
			final_state = s

	# Trace our steps backwards in time to find the complete tag sequence.
	final = final_state
	backtrace = [final]
	for t in range(T - 1, 0, -1):
		index = num_to_token.index(final)
		final = backpointer[index][t]
		backtrace = [final] + backtrace

	# Construct the result.
	result = []
	for i, word in enumerate(obs):
		result.append(word + '/' + backtrace[i])

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

	# Find seen words.
	seen = {}
	for state in matrix_b:
		if state not in [0]:
			for word in matrix_b[state]:
				if word not in seen:
					seen[word] = 1

	# Find dictionary words
	dictionary = {}
	for word in open('/usr/share/dict/words', 'r'):
		word = word.strip()
		
		# Add words we havent seen to dictionary.
		if word not in seen:
			dictionary[word] = 1

	prob_unk = (len(dictionary) - len(seen)) / len(dictionary)
	prob_unk_state = prob_unk / 12

	# Flag unknown words.
	std_input_alter = []
	for index in range(len(std_input)):
		std_input_alter.append(std_input[index])
		if std_input[index].lower() not in seen:
			std_input_alter[index] = '<UNK>'

	# Gets result from viterbi.
	result_raw = viterbi(matrix_a, matrix_b, std_input_alter, prob_unk_state)

	# Transform flagged unknown words back to 'normal' words.
	for index in range(len(result_raw)):
		word = result_raw[index].split('/')[0]
		tag = result_raw[index].split('/')[1]

		if word == '<UNK>':
			result_raw[index] = std_input[index] + '/' + tag

	result = ' '.join(result_raw)
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