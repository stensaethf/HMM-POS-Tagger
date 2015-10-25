'''
hmm_tagger.py
Frederik Roenn Stensaeth
10.23.15

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
	printError() prints out a generic error message and a usage statement.

	@params: n/a.
	@return: n/a.
	"""
	print('Error.')
	print('Usage: $ echo <string to be tagged> | python3 hmm_tagger.py',
	      '[countmodel.dat | trainmodel.dat]')
	sys.exit()

def viterbi(matrix_a, matrix_b, obs, prob_unk_state, real_obs):
	"""
	viterb() takes A (tranisition) and B (emission) matrices, a list of
	observations (list of words) and a probability for <unk> words.

	@params: A and B matrices,
			 list of observations (list of words),
			 probability of <unk> words,
			 real_obs.
	@return: list of observations with tags (list of strings).
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

	# Initialization
	# Sets up the viterbi and backpointer matrices.
	# Loops over all the states and sets calculates the probability of going
	# from <START> to that state.
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

		# Set backpointers to 0 (no backpointer).
		vit[state][0] = transition + emission
		backpointer[state][0] = 0

	# Recursion
	# Fills in the rest of the viterbi and backpointer matrices.
	# Loops over all times and states, and then finds the probability
	# of reaching that particular state and that given time.
	for t in range(1, T):
		for state in range(1, N + 1):
			vit[state][t] = float('-inf')

			s = num_to_token[state]

			# Loops over the states, so that we can find the different
			# transitions.
			for sprev in range(1, N + 1):
				sp = num_to_token[sprev]
				if sp in matrix_a:
					if s in matrix_a[sp]:
						# Everything is good.
						transition = math.log(matrix_a[sp][s])
					else:
						# Havent seen that state after sp.
						transition = float('-inf')
				else:
					# Havent seen that state.
					transition = float('-inf')

				if s in matrix_b:
					if obs[t].lower() in matrix_b[s]:
						# Everything is good
						emission = math.log(matrix_b[s][obs[t].lower()])
					else:
						# Havent seen that combination.
						emission = math.log(prob_unk_state)
						if obs[t] == '<UNK>':
							emission = getEmissionUNK(s, t, real_obs, \
													  prob_unk_state)
						else:
							if s == 'NUM' and obs[t].isdigit():
								emission = math.log(1)
				else:
					# Havent seen that state.
					emission = float('-inf')

				vtj = vit[sprev][t - 1] + transition + emission
				# Checks if we have found a cheaper way of getting to the
				# state at time t. If we have found a cheaper way, we update
				# the cost of getting there and set the backpointer to
				# whatever previous state got us there at this cheaper cost.
				if vtj > vit[state][t]:
					vit[state][t] = vtj
					backpointer[state][t] = num_to_token[sprev]

	# Termination
	# Loops over the states to find the cost of reaching the end from that
	# given state.
	maximum = float('-inf')
	final_state = None
	for state in range(1, N + 1):
		s = num_to_token[state]

		vit_num = vit[state][T - 1]
		if s in matrix_a:
			if '<END>' in matrix_a[s]:
				# We are good
				transition = math.log(matrix_a[s]['<END>'])
			else:
				# Havent seen <END> after that state
				transition = float('-inf')
		else:
			# Havent seen that state
			transition = float('-inf')

		temp = vit_num + transition

		if temp > maximum:
			maximum = temp
			final_state = s

	# Traces our steps backwards in time to find the complete tag sequence.
	final = final_state
	backtrace = [final]
	for t in range(T - 1, 0, -1):
		index = num_to_token.index(final)
		final = backpointer[index][t]
		backtrace = [final] + backtrace

	# Constructs the result.
	result = []
	for i, word in enumerate(obs):
		result.append(word + '/' + backtrace[i])

	return result

def getEmissionUNK(s, t, real_obs, prob_unk_state):
	"""
	getEmissionUNK() finds the emission probabilty of an unknown word using
	suffix patterns that are associated with certain parts of speech. The
	log of the emission probability is returned.

	@params: s (state),
			 t (time),
			 list of observations,
			 probability of unk word.
	@return: emission (~ math.log(prob)).
	"""
	# Checks for special suffixes and assigns emission probabilities 
	# accordingly.
	emission = math.log(prob_unk_state)
	if s == 'NOUN':
		if "'" in real_obs[t]:
			emission = math.log(0.9)
		elif real_obs[t][-1].lower() == '.':
			if real_obs[t][0].upper() == real_obs[t][0]:
				if len(real_obs[t]) <= 4:
					if len(real_obs[t]) != 1:
						emission = math.log(1)
					else:
						emission = math.log(0.5)
				else:
					emission = math.log(0.5)
			else:
				emission = math.log(0.5)
		elif real_obs[t][-5:].lower() == 'arian':
			emission = math.log(1)
		elif real_obs[t][-4:].lower() in ['ment', 'ness', 'ship']:
			emission = math.log(1)
		elif real_obs[t][-3:].lower() in ['ion', 'ess', 'ent', 'ant', 'ity']:
			emission = math.log(0.95)
		elif real_obs[t][-2:].lower() in ['er', 'or', 'ar', 'ee', 'ty']:
			emission = math.log(0.95)
		elif real_obs[t][-1].lower() == 's':
			emission = math.log(0.9)
		else:
			emission = math.log(0.5)
	elif s == 'ADJ':
		if real_obs[t][-4:].lower() in ['able', 'ible', 'less', 'like']:
			emission = math.log(0.95)
		elif real_obs[t][-2:].lower() == 'er':
			emission = math.log(0.8)
		elif real_obs[t][-3:].lower() in ['ive', 'ful', 'est', 'ese']:
			emission = math.log(0.95)
		elif real_obs[t][-3:].lower() == 'ish':
			emission = math.log(1)
		elif real_obs[t][-1].lower() == 'y':
			emission = math.log(0.95)
		else:
			emission = math.log(0.25)
	elif s == 'VERB':
		if real_obs[t][-2:].lower() in ['ed', 'en']:
			emission = math.log(0.9)
		elif real_obs[t][-2:].lower() in ['es']:
			emission = math.log(0.95)
		elif real_obs[t][-3:].lower() in ['ing', 'ize']:
			emission = math.log(1)
		elif real_obs[t][-1].lower() == 's':
			emission = math.log(0.8)
		else:
			emission = math.log(0.25)
	elif s == 'NUM':
		if real_obs[t].isdigit():
			emission = math.log(1)
		elif real_obs[t][-3:] == 'eth':
			emission = math.log(1)
		elif real_obs[t][-2:] == 'th':
			emission = math.log(0.95)
	elif s == 'ADV':
		if real_obs[t][-3:].lower() == 'ily':
			emission = math.log(1)
		elif real_obs[t][-2:].lower() == 'ly':
			emission = math.log(1)
	else:
		emission = math.log(prob_unk_state)

	return emission

def hmmTagger(f, std_in_raw):
	"""
	hmmTagger() takes the content of a file and input from standard input, and
	uses this to find the tag sequence for the input.

	@params: content of file and string to be tagged (standard input).
	@return: n/a (result is output to standard output).
	"""
	# Cleans the input and splits it into a list (split on spaces).
	std_input = std_in_raw.read().strip().split(' ')

	# Loads the file using pickle.
	model = pickle.load(open(f, 'rb'))

	# Tries to obtain the A and B matrices.
	try:
		matrix_a = model['a']
		matrix_b = model['b']
	except:
		printError()

	# Finds seen words.
	seen = {}
	for state in matrix_b:
		if state not in [0]:
			for word in matrix_b[state]:
				if word not in seen:
					seen[word] = 1

	# Finds dictionary words
	dictionary = {}
	for word in open('/usr/share/dict/words', 'r'):
		word = word.strip()
		
		# Adds words we havent seen to dictionary.
		if word not in seen:
			dictionary[word] = 1

	prob_unk = (len(dictionary) - len(seen)) / len(dictionary)
	prob_unk_state = prob_unk / 12

	# Flags unknown words.
	std_input_alter = []
	for index in range(len(std_input)):
		std_input_alter.append(std_input[index])
		if std_input[index].lower() not in seen:
			std_input_alter[index] = '<UNK>'

	# Gets result from viterbi.
	result_raw = viterbi(matrix_a, matrix_b, \
						 std_input_alter, \
						 prob_unk_state, \
						 std_input)

	# Transforms flagged unknown words back to 'normal' words.
	for index in range(len(result_raw)):
		word = result_raw[index].split('/')[0]
		tag = result_raw[index].split('/')[1]

		if word == '<UNK>':
			result_raw[index] = std_input[index] + '/' + tag

	result = ' '.join(result_raw)

	# Sends/ prints the result to standard output.
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