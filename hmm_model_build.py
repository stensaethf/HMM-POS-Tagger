'''
hmm_model_build.py
Frederik Roenn Stensaeth
10.23.15

A Python program that creates a model from the pos-tagged training data. 
It takes as a command-line argument a file to use for training, and output 
the model to a file named countmodel.dat.
'''

import sys
import pickle
import re

def printError():
	"""
	printError() prints out a generic error message and a usage statement.

	@params: n/a.
	@return: n/a.
	"""
	print('Error.')
	print('Usage: $ python3 hmm_model_build.py <file for building>')
	sys.exit()

def getTransitionCounts(sent_list):
	"""
	getTransitionCounts() takes a list of sentences and counts all the
	different transitions. The transition counts are stored in a dictionary
	and returned.

	@params: list of sentences (list of strings).
	@return: dictionary[<original state>][<next state>] (keys are strings).
	"""
	model = {}
	model[0] = 0
	model['<SEEN>'] = 0

	# Loops over the sentences in the list and counts the different
	# transitions between states.
	for sent_raw in sent_list:
		sent = ['<START>/<START>'] + sent_raw.split(' ') + ['<END>/<END>']
		for i, word_tag in enumerate(sent):
			model[0] += 1

			word_tag_split = word_tag.split('/')

			# Makes sure the word/tag string is valid.
			if len(word_tag_split) == 2:
				word = word_tag_split[0].lower()
				tag = word_tag_split[1]

				# Checks if the tag has been seen before. Adds the tag to
				# the dictionary if it has not been seen before, if it has
				# been seen the count is incremented.
				if tag in model:
					model[tag][0] += 1
				else:
					model[tag] = {}
					model[tag][0] = 1

				# Checks whether we are too close to the end of the sentence
				# to check the next token.
				if i + 1 < len(sent):
					next_word_tag_split = sent[i + 1].split('/')

					if len(next_word_tag_split) == 2:
						next_tag = next_word_tag_split[1]
						if next_tag in model[tag]:
							model[tag][next_tag][0] += 1
						else:
							model['<SEEN>'] += 1

							model[tag][next_tag] = {}
							model[tag][next_tag][0] = 1

	return model

def getTransitionProbabilities(transition_counts):
	"""
	getTransitionProbabilities() takes a dictionary with transition counts
	and calculates the probabilities for the different transitions. The
	probabilities are stored in a dictionary and returned.

	@params: dictionary of counts (keys are states).
	@return: dictionary of probabilities (keys are states).
	"""
	matrix_a = {}
	matrix_a[0] = transition_counts[0]

	# Loops over and finds all the different transitions. Calculates the
	# probability of each and stores them in matrix_a.
	for first in transition_counts:
		if first != 0 and first != '<SEEN>':
			for second in transition_counts[first]:
				if second != 0 and second != '<SEEN>':
					count_bi = transition_counts[first][second][0]
					count_uni = transition_counts[first][0]

					if first in matrix_a:
						matrix_a[first][second] = (count_bi / count_uni)
					else:
						matrix_a[first] = {}
						matrix_a[first][second] = (count_bi / count_uni)

	possible = len(matrix_a) ** 2
	not_seen = (possible - transition_counts['<SEEN>']) / possible
	matrix_a['<NOT_SEEN_P>'] = not_seen

	return matrix_a

def getTransitionMatrix(sent_list):
	"""
	getTransitionMatrix() takes a list of sentences and finds the
	probabilities for transiting between two given states. A dictionary
	with the probabilities is returned.

	@params: list of sentences (list of strings).
	@return: dictionary of probabilities (keys are states).
	"""
	transition_counts = getTransitionCounts(sent_list)

	matrix_a = getTransitionProbabilities(transition_counts)

	return matrix_a

def getEmissionCounts(sent_list):
	"""
	getEmissionCounts() takes a list of sentences and counts all the
	different emissions. The emission counts are stored in a dictionary
	and returned.

	@params: list of sentences (list of strings).
	@return: dictionary[<state>][<word>] (keys are strings).
	"""
	model = {}
	model[0] = 0

	# Loops over all the sentences and counts the emissions for the states
	# and words.
	for sent_raw in sent_list:
		sent = ['<START>/<START>'] + sent_raw.split(' ') + ['<END>/<END>']
		for i, word_tag in enumerate(sent):
			word_tag_split = word_tag.split('/')

			# Make sure word/tag is valid.
			if len(word_tag_split) == 2:

				word = word_tag_split[0].lower()
				tag = word_tag_split[1]

				# If we have not seen the tag before, we add it. If we have
				# seen it we increment the count.
				if tag in model:
					model[tag][0] += 1
				else:
					model[tag] = {}
					model[tag][0] = 1

				# If we have not seen the word before, we add it. If we have
				# seen it we increment the count.
				if word in model[tag]:
					model[tag][word][0] += 1
				else:
					model[tag][word] = {}
					model[tag][word][0] = 1

	return model

def getEmissionProbabilities(emission_counts):
	"""
	getEmissionProbabilities() takes a dictionary with emission counts
	and calculates the probabilities for the different emissions. The
	probabilities are stored in a dictionary and returned.

	@params: dictionary of counts (keys are states and words).
	@return: dictionary of probabilities (keys are states and words).
	"""
	matrix_b = {}
	matrix_b[0] = emission_counts[0]

	# Loops over all the states and words in the emission count dictionary
	# and calculates the probabilities for those emissions.
	for state in emission_counts:
		if state != 0:
			for word in emission_counts[state]:
				if word != 0:
					count_bi = emission_counts[state][word][0]
					count_uni = emission_counts[state][0]

					# Calculate the probability of the emission.
					if state in matrix_b:
						matrix_b[state][word] = (count_bi / count_uni)
					else:
						matrix_b[state] = {}
						matrix_b[state][word] = (count_bi / count_uni)

	return matrix_b

def getEmissionMatrix(sent_list):
	"""
	getEmissionMatrix() takes a list of sentences and finds the
	probabilities for emissions between a given state and a given word. 
	A dictionary with the probabilities is returned.

	@params: list of sentences (list of strings).
	@return: dictionary of probabilities (keys are states and words).
	"""
	emission_counts = getEmissionCounts(sent_list)

	matrix_b = getEmissionProbabilities(emission_counts)

	return matrix_b

def hmmBuilder(f):
	"""
	hmmBuilder() takes the content of a file and builds the A (tranistion)
	and B (emission) matrices for that file. The matrices are pickled to
	countmodel.dat (dictionary).

	@params: content of a file (... word/tag word/tag ...)
	@return: n/a (results are pickled).
	"""
	print('Calculating HMM probabilities...')

	# Creates a list containing the sentences of the file.
	data = []
	for line in f:
		data.append(line)

	# Seperates the data into sentences.
	sent_list = data
	for i, sent in enumerate(sent_list):
		sent = re.sub('(\n)+', ' ', sent)
		sent_list[i] = sent.strip()

	matrix_a = getTransitionMatrix(sent_list)

	matrix_b = getEmissionMatrix(sent_list)

	# Prepares the results for pickling.
	model = {}
	model['a'] = matrix_a
	model['b'] = matrix_b

	# Results are dumped (saved) as countmodel.dat.
	pickle.dump(model, open('countmodel.dat', 'wb'))
	print('Saving to countmodel.dat')

def main():
	if len(sys.argv) != 2:
		printError()

	try:
		f = open(sys.argv[1])
	except:
		printError()

	hmmBuilder(f)


if __name__ == '__main__':
	main()