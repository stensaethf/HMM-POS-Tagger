'''
hmm_model_build.py
Frederik Roenn Stensaeth
10.18.15

A Python program that creates a model from the pos-tagged training data. 
It takes as a command-line argument a file to use for training, and output 
the model to a file named countmodel.dat.
'''

import sys
import pickle
import re

def printError():
	"""
	WORKS
	"""
	print('Error.')
	print('Usage: $ python3 hmm_model_build.py <file for building>')
	sys.exit()

def getTransitionCounts(sent_list):
	"""
	SHOULD WORK
	"""
	model = {}
	model[0] = 0
	###
	model['<SEEN>'] = 0
	###

	for sent_raw in sent_list:
		sent = ['<START>/<START>'] + sent_raw.split(' ') + ['<END>/<END>']
		for i, word_tag in enumerate(sent):
			model[0] += 1

			word_tag_split = word_tag.split('/')

			if len(word_tag_split) == 2:
				word = word_tag_split[0].lower()
				tag = word_tag_split[1]

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
							###
							model['<SEEN>'] += 1
							###
							model[tag][next_tag] = {}
							model[tag][next_tag][0] = 1

	return model

def getTransitionProbabilities(transition_counts):
	"""
	SHOULD WORK -- smoothing?
	"""
	matrix_a = {}
	matrix_a[0] = transition_counts[0]

	# Loops over and finds all the different bigrams. Calculates the
	# probability of each and stores them in matrix_a.
	for first in transition_counts:
		if first != 0 and first != '<SEEN>':
			# print(transition_counts[first])
			for second in transition_counts[first]:
				if second != 0 and second != '<SEEN>':
					count_bi = transition_counts[first][second][0]
					count_uni = transition_counts[first][0]

					if first in matrix_a:
						matrix_a[first][second] = (count_bi / count_uni)
					else:
						matrix_a[first] = {}
						matrix_a[first][second] = (count_bi / count_uni)

	###
	possible = len(matrix_a) ** 2
	not_seen = (possible - transition_counts['<SEEN>']) / possible
	matrix_a['<NOT_SEEN_P>'] = not_seen
	###

	return matrix_a

def getTransitionMatrix(sent_list):
	"""
	SHOULD WORK
	"""
	transition_counts = getTransitionCounts(sent_list)

	matrix_a = getTransitionProbabilities(transition_counts)

	return matrix_a

def getEmissionCounts(sent_list):
	"""
	SHOULD WORK
	"""
	model = {}
	model[0] = 0

	for sent_raw in sent_list:
		sent = ['<START>/<START>'] + sent_raw.split(' ') + ['<END>/<END>']
		for i, word_tag in enumerate(sent):
			word_tag_split = word_tag.split('/')

			if len(word_tag_split) == 2:

				word = word_tag_split[0].lower()
				tag = word_tag_split[1]

				if tag in model:
					model[tag][0] += 1
				else:
					model[tag] = {}
					model[tag][0] = 1

				if word in model[tag]:
					model[tag][word][0] += 1
				else:
					model[tag][word] = {}
					model[tag][word][0] = 1

	return model

def getEmissionProbabilities(emission_counts):
	"""
	SHOULD WORK -- smoothing?
	"""
	matrix_b = {}
	matrix_b[0] = emission_counts[0]

	for state in emission_counts:
		if state != 0:
			for word in emission_counts[state]:
				if word != 0:
					count_bi = emission_counts[state][word][0]
					count_uni = emission_counts[state][0]

					if state in matrix_b:
						matrix_b[state][word] = (count_bi / count_uni)
					else:
						matrix_b[state] = {}
						matrix_b[state][word] = (count_bi / count_uni)

	return matrix_b

def getEmissionMatrix(sent_list):
	"""
	SHOULD WORK
	b_i(O_t) = P(O_t | q_i)
	"""
	emission_counts = getEmissionCounts(sent_list)

	matrix_b = getEmissionProbabilities(emission_counts)

	return matrix_b

def hmmBuilder(f):
	"""
	SHOULD WORK
	"""
	print('Calculating HMM probabilities...')

	# data = ''
	data = []
	for line in f:
		# data += line
		data.append(line)

	# Seperates the data into sentences.
	# sent_list = data.split('./.')
	sent_list = data
	for i, sent in enumerate(sent_list):
		sent = re.sub('\n', ' ', sent)
		sent_list[i] = sent.strip() #+ ' ./.'
		# print(sent)
	# print(sent_list)

	matrix_a = getTransitionMatrix(sent_list)

	matrix_b = getEmissionMatrix(sent_list)

	# print()
	# print(matrix_b)

	model = {}
	model['a'] = matrix_a
	model['b'] = matrix_b

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