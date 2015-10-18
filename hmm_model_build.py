'''
'''

import sys
import pickle

def printError():
	"""
	"""
	print('Error.')
	print('Usage: $ python3 hmm_model_build.py <file for building>')
	sys.exit()

def getTransitionCounts(sent_list):
	"""
	"""
	model = {}
	model[0] = 0

	for sent_raw in sent_list:
		sent = sent_raw.split(' ')
		for i, word_tag in enumerate(sent):
			word_tag_split = word_tag.split('/')

			if len(word_tag_split) == 2:
				word = word_tag_split[0]
				tag = word_tag_split[1]

				if tag in model:
					model[tag][0] += 1
				else:
					model[tag] = {}
					model[tag][0] = 1

				# Checks whether we are too close to the end of the sentence to
				# create a bigram.
				if i + 1 < len(sent):
					next_word_tag_split = sent[i + 1].split('/')

					if len(next_word_tag_split) == 2:
						next_tag = next_word_tag_split[1]
						if next_tag in model[tag]:
							model[tag][next_tag][0] += 1
						else:
							model[tag][next_tag] = {}
							model[tag][next_tag][0] = 1

	return model

def getTransitionProbabilities(transition_counts):
	"""
	"""
	matrix_a = {}
	matrix_a[0] = transition_counts[0]

	# Loops over and finds all the different bigrams. Calculates the
	# probability of each and stores them in matrix_a.
	for first in transition_counts:
		if first != 0:
			for second in transition_counts[first]:
				if second != 0:
					count_bi = transition_counts[first][second][0]
					count_uni = transition_counts[first][0]

					matrix_a[first] = {}
					matrix_a[first][second] = (count_bi / count_uni)

	return matrix_a

def getTransitionMatrix(sent_list):
	"""
	"""
	transition_counts = getTransitionCounts(sent_list)

	matrix_a = getTransitionProbabilities(transition_counts)

	return matrix_a

def getEmissionCounts(sent_list):
	"""
	"""
	model = {}
	model[0] = 0

	for sent_raw in sent_list:
		sent = sent_raw.split(' ')
		for i, word_tag in enumerate(sent):
			word_tag_split = word_tag.split('/')

			if len(word_tag_split) == 2:

				word = word_tag_split[0]
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
	"""
	matrix_b = {}
	matrix_b[0] = emission_counts[0]

	for state in emission_counts:
		if state != 0:
			for word in emission_counts[state]:
				if word != 0:
					count_bi = emission_counts[state][word][0]
					count_uni = emission_counts[state][0]

					matrix_b[state] = {}
					matrix_b[state][word] = (count_bi / count_uni)

	return matrix_b

def getEmissionMatrix(sent_list):
	"""
	b_i(O_t) = P(O_t | q_i)
	"""
	emission_counts = getEmissionCounts(sent_list)

	matrix_b = getEmissionProbabilities(emission_counts)

	return matrix_b

def hmmBuilder(f):
	"""
	"""
	data = ''
	for line in f:
		data += line

	# Seperates the 
	sent_list = data.split('./.')
	matrix_a = getTransitionMatrix(sent_list)

	matrix_b = getEmissionMatrix(sent_list)

	model = {}
	model['a'] = matrix_a
	model['b'] = matrix_b

	pickle.dump(model, open('countmodel.dat', 'wb'))

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