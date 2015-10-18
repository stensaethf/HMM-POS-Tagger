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

def getTransitionBigrams(sent_list):
	"""
	"""
	model = {}
	model[0] = 0

	for sent_raw in sent_list:
		sent = sent_raw.split(' ')
		for i, word_tag in enumerate(sent):
			word = word_tag.split('/')[0]
			tag = word_tag.split('/')[1]

			if tag in model:
				model[tag][0] += 1
			else:
				model[tag] = {}
				model[tag][0] = 1

			# Checks whether we are too close to the end of the sentence to
			# create a bigram.
			if i + 1 < len(sent):
				next_tag = sent[i + 1].split('/')[1]
				if next_tag in model[tag]:
					model[tag][next_tag][0] += 1
				else:
					model[tag][next_tag] = {}
					model[tag][next_tag][0] = 1

	return model

def getEmissions(sent_list):
	"""
	"""
	return None

def hmmBuilder(f):
	"""
	"""
	sent_list = f.split('./.')
	matrix_a = getTransitionBigrams(sent_list)

	matrix_b = getEmissionBigrams(sent_list)

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