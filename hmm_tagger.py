'''
'''

import sys
import pickle
import os.path

def printError():
	"""
	"""
	print('Error.')
	print('Usage: $ echo <string to be tagged> | python hmm_tagger.py [countmodel.dat | trainmodel.dat]')
	sys.exit

def viterbi(matrix_a, matrix_b, observations):
	"""
	"""
	Xx

	return None

def hmmTagger(f, std_input):
	"""
	"""
	std_input = std_input.read().strip().split(' ')

	model = pickle.load(open(f, 'rb'))
	matrix_a = model['a']
	matrix_b = model['b']

	result = ' '.join(viterbi(matrix_a, matrix_b, std_input))

	print(result)

def main():
	# if len(sys.argv) != 2:
	# 	printError()

	if not os.path.isfile(sys.argv[1]):
		printError()

	hmmTagger(sys.argv[1], sys.stdin)


if __name__ == '__main__':
	main()