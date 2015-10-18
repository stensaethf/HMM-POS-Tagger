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

def hmmTagger(f, st_input):
	"""
	"""
	result = st_input.strip().split(' ')

	model = pickle.load(open(f, 'rb'))
	print(model['a'])
	print()
	print(model['b'])




	# result = ' '.join(viterbi(Xx))

	print(result)

def main():
	# if len(sys.argv) != 2:
	# 	printError()

	if not os.path.isfile(sys.argv[1]):
		printError()

	hmmTagger(sys.argv[1], sys.stdin.read())


if __name__ == '__main__':
	main()