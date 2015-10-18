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

def hmmBuilder(f):
	"""
	"""
	matrix_a = Xx

	matrix_b = Xx

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