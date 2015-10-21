'''
'''

def forward_alg(matrix_a, matrix_b, sent_list):
	"""
	DOESNT WORK
	"""
	return None

def init_forward(matrix_a, matrix_b, sent_list, alpha):
	"""
	DOESNT WORK
	"""
	n = len(matrix_a) - 2

	for j in range(1, n):
		alpha[1][j] = matrix_a[0][j] * matrix_b[j][sent_list[1]]


	return alpha

def backward_alg(matrix_a, matrix_b):
	"""
	DOESNT WORK
	"""
	return None

def main():
	print()


if __name__ == '__main__':
	main()