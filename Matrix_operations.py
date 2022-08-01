#this file contains the matrix function necessary for learning

def matrix_subtract(matrix1, matrix2):
    '''Input: accepts two matrices, assumes them to be the same size
    Output: returns the result of matrix subtraction'''
    return [[matrix1[j][i] - matrix2[j][i] for i in range(len(matrix1))] for j in range(len(matrix1[0]))]

def matrix_addition(matrix1, matrix2):
    '''Input: accepts two matrices, assumes them to be the same size
    Output: returns the result of matrix subtraction'''
    return [[matrix1[j][i] + matrix2[j][i] for i in range(len(matrix1))] for j in range(len(matrix1[0]))]

def matrix_dot_prod(matrix, scalar):
    '''Input: accepts one matrix and one scalar value
    Output: returns the result of matrix dot product'''
    return [[element * scalar for element in row ] for row in matrix]

def matrix_transpose(matrix):
    '''Input: accepts a matrix
    Output: returns the transpose of the matrix'''
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

def matrix_multiplication(matrix1, matrix2):
    '''Input: accepts two matrices, assumes them to be the same size
    Output: returns the result of matrix multiplication'''
    return [[sum([matrix1[j][i] * matrix2[i][k] for i in range(len(matrix1))]) for k in range(len(matrix2[0]))] for j in range(len(matrix1[0]))]