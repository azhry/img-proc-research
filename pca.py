# -*- coding: utf-8 -*-
import numpy as np

dirName = "C:\\Users\\acer\\Documents\\NetBeansProjects\\DWT\\img\\w21\\and"

def createDataMatrix(images):
    
    smallestSize = getSmallestSize(images)
    data = np.zeros((smallestSize, len(images)))
    
    for x in range(0, len(images)):
        idx = 0
        for i in range(0, len(images[x])):
            for j in range(0, len(images[x][0])):
                data[idx][x] = images[x][i][j]
                idx += 1
        
    return data

def getSmallestSize(images):
    size = []
    for image in images:
        size.append(getMatrixSize(image))
    return min(size)

def getMatrixSize(matrix):
    return len(matrix) * len(matrix[0])
    
def getVectorMean(matrix):
    result = np.zeros((len(matrix), 1))
    for i in range(0, len(matrix)):
        result[i][0] = np.mean(matrix[i])
    return result

def joinMatrix(vector, images):
    smallestSize = getSmallestSize(images)
    result = np.zeros((smallestSize, len(images)))
    for x in range(0, len(images)):
        flattened = np.array(images[x]).flatten()
        subtracted = np.subtract(flattened, vector.flatten())
        for i in range(0, smallestSize):
            result[i][x] = subtracted[i]
        
    return result
    
def getCovariance(matrix):
    return np.matmul(matrix.transpose(), matrix)

def getPrincipalComponent(eigenvalues, eigenvectors):
    tot = sum(eigenvalues)
    var_exp = [(i / tot)*100 for i in sorted(eigenvalues, reverse=True)] 
    numComponents = 0
    for variance in var_exp:
        if variance >= 10:
            numComponents += 1 
    return constructProjection(eigenvalues, eigenvectors, numComponents)
    
def constructProjection(eigenvalues, eigenvectors, numComponents):
        eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
        tuples = ()
        for i in range(numComponents):
            tuples += (eigen_pairs[i][1].reshape(len(eigenvalues), 1),)
        matrix_w = np.hstack(tuples)
        
        return matrix_w

images = [
    [
        [1, 2],
        [3, 4]
    ],
    [
        [3, 6],
        [8, 5]
    ],
    [
        [9, 8],
        [4, 5]
    ]
]

vectorMean = getVectorMean(createDataMatrix(images))
covariance = getCovariance(joinMatrix(vectorMean, images))
eigenvalues, eigenvectors = np.linalg.eig(covariance)
matrix_w = getPrincipalComponent(eigenvalues, eigenvectors)
print(matrix_w)