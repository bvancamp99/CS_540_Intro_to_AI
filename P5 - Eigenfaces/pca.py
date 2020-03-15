# Author: Bryce Van Camp
# Project: p5
# File: pca.py

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.linalg import eigh

# Load the dataset from a provided .mat file, re-center it around the origin,
# and return it as a NumPy array of floats.
#
# filename - the provided .mat file
def load_and_center_dataset(filename):
    dataset = loadmat(filename)
    x = dataset['fea'].astype('float64')
    
    # re-center
    x -= np.mean(x, axis=0)
    
    return x
    
# Calculate and return the covariance matrix of the dataset as a NumPy matrix 
# (d x d array).
#
# dataset - dataset loaded from the .mat file
def get_covariance(dataset):
    x = dataset
    n = len(x)
    x_t = np.transpose(x)
    
    # Sample covariance matrix formula
    S = 1 / (n - 1) * np.dot(x_t, x)
    
    # S is 1024x1024
    return S
    
# Perform eigen decomposition on the covariance matrix S and return a diagonal
# matrix (NumPy array) with the largest m eigenvalues on the diagonal, and a
# matrix (NumPy array) with the corresponding eigenvectors as columns.
#
# S - the covariance matrix
# m - number of largest eigenvalues on the diagonal
def get_eig(S, m):
    # eigen decomposition
    eigvals, eigvecs = eigh(S, eigvals=(len(S) - m, len(S) - 1))
    
    # rearrange output
    Lambda = np.asarray([[0.] * m for i in range(m)])
    for i in range(m):
        Lambda[i][i] = eigvals[m - 1 - i]
    U = np.asarray([list(reversed(x)) for x in eigvecs])
    
    return Lambda, U

# Project each image into your m-dimensional space and return the new 
# representation as a d x 1 NumPy array.
#
# image - image to project
# U - eigenvectors
def project_image(image, U):
    U_t = np.transpose(U)
    
    # 2x1024 dot 1024x1 = 2x1 matrix
    proj = np.dot(U_t, image)
    
    # 1024x2 dot 2x1 = 1024x1 projected image
    proj = np.dot(U, proj)
    
    return proj
    
# Use matplotlib to display a visual representation of the original image and 
# the projected image side-by-side.
#
# orig - original image
# proj - projected image
def display_image(orig, proj):
    orig = orig.reshape(32, 32).transpose()
    proj = proj.reshape(32, 32).transpose()
    
    # setup
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.set_title('Original')
    ax2.set_title('Projection')
    
    # render imgs and colorbar
    bar1 = ax1.imshow(orig, aspect='equal')
    fig.colorbar(bar1, ax=ax1)
    bar2 = ax2.imshow(proj, aspect='equal')
    fig.colorbar(bar2, ax=ax2)
    
    # render plots
    plt.show()