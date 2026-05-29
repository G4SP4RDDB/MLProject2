import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

def PCA(X, d,train_labels_classif):
    cls_names = ["low", "mid", "high"]
    colors = ["green", "yellow", "red"]
    '''
    Input:
        X: NxD matrix representing our normalised data 
        d: Number of principal components to be used to reduce dimensionality

    Output:
        no output: uncomment the plot lines to see the data visualisation.
    '''
    ### WRITE YOUR CODE BELOW ###

    C = X.T @ X / X.shape[0]
    # Compute the eigenvectors and eigenvalues. Hint: look into np.linalg.eigh()
    eigvals, eigvecs = np.linalg.eigh(C)
    # Choose the top d eigenvalues and corresponding eigenvectors.
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    W = eigvecs[:, 0:d]
    eg = eigvals[0:d]

    Y = X @ W

    # Compute the explained variance
    exvar = 100 * eg.sum() / eigvals.sum()

    print(f'The total variance explained by the first {d} principal components is {exvar:.3f} %')
    plt.figure()
    for ind, name in enumerate(cls_names):
        filtered_class = train_labels_classif == ind
        plt.scatter(Y[filtered_class, 0], Y[filtered_class, 1], c=colors[ind], label=name)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.title(f"PCA - 2 principal components ({exvar:.1f}% variance explained)")
    #plt.show()
    #plt.close()