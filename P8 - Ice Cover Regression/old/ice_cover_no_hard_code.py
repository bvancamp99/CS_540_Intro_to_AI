# Author: Bryce Van Camp
# Project: P8
# File: ice_cover.py


import statistics
import random


# Takes no arguments and returns the data as described in the specifications 
# in an n-by-2 array.
def get_dataset():
    data_count = 165
    dataset = [[None] * 2 for i in range(data_count)]
    
    with open('x_data.txt', 'r') as f:
        i = 0
        for line in f:
            dataset[i][0] = int(line)
            i += 1
    
    with open('y_data.txt', 'r') as f:
        i = 0
        for line in f:
            dataset[i][1] = int(line)
            i += 1
    
    return dataset


# Takes the dataset as produced by the previous function and prints several 
# statistics about the data; does not return anything.
#
# dataset - data returned by get_dataset()
def print_stats(dataset):
    y_data = [dataset[i][1] for i in range(len(dataset))]
    
    print(len(y_data))
    print('{:.2f}'.format(statistics.mean(y_data)))
    print('{:.2f}'.format(statistics.stdev(y_data)))


# Calculates and returns the mean squared error on the dataset given fixed 
# betas.
#
# beta_0 - first fixed beta
# beta_1 - second fixed beta
def regression(beta_0, beta_1, dataset=get_dataset()):
    MSE = 0
    for x, y in dataset:
        MSE += (beta_0 + beta_1*x - y)**2
    MSE /= len(dataset)
    
    return MSE


# Performs a single step of gradient descent on the MSE and returns the 
# derivative values as a tuple.
#
# beta_0 - first fixed beta
# beta_1 - second fixed beta
def gradient_descent(beta_0, beta_1, dataset=get_dataset()):
    vec1 = 0
    for x, y in dataset:
        vec1 += beta_0 + beta_1*x - y
    vec1 *= (2 / len(dataset))
    
    vec2 = 0
    for x, y in dataset:
        vec2 += (beta_0 + beta_1*x - y) * x
    vec2 *= (2 / len(dataset))
    
    return (vec1, vec2)


# Performs T iterations of gradient descent starting at (beta_0, beta_1) = 
# (0,0) with the given parameter and prints the results; does not return 
# anything.
#
# T - number of iterations of gradient descent to perform
# eta - multiplier used for gradient descent
def iterate_gradient(T, eta, dataset=get_dataset()):
    beta_0 = 0
    beta_1 = 0
    
    for i in range(1, T+1):
        vec1, vec2 = gradient_descent(beta_0, beta_1, dataset)
        beta_0 -= eta * vec1
        beta_1 -= eta * vec2
        MSE = regression(beta_0, beta_1, dataset)
        
        print('{} {:.2f} {:.2f} {:.2f}'.format(i, beta_0, beta_1, MSE))


# Using the closed-form solution, calculates and returns the values of beta_0 
# and beta_1 and the corresponding MSE as a three-element tuple.
def compute_betas():
    dataset = get_dataset()
    x_mean = statistics.mean([dataset[i][0] for i in range(len(dataset))])
    y_mean = statistics.mean([dataset[i][1] for i in range(len(dataset))])
    
    beta_1 = 0
    numerator = 0
    denominator = 0
    for x, y in dataset:
        numerator += (x - x_mean) * (y - y_mean)
        denominator += (x - x_mean)**2
    beta_1 = numerator / denominator
    
    beta_0 = y_mean - beta_1*x_mean
    MSE = regression(beta_0, beta_1)
    
    return (beta_0, beta_1, MSE)


# Using the closed-form solution betas, return the predicted number of ice 
# days for that year.
#
# year - the year for which the number of ice days is predicted
def predict(year):
    beta_0, beta_1, MSE = compute_betas()
    y_predicted = beta_0 + beta_1*year
    
    return y_predicted


# Normalizes the data before performing gradient descent, prints results as in 
# iterate_gradient().
#
# T - number of iterations of gradient descent to perform
# eta - multiplier used for gradient descent
def iterate_normalized(T, eta):
    dataset = get_dataset()
    x_data = [dataset[i][0] for i in range(len(dataset))]
    x_mean = statistics.mean(x_data)
    x_std = statistics.stdev(x_data)
    
    normalized_dataset = [[(dataset[i][0] - x_mean) / x_std, dataset[i][1]] for i in range(len(dataset))]
    iterate_gradient(T, eta, dataset=normalized_dataset)


# Performs stochastic gradient descent, prints results as in function 
# iterate_gradient().
#
# T - number of iterations of gradient descent to perform
# eta - multiplier used for gradient descent
def sgd(T, eta):
    dataset = get_dataset()
    x_data = [dataset[i][0] for i in range(len(dataset))]
    x_mean = statistics.mean(x_data)
    x_std = statistics.stdev(x_data)
    
    normalized_dataset = [[(dataset[i][0] - x_mean) / x_std, dataset[i][1]] for i in range(len(dataset))]
    
    beta_0 = 0
    beta_1 = 0
    
    for i in range(1, T+1):
        xy_j = random.choice(normalized_dataset)
        vec1 = 2 * (beta_0 + beta_1*xy_j[0] - xy_j[1])
        vec2 = vec1 * xy_j[0]
        beta_0 -= eta * vec1
        beta_1 -= eta * vec2
        MSE = regression(beta_0, beta_1, dataset=normalized_dataset)
        
        print('{} {:.2f} {:.2f} {:.2f}'.format(i, beta_0, beta_1, MSE))


def main():
    #dataset = get_dataset()
    #print_stats(dataset)
    #MSE = regression(200,-.2)
    #gd = gradient_descent(200,-.2)
    #iterate_gradient(10000, 1e-9)
    #betas = compute_betas()
    #y_predicted = predict(2463)
    #iterate_normalized(5,0.01)
    random.seed(0)
    sgd(5,0.1)
    
    #print(dataset)
    #print(MSE)
    #print(gd)
    #print(betas)
    #print(y_predicted)


if __name__ == '__main__':
    main()