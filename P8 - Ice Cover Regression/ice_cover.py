# Author: Bryce Van Camp
# Project: P8
# File: ice_cover.py


import statistics
import random


# Takes no arguments and returns the data as described in the specifications 
# in an n-by-2 array.
def get_dataset():
    dataset = []
    
    dataset.append([1855, 118])
    dataset.append([1856, 151])
    dataset.append([1857, 121])
    dataset.append([1858, 96])
    dataset.append([1859, 110])
    dataset.append([1860, 117])
    dataset.append([1861, 132])
    dataset.append([1862, 104])
    dataset.append([1863, 125])
    dataset.append([1864, 118])
    dataset.append([1865, 125])
    dataset.append([1866, 123])
    dataset.append([1867, 110])
    dataset.append([1868, 127])
    dataset.append([1869, 131])
    dataset.append([1870, 99])
    dataset.append([1871, 126])
    dataset.append([1872, 144])
    dataset.append([1873, 136])
    dataset.append([1874, 126])
    dataset.append([1875, 91])
    dataset.append([1876, 130])
    dataset.append([1877, 62])
    dataset.append([1878, 112])
    dataset.append([1879, 99])
    dataset.append([1880, 161])
    dataset.append([1881, 78])
    dataset.append([1882, 124])
    dataset.append([1883, 119])
    dataset.append([1884, 124])
    dataset.append([1885, 128])
    dataset.append([1886, 131])
    dataset.append([1887, 113])
    dataset.append([1888, 88])
    dataset.append([1889, 75])
    dataset.append([1890, 111])
    dataset.append([1891, 97])
    dataset.append([1892, 112])
    dataset.append([1893, 101])
    dataset.append([1894, 101])
    dataset.append([1895, 91])
    dataset.append([1896, 110])
    dataset.append([1897, 100])
    dataset.append([1898, 130])
    dataset.append([1899, 111])
    dataset.append([1900, 107])
    dataset.append([1901, 105])
    dataset.append([1902, 89])
    dataset.append([1903, 126])
    dataset.append([1904, 108])
    dataset.append([1905, 97])
    dataset.append([1906, 94])
    dataset.append([1907, 83])
    dataset.append([1908, 106])
    dataset.append([1909, 98])
    dataset.append([1910, 101])
    dataset.append([1911, 108])
    dataset.append([1912, 99])
    dataset.append([1913, 88])
    dataset.append([1914, 115])
    dataset.append([1915, 102])
    dataset.append([1916, 116])
    dataset.append([1917, 115])
    dataset.append([1918, 82])
    dataset.append([1919, 110])
    dataset.append([1920, 81])
    dataset.append([1921, 96])
    dataset.append([1922, 125])
    dataset.append([1923, 104])
    dataset.append([1924, 105])
    dataset.append([1925, 124])
    dataset.append([1926, 103])
    dataset.append([1927, 106])
    dataset.append([1928, 96])
    dataset.append([1929, 107])
    dataset.append([1930, 98])
    dataset.append([1931, 65])
    dataset.append([1932, 115])
    dataset.append([1933, 91])
    dataset.append([1934, 94])
    dataset.append([1935, 101])
    dataset.append([1936, 121])
    dataset.append([1937, 105])
    dataset.append([1938, 97])
    dataset.append([1939, 105])
    dataset.append([1940, 96])
    dataset.append([1941, 82])
    dataset.append([1942, 116])
    dataset.append([1943, 114])
    dataset.append([1944, 92])
    dataset.append([1945, 98])
    dataset.append([1946, 101])
    dataset.append([1947, 104])
    dataset.append([1948, 96])
    dataset.append([1949, 109])
    dataset.append([1950, 122])
    dataset.append([1951, 114])
    dataset.append([1952, 81])
    dataset.append([1953, 85])
    dataset.append([1954, 92])
    dataset.append([1955, 114])
    dataset.append([1956, 111])
    dataset.append([1957, 95])
    dataset.append([1958, 126])
    dataset.append([1959, 105])
    dataset.append([1960, 108])
    dataset.append([1961, 117])
    dataset.append([1962, 112])
    dataset.append([1963, 113])
    dataset.append([1964, 120])
    dataset.append([1965, 65])
    dataset.append([1966, 98])
    dataset.append([1967, 91])
    dataset.append([1968, 108])
    dataset.append([1969, 113])
    dataset.append([1970, 110])
    dataset.append([1971, 105])
    dataset.append([1972, 97])
    dataset.append([1973, 105])
    dataset.append([1974, 107])
    dataset.append([1975, 88])
    dataset.append([1976, 115])
    dataset.append([1977, 123])
    dataset.append([1978, 118])
    dataset.append([1979, 99])
    dataset.append([1980, 93])
    dataset.append([1981, 96])
    dataset.append([1982, 54])
    dataset.append([1983, 111])
    dataset.append([1984, 85])
    dataset.append([1985, 107])
    dataset.append([1986, 89])
    dataset.append([1987, 87])
    dataset.append([1988, 97])
    dataset.append([1989, 93])
    dataset.append([1990, 88])
    dataset.append([1991, 99])
    dataset.append([1992, 108])
    dataset.append([1993, 94])
    dataset.append([1994, 74])
    dataset.append([1995, 119])
    dataset.append([1996, 102])
    dataset.append([1997, 47])
    dataset.append([1998, 82])
    dataset.append([1999, 53])
    dataset.append([2000, 115])
    dataset.append([2001, 21])
    dataset.append([2002, 89])
    dataset.append([2003, 80])
    dataset.append([2004, 101])
    dataset.append([2005, 95])
    dataset.append([2006, 66])
    dataset.append([2007, 106])
    dataset.append([2008, 97])
    dataset.append([2009, 87])
    dataset.append([2010, 109])
    dataset.append([2011, 57])
    dataset.append([2012, 87])
    dataset.append([2013, 117])
    dataset.append([2014, 91])
    dataset.append([2015, 62])
    dataset.append([2016, 65])
    dataset.append([2017, 94])
    dataset.append([2018, 86])
    dataset.append([2019, 70])
    
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
    #iterate_gradient(5, 1e-8)
    #betas = compute_betas()
    #y_predicted = predict(2021)
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