# Author: Bryce Van Camp
# Project: p7
# File: ten_hundred.py


import csv
import math
import numpy


# Takes in a string with a path to an appropriately formatted CSV file and 
# returns the data (without the lat/long columns but retaining all other 
# columns) in a single structure.
#
# filepath - string path to the CSV file
def load_data(filepath):
    data = []
    with open(filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            del row['Lat']
            del row['Long']
            data.append(row)
    
    return data


# Takes in one row from the data loaded from the above function, calculates 
# the corresponding x, y values for that region as specified in the video, 
# and returns them in a single structure.
#
# x refers to the n/10 day, and y is the n/100 day.
#
# time_series - case data for a particular region
def calculate_x_y(time_series):
    D = time_series.copy()
    
    # remove non-case data
    del D['Province/State']
    del D['Country/Region']
    
    x = math.nan
    y = math.nan
    
    if len(D) == 0:
        return x, y
    
    # get cases for current day
    last_key = next(reversed(D))
    cur_cases = int(D[last_key])
    
    # exclude rows with 0 cases on the current day
    if cur_cases == 0:
        return x, y
    
    cur_day = len(D) - 1
    
    # find x
    n10 = cur_cases / 10
    n10_day = 0
    i_day = 0
    for k, v in D.items():
        v = int(v)
        if v <= n10 and i_day >= n10_day:
            n10_day = i_day
            x = cur_day - i_day
        
        i_day += 1
    
    # find y
    n100 = cur_cases / 100
    n100_day = 0
    i_day = 0
    for k, v in D.items():
        v = int(v)
        if v <= n100 and i_day >= n100_day:
            n100_day = i_day
            y = n10_day - n100_day
        
        i_day += 1
    
    return x, y


# Returns the euclidean distance of the closest (x, y) tuples in a pair of 
# clusters.
#
# A - first cluster
# B - second cluster
def closest_distance(A, B):
    best_dist = math.inf
    for x1, y1 in A:
        for x2, y2 in B:
            cur_dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            best_dist = min(best_dist, cur_dist)
    
    return best_dist


# Performs single linkage hierarchical agglomerative clustering on the 
# regions with the (x,y) feature representation and returns a data structure 
# representing the clustering.
#
# dataset - (x,y) feature representation of each region
def hac(dataset):
    # filter out invalid rows
    S = [(x, y) for x, y in dataset if math.isfinite(x) and math.isfinite(y)]
    
    # (m-1) by 4 matrix
    Z = []
    
    # Step 1 of HAC - Place each instance in its own singleton cluster.
    D = {i:[S[i]] for i in range(len(S))}
    
    # Step 2 - While (number of clusters > 1) do:
    m = len(D)
    i = 0
    while len(D) > 1:
        # Step 3 - Find the closest cluster pair A, B.
        A_index = math.inf
        B_index = math.inf
        best_dist = math.inf
        for j in D.keys():
            for k in D.keys():
                if j == k:
                    continue
                
                # find distance of the two closest (x, y) tuples in the pair 
                # of clusters
                cur_dist = closest_distance(D[j], D[k])
                
                win_tiebreak = cur_dist == best_dist and (min(j, k) < A_index or (min(j, k) == A_index and max(j, k) < B_index))
                
                # pick closest cluster pair or the tiebreak winner
                if cur_dist < best_dist or win_tiebreak:
                    A_index = min(j, k)
                    B_index = max(j, k)
                    best_dist = cur_dist
        
        # Step 4 - Merge A and B to form a new cluster.
        Z.append([A_index, B_index, best_dist, len(D[A_index]) + len(D[B_index])])
        
        # maintain dictionary
        D[m+i] = D[A_index] + D[B_index]
        del D[A_index]
        del D[B_index]
        
        i += 1
        
    return numpy.asmatrix(Z)


def main():
    data = load_data('time_series_covid19_confirmed_global.csv')
    all_xy = [calculate_x_y(D) for D in data]
    Z = hac(all_xy)
    print(Z)


if __name__ == "__main__":
    main()