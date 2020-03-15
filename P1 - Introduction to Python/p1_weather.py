# Author: Bryce Van Camp
# Project: P1
# File: p1_weather.py

import math
import operator

# Computes 3D euclidean distance of two points
# 
# data_point1 - first data point dictionary
# data_point2 - second data point dictionary
def euclidean_distance(data_point1, data_point2):
	# precipitation amt will serve as x's
	x1 = data_point1['PRCP']
	x2 = data_point2['PRCP']
	
	# max temp for y's
	y1 = data_point1['TMAX']
	y2 = data_point2['TMAX']
	
	# min temp for z's
	z1 = data_point1['TMIN']
	z2 = data_point2['TMIN']
	
	# return euclidean distance in 3D space
	return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

# Reads dataset file
# 
# filename - name of the dataset file
def read_dataset(filename):
	# store list of data point dictionaries
	dataset = []
	
	# read file and populate the list
	with open(filename, 'r') as f:
		for line in f:
			# split the line by whitespace
			split_line = line.split()
			
			# create data point dict
			dp_dict = {
				'DATE': split_line[0], 
				'PRCP': float(split_line[1]), 
				'TMAX': float(split_line[2]), 
				'TMIN': float(split_line[3]), 
				'RAIN': split_line[4]
			}
			
			# add to list
			dataset.append(dp_dict)
			
	return dataset

# Computes a majority vote on rain forecast
# 
# nearest_neighbors - data point dictionaries to compare
def majority_vote(nearest_neighbors):
	# store counters for TRUE and FALSE
	true_ct = 0
	false_ct = 0
	
	# compute majority vote
	for data_point in nearest_neighbors:
		vote = data_point['RAIN']
		if vote == 'TRUE':
			true_ct += 1
		elif vote == 'FALSE':
			false_ct += 1
	
	# return majority vote result
	if true_ct >= false_ct:
		return 'TRUE'
	else:
		return 'FALSE'

# Computes a majority vote on the k nearest neighbors
# 
# filename - name of dataset file to read
# test_point - base data point to compare with neighbors
# k - specifies the number of nearest neighbors to compare
def k_nearest_neighbors(filename, test_point, k):
	# get list of data point dictionaries
	dataset = read_dataset(filename)
	
	# store list of tuples, with tuples containing euclidean distance and the data point dictionary
	tup_list = []
	
	# populate list with tuples
	for data_point in dataset:
		dist_tuple = euclidean_distance(test_point, data_point), data_point
		tup_list.append(dist_tuple)
	
	# sort list by euclidean distances and keep nearest k
	tup_list.sort(key=operator.itemgetter(0))
	tup_list = tup_list[:k]
	
	# store list of the nearest neighbors
	nearest_neighbors = [x[1] for x in tup_list]
	
	return majority_vote(nearest_neighbors)
