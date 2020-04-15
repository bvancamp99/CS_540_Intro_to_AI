# Author: Bryce Van Camp
# Project: Assists with hard-coding 165 dataset.append() calls.
# File: hard_code_helper.py


def main():
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
    
    with open('hard_code_data.txt', 'w') as f:
        for x, y in dataset:
            f.write('dataset.append([{}, {}])\n'.format(x, y))


if __name__ == '__main__':
    main()