# Author: Bryce Van Camp
# Project: Reads x_data_old.txt, omitting indexes 4-6 of each line, and writes 
#          the modified contents to a new file called x_data.txt.
# File: rm_chars[4-6].py


def main():
    new_contents = []
    with open('x_data_old.txt', 'r') as f:
        for line in f:
            # indexes 4-6 are the last 3 chars in each line
            new_contents.append(line.rstrip()[:-3])
    
    with open('x_data.txt', 'w') as f:
        for year in new_contents:
            f.write('{}\n'.format(year))


if __name__ == '__main__':
    main()