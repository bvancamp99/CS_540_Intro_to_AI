# Author: Bryce Van Camp
# Project: P2
# File: torus_puzzle.py

import math

''' author: hobbes
    source: cs540 canvas
'''
class PriorityQueue(object):
    def __init__(self):
        self.queue = []
        self.max_len = 0

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    def is_empty(self):
        return len(self.queue) == 0

    def enqueue(self, state_dict):
        """ Items in the priority queue are dictionaries:
             -  'state': the current state of the puzzle
             -      'h': the heuristic value for this state
             - 'parent': a reference to the item containing the parent state
             -      'g': the number of moves to get from the initial state to
                         this state, the "cost" of this state
             -      'f': the total estimated cost of this state, g(n)+h(n)

            For example, an item in the queue might look like this:
             {'state':[1,2,3,4,5,6,7,8,0], 'parent':[1,2,3,4,5,6,7,0,8],
              'h':0, 'g':14, 'f':14}

            Please be careful to use these keys exactly so we can test your
            queue, and so that the pop() method will work correctly.
        """
        # store 2D list containing all states
        all_states = [cur_dict['state'] for cur_dict in self.queue]
        
        # set in_open to True if the state is in the queue already
        in_open = True if state_dict['state'] in all_states else False
        
        # handle that case correctly
        if in_open:
            # store dict in queue with the same state as state_dict
            dict_index = all_states.index(state_dict['state'])
            
            # update cost to lower of the two options
            if state_dict['g'] < self.queue[dict_index]['g']:
                self.queue[dict_index] = state_dict
        else:
            self.queue.append(state_dict)

        # track the maximum queue length
        if len(self.queue) > self.max_len:
            self.max_len = len(self.queue)

    def requeue(self, from_closed):
        """ Re-queue a dictionary from the closed list (see lecture slide 21)
        """
        self.queue.append(from_closed)

        # track the maximum queue length
        if len(self.queue) > self.max_len:
            self.max_len = len(self.queue)

    def pop(self):
        """ Remove and return the dictionary with the smallest f(n)=g(n)+h(n)
        """
        minf = 0
        for i in range(1, len(self.queue)):
            if self.queue[i]['f'] < self.queue[minf]['f']:
                minf = i
        state = self.queue[minf]
        del self.queue[minf]
        return state

# Returns a copy of the list parameter that swaps the specified
# indexes denoted by i and j.
#
# old_list - parameter list that is copied
# i - first index to be swapped
# j - second index to be swapped
def swapped(old_list, i, j):
    # make copy of list
    new_list = old_list.copy()
    
    # swap specified indexes
    new_list[i], new_list[j] = new_list[j], new_list[i]
    
    # return new list with swapped indexes
    return new_list

# Returns the heuristic of the given state.  The heuristic is
# described as "the count of tiles which are not in their goal 
# spaces."
#
# state - a single 8-puzzle state
# N - length of state (should always be 9)
def get_heuristic(state, N):
    # init return value
    heuristic = 0
    
    # increment number of tiles in wrong place for 1-8
    for i in range(N - 1):
        if state[i] != i + 1:
            heuristic += 1
            
    # blank space doesn't count toward heuristic
        
    # return heuristic for the state
    return heuristic

# Returns the index of the blank space's neighbor.  The neighbor is 
# specified by row and col.
#
# width - width of a would-be matrix representation of the list
# row - row of the neighbor
# col = col of the neighbor
def nearby_index(width, row, col):
    return row * width + col

# Returns all 4 possible successors of a given state.
#
# state - a single 8-puzzle state
# N - length of state (should always be 9)
def get_successors(state, N):
    WIDTH = int(math.sqrt(N)); # width of state (should always be 3)
    
    i_bl = state.index(0) # index of the blank space
    row_bl = i_bl // WIDTH # row of blank space in a would-be matrix
    col_bl = i_bl % WIDTH # col of blank space in a would-be matrix
    
    i_neighbor = 0 # index of the specified neighbor of blank space
    cur_succ = None # current successor
    
    successors = [] # 2D list that will store all 4 successors
    
    # append successor that swaps 0 with its left neighbor
    i_neighbor = nearby_index(WIDTH, row_bl, (col_bl - 1) % WIDTH)
    cur_succ = swapped(state, i_bl, i_neighbor)
    successors.append(cur_succ)
    
    # append successor that swaps 0 with its right neighbor
    i_neighbor = nearby_index(WIDTH, row_bl, (col_bl + 1) % WIDTH)
    cur_succ = swapped(state, i_bl, i_neighbor)
    successors.append(cur_succ)
    
    # append successor that swaps 0 with its above neighbor
    i_neighbor = nearby_index(WIDTH, (row_bl - 1) % WIDTH, col_bl)
    cur_succ = swapped(state, i_bl, i_neighbor)
    successors.append(cur_succ)
    
    # append successor that swaps 0 with its below neighbor
    i_neighbor = nearby_index(WIDTH, (row_bl + 1) % WIDTH, col_bl)
    cur_succ = swapped(state, i_bl, i_neighbor)
    successors.append(cur_succ)
    
    return successors

# Prints all possible successor states given the parameter state.
#
# state - a single 8-puzzle state
def print_succ(state):
    # store length of state (should always be 9)
    N = len(state)
    
    # get all 4 successors and sort
    successors = get_successors(state, N)
    successors.sort()
    
    # print each successor, along with its heuristic
    for succ in successors:
        h = get_heuristic(succ, N)
        print('{} h={}'.format(succ, h))

# Performs the A* search algorithm and prints the path from the 
# current state to the goal state.
#
# state - a single 8-puzzle state
def solve(state):
    # store length of state (should always be 9)
    N = len(state)
    
    # init the priority queue
    pq = PriorityQueue()
    
    # create initial state_dict
    g = 0
    h = get_heuristic(state, N)
    state_dict = {
        'state': state,
        'h': h,
        'g': g,
        'parent': None,
        'f': g + h
    };
    
    # A* search until goal state found
    while (state_dict['h'] > 0):
        # get all 4 successors of the current state
        successors = get_successors(state_dict['state'], N)
        
        # add each successor to pq
        g = state_dict['g'] + 1
        for succ in successors:
            # create state dict for successor
            h = get_heuristic(succ, N)
            succ_dict = {
                'state': succ,
                'h': h,
                'g': g,
                'parent': state_dict,
                'f': g + h
            };
    
            # add state_dict to pq
            pq.enqueue(succ_dict)
        
        # update state_dict
        state_dict = pq.pop()
    
    # goal state found!
    
    # store list of path strings
    path = []
    
    while state_dict['parent'] != None:
        # append state info to the path list
        path.append('{}  h={}  moves: {}'.format(state_dict['state'], state_dict['h'], state_dict['g']))
        
        # update state_dict
        state_dict = state_dict['parent']
        
    # add last state info to the path list
    path.append('{}  h={}  moves: {}'.format(state_dict['state'], state_dict['h'], state_dict['g']))
        
    # print in correct order
    for state_str in reversed(path):
        print(state_str)
        
    # print max queue length
    print('Max queue length: {}'.format(pq.max_len))
