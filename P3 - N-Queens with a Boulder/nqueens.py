# Author: Bryce Van Camp
# Project: P3
# File: nqueens.py

import math
import random

# Returns successor of the state given i and j.
#
# state - given state of the board
# i - current queen column
# j - new queen row
def get_succ(state, i, j):
    # create copy of state
    my_succ = state.copy()
    
    # update copy to be successor
    my_succ[i] = j
    
    return my_succ

# Returns all valid successors of a given state.
#
# state - current state of the board
# boulderX - boulder col
# boulderY - boulder row
def succ(state, boulderX, boulderY):
    # store size of board
    n = len(state)

    # 2D list will store all successor states
    succ_states = []

    # traverse queens by column
    for i in range(n):
        # if queen col = boulder col, we need to ensure no collision
        if i == boulderX:
            for j in range(n):
                # skip identical successor
                if j == state[i]:
                    continue
                
                # skip successor that would collide with the boulder
                if j == boulderY:
                    continue
                    
                # add successor to succ_states
                succ_states.append(get_succ(state, i, j))
                
        # else no need to check for collision
        else:
            for j in range(n):
                # skip identical successor
                if j == state[i]:
                    continue
                
                # add successor to succ_states
                succ_states.append(get_succ(state, i, j))

    return succ_states


# Returns score of the current state, i.e. number of queens being attacked.
#
# state - current state of the board
# boulderX - boulder col
# boulderY - boulder row
def f(state, boulderX, boulderY):
    # store size of board
    n = len(state)

    # create score var
    f = 0

    # traverse queens
    for i in range(n):
        # check if each queen can attack the current
        for j in range(n):
            # skip same queen
            if j == i:
                continue

            # compute bools for row check
            same_row = state[i] == state[j]
            impeded = boulderY == state[i] and min(i, j) < boulderX < max(i, j)

            # if queens on same row and not impeded by boulder
            if same_row and not impeded:
                f += 1
                break

            # compute differences for diagonal check
            col_dif = abs(i - j)
            row_dif = abs(state[i] - state[j])
            col_dif_boulder = i - boulderX
            row_dif_boulder = state[i] - boulderY

            # compute bools for diagonal check
            same_diag = col_dif == row_dif
            impeded = col_dif_boulder == row_dif_boulder and min(i, j) < boulderX < max(i, j) and min(state[i], state[j]) < boulderY < max(state[i], state[j])

            # check for diagonal attack
            if same_diag and not impeded:
                f += 1
                break

    return f


# Chooses the best successor, i.e. lowest score.
#
# If there is a tie, select the "lowest" state from the sorted list
# of successors.
#
# If the state selected is the current state, return None.
#
# curr - current state of the board
# boulderX - boulder col
# boulderY - boulder row
def choose_next(curr, boulderX, boulderY):
    # sort 2D list of all successors from succ()
    succ_states = succ(curr, boulderX, boulderY)

    # add current state and sort
    succ_states.append(curr)
    succ_states.sort()

    # find best successor
    succ_best = None
    f_best = math.inf
    for x in succ_states:
        # store current f
        f_cur = f(x, boulderX, boulderY)

        # update succ_best if its score is the new best
        if f_cur < f_best:
            f_best = f_cur
            succ_best = x

    if succ_best == curr:
        return None
    else:
        return succ_best


# Runs the hill-climbing algorithm from a given initial state and
# returns the convergence state.
#
# curr - initial state of the board
# boulderX - boulder col
# boulderY - boulder row
def nqueens(initial_state, boulderX, boulderY):
    # print the initial state with its score
    f_cur = f(initial_state, boulderX, boulderY)
    print('{} - f={}'.format(initial_state, f_cur))
    
    # perform hill-climbing alg until stuck (or solution is found!)
    cur_state = initial_state
    while f_cur != 0:
        # store cur_state in temp var
        temp = cur_state
        
        # find next successor
        cur_state = choose_next(cur_state, boulderX, boulderY)
        
        # leave if stuck
        if cur_state is None:
            return temp
        
        # print cur_state with its f
        f_cur = f(cur_state, boulderX, boulderY)
        print('{} - f={}'.format(cur_state, f_cur))
        
    # if this point reached, solution was found!
    return cur_state

# Run the hill-climbing algorithm on a randomly generated n*n board 
# with random restarts.
#
# n - board size
# k - num times to restart if no success
# boulderX - boulder col
# boulderY - boulder row
def nqueens_restart(n, k, boulderX, boulderY):
    # create 2D list that will store all failed, best solutions
    solns = []
    
    # perform nqueens until k reached or solution found
    for attempt in range(k):
        # init list that will serve as the current random state
        rand_state = []
        
        # create random (valid!) state
        for i in range(n):
            # declare random row
            j_rand = random.randint(0, n - 1)
            
            # if on same column as boulder
            if i == boulderX:
                # regenerate random row until valid
                while j_rand == boulderY:
                    j_rand = random.randint(0, n - 1)
            
            # set queen at col with row=j_rand
            rand_state.append(j_rand)
            
        # run nqueens on random state
        min_state = nqueens(rand_state, boulderX, boulderY)
        
        # if solution found, print state and terminate
        f_cur = f(min_state, boulderX, boulderY)
        if f_cur == 0:
            print(min_state)
            return
        
        # add to failed solns list
        solns.append(min_state)
            
    # if solution not found, sort and print solns
    solns.sort()
    for x in solns:
        print(x)
    