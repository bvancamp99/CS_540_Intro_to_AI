# Author: Bryce Van Camp
# Project: P4
# Files: envelope_sim.py, classify.py
# File: envelope_sim.py

import random

# This function expects two boolean parameters and returns True or False based
# on whether you selected the correct envelope.
#
# switch - whether you switch envelopes or not
# verbose - whether you want to see the printed explanation of the simulation
def pick_envelope(switch, verbose):
    num_balls = 4
    num_envs = 2
    
    # randomly select location of red ball
    r_loc = random.randrange(num_balls)
    
    # create envelopes
    envelopes = ['b' if i != r_loc else 'r' for i in range(num_balls)]
    envelopes = [envelopes[: num_balls // 2], envelopes[num_balls // 2 :]]
    
    # randomly select one envelope
    cur_env = random.randrange(num_envs)
    
    # randomly select a ball from the envelope
    cur_ball = random.randrange(num_balls // 2)
    
    if verbose:
        print('Envelope 0: {} {}'.format(envelopes[0][0], envelopes[0][1]))
        print('Envelope 1: {} {}'.format(envelopes[1][0], envelopes[1][1]))
        print('I picked envelope {}'.format(cur_env))
        print('and drew a {}'.format(envelopes[cur_env][cur_ball]))
    
    # return true if red ball was picked
    if envelopes[cur_env][cur_ball] == 'r':
        return True
    
    # switch envelopes depending on param
    if switch:
        cur_env = 0 if cur_env is 1 else 1
        
        if verbose:
            print('Switch to envelope {}'.format(cur_env))
    
    # determine whether payout envelope was picked
    for i in range(num_balls // 2):
        if envelopes[cur_env][i] == 'r':
            return True
    
    # if this point reached, not payout envelope
    return False
    

# This function runs n simulations of envelope-picking under both strategies 
# (switch n times, don't switch n times) and prints the percent of times the 
# correct envelope was chosen for each.
#
# n - number of simulations to run
def run_simulation(n):
    # n sims for switch
    switch = 0
    for i in range(n):
        switch += 1 if pick_envelope(switch=True, verbose=False) else 0
    
    # n sims for no-switch
    no_switch = 0
    for i in range(n):
        no_switch += 1 if pick_envelope(switch=False, verbose=False) else 0
    
    print('Switch successful: {} %'.format(switch / n * 100))
    print('No-switch successful: {} %'.format(no_switch / n * 100))