# Author: Bryce Van Camp
# Project: P1
# File: p1_statespace.py

# Fills the jug specified by which
#
# state - current state of the jugs
# max - max size state
# which - index that identifies which jug to use
def fill(state, max, which):
	# store copy of state that will be returned
	new_state = state.copy()
	
	# fill the specified jug to its max capacity
	new_state[which] = max[which]
	
	return new_state

# Empties the jug specified by which
#
# state - current state of the jugs
# max - max size state
# which - index that identifies which jug to use
def empty(state, max, which):
	# store copy of state that will be returned
	new_state = state.copy()
	
	# empty the specified jug
	new_state[which] = 0
	
	return new_state

# Transfers from source to dest jug
#
# state - current state of the jugs
# max - max size state
# source - source jug
# dest - destination jug
def xfer(state, max, source, dest):
	# store copy of state that will be returned
	new_state = state.copy()
	
	# compute which amount should be transferred
	max_empty = new_state[source]
	max_fill = max[dest] - new_state[dest]
	xfer_amt = min(max_empty, max_fill)
	
	# transfer water
	new_state[source] -= xfer_amt
	new_state[dest] += xfer_amt
	
	return new_state

# Displays unique successors given current state
#
# state - current state of the jugs
# max - max size state
def succ(state, max):
	# init list to store successors computed
	successors = []
	
	# will store the current successor state
	cur_state = []
	
	# fill jug 0
	cur_state = fill(state, max, 0)
	print(str(cur_state))
	successors.append(fill(state, max, 0))
	
	# fill jug 1
	cur_state = fill(state, max, 1)
	if cur_state not in successors:
		print(str(cur_state))
		successors.append(cur_state)
	
	# empty jug 0
	cur_state = empty(state, max, 0)
	if cur_state not in successors:
		print(str(cur_state))
		successors.append(cur_state)
	
	# empty jug 1
	cur_state = empty(state, max, 1)
	if cur_state not in successors:
		print(str(cur_state))
		successors.append(cur_state)
	
	# xfer jug 0 -> jug 1
	cur_state = xfer(state, max, 0, 1)
	if cur_state not in successors:
		print(str(cur_state))
		successors.append(cur_state)
	
	# xfer jug 1 -> jug 0
	cur_state = xfer(state, max, 1, 0)
	if cur_state not in successors:
		print(str(cur_state))
		successors.append(cur_state)
	