# Author: Bryce Van Camp (with skeleton code from assignment page)
# Project: p6
# File: teeko_player.py

import random
import copy as cp
import math
import time

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]
    

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                raise Exception("You don't have a piece there!")
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)
        
    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece
        
        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
                
                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece
        
    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")
        
    def game_value(self, state):
        """ Checks the current board status for a win condition
        
        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # check \ diagonal wins
        if state[1][0] != ' ' and state[1][0] == state[2][1] == state[3][2] == state[4][3]:
            return 1 if state[1][0] == self.my_piece else -1
        if state[0][1] != ' ' and state[0][1] == state[1][2] == state[2][3] == state[3][4]:
            return 1 if state[0][1] == self.my_piece else -1
        if state[0][0] != ' ' and state[0][0] == state[1][1] == state[2][2] == state[3][3]:
            return 1 if state[0][0] == self.my_piece else -1
        if state[1][1] != ' ' and state[1][1] == state[2][2] == state[3][3] == state[4][4]:
            return 1 if state[1][1] == self.my_piece else -1
        
        # check / diagonal wins
        if state[0][3] != ' ' and state[0][3] == state[1][2] == state[2][1] == state[3][0]:
            return 1 if state[0][3] == self.my_piece else -1
        if state[1][4] != ' ' and state[1][4] == state[2][3] == state[3][2] == state[4][1]:
            return 1 if state[1][4] == self.my_piece else -1
        if state[0][4] != ' ' and state[0][4] == state[1][3] == state[2][2] == state[3][1]:
            return 1 if state[0][4] == self.my_piece else -1
        if state[1][3] != ' ' and state[1][3] == state[2][2] == state[3][1] == state[4][0]:
            return 1 if state[1][3] == self.my_piece else -1
        
        # check 2x2 box wins
        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i][j] != ' ':
                    # top-right
                    if i < len(state) - 1 and j > 0 and state[i][j] == state[i][j-1] == state[i+1][j-1] == state[i+1][j]:
                        return 1 if state[i][j] == self.my_piece else -1
                    
                    # top-left
                    if i < len(state) - 1 and j < len(state[i]) - 1 and state[i][j] == state[i+1][j] == state[i+1][j+1] == state[i][j+1]:
                        return 1 if state[i][j] == self.my_piece else -1
                        
                    # bottom-left
                    if i > 0 and j < len(state[i]) - 1 and state[i][j] == state[i][j+1] == state[i-1][j+1] == state[i-1][j]:
                        return 1 if state[i][j] == self.my_piece else -1
                        
                    # bottom-right
                    if i > 0 and j > 0 and state[i][j] == state[i-1][j] == state[i-1][j-1] == state[i][j-1]:
                        return 1 if state[i][j] == self.my_piece else -1
        
        return 0 # no winner
    
    
    # Returns the distance from the player's current piece to the rest of the 
    # the player's pieces.
    #
    # state - the current board state
    # cur_piece - current player's game piece
    # coord - coordinates of the player's current piece
    def get_dist(self, state, cur_piece, coord):
        dist = 0
        
        for i in range(5):
            for j in range(5):
                if state[i][j] == cur_piece and (i, j) != coord:
                    dist += abs(i - coord[0]) + abs(j - coord[1])
        
        return dist
    
    
    # Returns the coordinates of all the pieces of the current player.
    # Return value is a list of tuple coordinates.
    #
    # state - the current board state
    # cur_piece - current player's game piece
    def get_coords(self, state, cur_piece):
        coords = []
        
        for i in range(5):
            for j in range(5):
                if state[i][j] == cur_piece:
                    coords.append((i, j))
        
        return coords
    
    
    # This heuristic function determines a heuristic based on the distance 
    # of each of the player's pieces from each other.
    #
    # state - the current board state
    # cur_piece - current player's game piece
    def heuristic_game_value(self, state, cur_piece):
        # if in a terminal state
        game_val = self.game_value(state)
        if game_val != 0:
            return game_val
        
        # get coordinates of the player's pieces
        coords = self.get_coords(state, cur_piece)
        
        total_dist = 0
        for cur_coord in coords:
            total_dist += self.get_dist(state, cur_piece, cur_coord)
        
        # heuristic is higher the closer the player's pieces are from each other
        h = 1 - total_dist / (len(state) * len(state[0]) * len(coords))
        
        # make negative if current player is the opponent
        if cur_piece == self.opp:
            h = -h
        
        return h
    
    
    # Takes in a board state and returns a list of the legal successors. 
    # 
    # Drop phase: Add a new piece of the current player's type to the board
    # Continued gameplay: Move piece to an unoccupied, adjacent location.
    #
    # state - the current board state
    # drop_phase - true if in drop phase, false otherwise
    # cur_piece - current player's game piece
    def succ(self, state, drop_phase, cur_piece):
        legal_succs = []
        
        if drop_phase:
            for i in range(len(state)):
                for j in range(len(state[i])):
                    if state[i][j] == ' ':
                        temp = cp.deepcopy(state)
                        temp[i][j] = cur_piece
                        legal_succs.append(temp)
        else:
            for i in range(len(state)):
                for j in range(len(state[i])):
                    if state[i][j] == cur_piece:
                        # left
                        if i > 0 and state[i-1][j] == ' ':
                            temp = cp.deepcopy(state)
                            temp[i-1][j] = cur_piece
                            temp[i][j] = ' '
                            legal_succs.append(temp)
                        
                        # right
                        if i < len(state) - 1 and state[i+1][j] == ' ':
                            temp = cp.deepcopy(state)
                            temp[i+1][j] = cur_piece
                            temp[i][j] = ' '
                            legal_succs.append(temp)
                        
                        # up
                        if j > 0 and state[i][j-1] == ' ':
                            temp = cp.deepcopy(state)
                            temp[i][j-1] = cur_piece
                            temp[i][j] = ' '
                            legal_succs.append(temp)
                        
                        # down
                        if j < len(state[i]) - 1 and state[i][j+1] == ' ':
                            temp = cp.deepcopy(state)
                            temp[i][j+1] = cur_piece
                            temp[i][j] = ' '
                            legal_succs.append(temp)
        
        return legal_succs
    
    
    # The Max_Value part of the Minimax algorithm
    #
    # state - the current board state
    # num_pieces - number of pieces currently on the board
    # cur_piece - current player's game piece
    # max_depth - maximum recursion depth allowed
    # cur_depth - the current recursion depth
    def max_value(self, state, num_pieces, cur_piece, max_depth, cur_depth=0):
        if cur_depth == max_depth:
            return self.heuristic_game_value(state, cur_piece), cur_depth
        
        # if in a terminal state
        game_val = self.game_value(state)
        #print('Game val in max_value: {}'.format(game_val))
        if game_val != 0:
            return game_val, cur_depth
        
        alpha = -math.inf, None
        for cur_succ in self.succ(state, num_pieces + cur_depth < 8, cur_piece):
            min_val = self.min_value(cur_succ, num_pieces, self.my_piece if cur_piece == self.opp else self.opp, max_depth, cur_depth=cur_depth+1)
            alpha = max(alpha[0], min_val[0]), min_val[1]
        
        return alpha
        
    
    # The Min_Value part of the Minimax algorithm
    #
    # state - the current board state
    # num_pieces - number of pieces currently on the board
    # cur_piece - current player's game piece
    # max_depth - maximum recursion depth allowed
    # cur_depth - the current recursion depth
    def min_value(self, state, num_pieces, cur_piece, max_depth, cur_depth=0):
        if cur_depth == max_depth:
            return self.heuristic_game_value(state, cur_piece), cur_depth
        
        # if in a terminal state
        game_val = self.game_value(state)
        #print('Game val in min_value: {}'.format(game_val))
        if game_val != 0:
            return game_val, cur_depth
        
        beta = math.inf, None
        for cur_succ in self.succ(state, num_pieces + cur_depth < 8, cur_piece):
            max_val = self.max_value(cur_succ, num_pieces, self.my_piece if cur_piece == self.opp else self.opp, max_depth, cur_depth=cur_depth+1)
            beta = min(beta[0], max_val[0]), max_val[1]
        
        return beta
    
    
    # Takes in a board state and returns a tuple of (list of the legal 
    # successors, move list).
    # 
    # Drop phase: Add a new piece of the current player's type to the board
    # Continued gameplay: Move piece to an unoccupied, adjacent location.
    #
    # state - the current board state
    # drop_phase - true if in drop phase, false otherwise
    # cur_piece - current player's game piece
    def succ_with_move(self, state, drop_phase, cur_piece):
        legal_succs = []
        
        if drop_phase:
            for i in range(len(state)):
                for j in range(len(state[i])):
                    if state[i][j] == ' ':
                        temp = cp.deepcopy(state)
                        temp[i][j] = cur_piece
                        legal_succs.append((temp, [(i, j)]))
        else:
            for i in range(len(state)):
                for j in range(len(state[i])):
                    if state[i][j] == cur_piece:
                        # left
                        if i > 0 and state[i-1][j] == ' ':
                            temp = cp.deepcopy(state)
                            temp[i-1][j] = cur_piece
                            temp[i][j] = ' '
                            legal_succs.append((temp, [(i-1, j), (i, j)]))
                        
                        # right
                        if i < len(state) - 1 and state[i+1][j] == ' ':
                            temp = cp.deepcopy(state)
                            temp[i+1][j] = cur_piece
                            temp[i][j] = ' '
                            legal_succs.append((temp, [(i+1, j), (i, j)]))
                        
                        # up
                        if j > 0 and state[i][j-1] == ' ':
                            temp = cp.deepcopy(state)
                            temp[i][j-1] = cur_piece
                            temp[i][j] = ' '
                            legal_succs.append((temp, [(i, j-1), (i, j)]))
                        
                        # down
                        if j < len(state[i]) - 1 and state[i][j+1] == ' ':
                            temp = cp.deepcopy(state)
                            temp[i][j+1] = cur_piece
                            temp[i][j] = ' '
                            legal_succs.append((temp, [(i, j+1), (i, j)]))
        
        return legal_succs
    
    
    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.
            
        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.
                
                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).
        
        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        
        #time_amt = time.time()
        
        # find number of pieces on the board
        num_pieces = 0
        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i][j] != ' ':
                    num_pieces += 1
        
        # get all possible successors with their corresponding moves
        move_succs = self.succ_with_move(state, num_pieces < 8, self.my_piece)
        #for cur_succ, cur_move in move_succs:
        #    print('{}, {}'.format(cur_succ, cur_move))
        
        # hyper-optimized max_depth
        max_depth = 2 if len(move_succs) > 18 else max(3, 7 - len(move_succs))
        
        # find the best move to make
        best_move = None
        best_val = -math.inf
        best_depth = math.inf
        for cur_succ, cur_move in move_succs:
            minimax_val, cur_depth = self.min_value(cur_succ, num_pieces+1, self.opp, max_depth)
            #print('{}, {}'.format(minimax_val, cur_depth))
            if minimax_val > best_val or (minimax_val == best_val and cur_depth < best_depth):
                best_val = minimax_val
                best_depth = cur_depth
                best_move = cur_move
        
        #print('{}, {}, {}'.format(best_move, best_val, best_depth))
        
        #time_amt = time.time() - time_amt
        #print(time_amt)
        
        return best_move
        

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################

def main():
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                     (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
