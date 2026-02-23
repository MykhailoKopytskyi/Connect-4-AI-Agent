import random
import math



EXACT = 0
LOWER = 1
UPPER = 2
EXP_BASE=10

class WorthyWindow:
	def __init__(self,coordinates: list[tuple[int,int]] , owners: list[str], window_type: str, winNum:int ):
		self.coordinates = coordinates
		self.owners= owners 
		self.window_type = window_type # Says whether a window is a horizontal (h), vertical (v) or diagonal (d)
		self.owner = self._determine_owner(owners)
		self.winNum = winNum
	
	
	def _determine_owner(self, owners: list[str]) -> str:
		for owner in owners:
			if owner != " ":
				return owner

	def get_empty_square_coordinates(self) -> list[tuple[int,int]]:
		empty_square_coords=[]
		for i in range(self.winNum):
			if(self.owners[i] == " "):
				empty_square_coords.append(self.coordinates[i])
		return empty_square_coords

	
	
	# # cur_player is the player whose turn it is, but he can not make a move since we cut a search at max_depth
	def get_window_score(self,game_board, cur_player: str ):
		counter=len(list(filter(lambda owner: owner != " ", self.owners))) # Since it is a worthy window, it is NON empty and it is owned by only 1 player, thus, all non-empty slots belong to one player
		does_line_win_next_ply = self.does_line_win_next_ply(game_board)

		score = 0

		# If we find that there is a line with winNum in a row, then we win and return + or - infinity
		if(self.owner == "X" and counter == game_board.winNum):
			score=math.inf
		elif(self.owner == "O" and counter == game_board.winNum):
			score = -math.inf

		# If player X would be next to move and he OWNS a (winNum-1) in a row, then if this is playable in the next move, then he wins. Same for player O
		elif(does_line_win_next_ply == True and cur_player == "X" and self.owner == "X"):
			score = math.pow(EXP_BASE, game_board.winNum + 4)
		elif(does_line_win_next_ply == True and cur_player == "O" and self.owner == "O"):
			score = -math.pow(EXP_BASE, game_board.winNum + 4)

		# But if e.g. X would be next to move, but we have a potentially playable  (winNum-1) in a row from player O (if X were NOT to block it in their current move), then we discourage it MORE THAN a not playable (winNum-1) in a row (-math.pow(EXP_BASE, game_board.winNum) < -math.pow(EXP_BASE, game_board.winNum-1))
		elif(does_line_win_next_ply == True and cur_player == "X" and self.owner == "O"):
			score = -math.pow(EXP_BASE, game_board.winNum)
		elif(does_line_win_next_ply == True and cur_player == "O" and self.owner == "X"):
			score = math.pow(EXP_BASE, game_board.winNum)
		
		# For any other case we simply do an exponentiation with respect to a number of spaces occupied by the player in a line in the current window
		elif(self.owner == "X"):
			score = math.pow(EXP_BASE, counter)
		elif(self.owner == "O"):
			score = -math.pow(EXP_BASE, counter)
		
		return score

	


	

	# Returns true if in a given window there is a line of (winNum-1) for ANY single player that can be completed ASSUMING that the owner of the line MOVES NEXT
	def does_line_win_next_ply(self,game_board):
		counter=len(list(filter(lambda owner: owner != " ", self.owners))) # Since it is a worthy window, it is NON empty and it is owned by only 1 player, thus, all non-empty slots belong to that one player
		if(counter != game_board.winNum-1): # As then we simply can not win with the next ply (or we have won already)
			return False
		if(self.window_type == "h" or self.window_type=="d"): # For horizontal and diagonal windows
			empty_space_coord = self.coordinates[self.owners.index(" ")]
			below_empty_space_coord = (empty_space_coord[0]-1, empty_space_coord[1])
			if(below_empty_space_coord[0] < 0 or (below_empty_space_coord[0] >= 0 and game_board.checkSpace(below_empty_space_coord[0], below_empty_space_coord[1]).value != " ")): # If that space exists and it is owned by someone, then current player can win in the next ply
				return True 
			return False
		elif(self.window_type=="v"): # For vertical windows
			top_space_coord = max(self.coordinates, key = lambda c : c[0]) # We find the highest space in the vertical line
			if(game_board.checkSpace(top_space_coord[0],top_space_coord[1]).value == " "): # If the top space is empty, then we can win in next ply
				return True 
			return False
		return False

			
				
				



class Player:

	def __init__(self, name):
		self.name = name
		self.numExpanded = 0 # Use this to track the number of nodes you expand
		self.numPruned = 0 # Use this to track the number of times you prune 



	# The function returns a heuristic estimate of the board, where the higher the value is, the better it is for MAX and the lower it is, the better it is for MIN
	def eval_game_board(self,game_board, cur_player):
		# Generate all windows of size winNum
		windows: list[WorthyWindow] = [] 
		for row in range(game_board.numRows):
			for column in range(game_board.numColumns):
				# Record a right window starting from (row,column) if such exists
				if(column + game_board.winNum <= game_board.numColumns):
					coordinates = []
					owners = []
					for i in range(game_board.winNum):
						coord = (row,column+i)
						coordinates.append(coord)
						owners.append(game_board.checkSpace(coord[0], coord[1]).value)
					if(self.is_worthy_window(owners)):
						windows.append(WorthyWindow(coordinates, owners, "h", game_board.winNum))

				# Record a bottom window starting from (row,column) if such exists
				if(row+game_board.winNum <= game_board.numRows):
					coordinates = []
					owners = []
					for i in range(game_board.winNum):
						coord = (row+i,column)
						coordinates.append(coord)
						owners.append(game_board.checkSpace(coord[0], coord[1]).value)
					if(self.is_worthy_window(owners)):
						windows.append(WorthyWindow(coordinates, owners, "v",  game_board.winNum))	

				# Record a top right window from (row,column) if such exists 
				if((row+1)-game_board.winNum >= 0 and column + game_board.winNum <= game_board.numColumns):
					coordinates = []
					owners = []
					for i in range(game_board.winNum):
						coord = (row-i,column+i)
						coordinates.append(coord)
						owners.append(game_board.checkSpace(coord[0], coord[1]).value)
					if(self.is_worthy_window(owners)):
						windows.append(WorthyWindow(coordinates, owners, "d" , game_board.winNum))	
				
				# Record a bottom right window from (row,column) if such exists 
				if(row+game_board.winNum <= game_board.numRows and column + game_board.winNum <= game_board.numColumns):
					coordinates = []
					owners = []
					for i in range(game_board.winNum):
						coord = (row+i, column+i)
						coordinates.append(coord)
						owners.append(game_board.checkSpace(coord[0], coord[1]).value)
					if(self.is_worthy_window(owners)):
						windows.append(WorthyWindow(coordinates, owners, "d" , game_board.winNum))
				
		total_heuristic_score=0
		opponent_prewin_combination_counter = 0  # Counts opponent playable (winNum-1) threats
		opponent_prewin_combination_empty_coordinates = set()  # Unique empty squares of those threats


		for window in windows:
			window_score = window.get_window_score(game_board, cur_player)
			total_heuristic_score+=window_score
			if(window_score == math.inf or window_score == -math.inf):
				return window_score # As if any window is infinity, then we do not need to count the rest

			#Tracks opponent's playable prewin combinations
			elif window.does_line_win_next_ply(game_board) and window.owner != cur_player:
				opponent_prewin_combination_counter += 1
				opponent_prewin_combination_empty_coordinates.add(window.get_empty_square_coordinates()[0])

			# If opponent has 2+ distinct immediate threats, then we cannot block both
			if (opponent_prewin_combination_counter >= 2 and
				len(opponent_prewin_combination_empty_coordinates) >= 2):
				if cur_player == "X":
					return -math.inf
				else:
					return math.inf
			
		return total_heuristic_score



	# Returns true if only one player occupies (fully or partially) the window and the window is not empty
	def is_worthy_window(self, owners: list[str]) -> bool:
		max_counter = len(list(filter(lambda owner: owner == "X", owners)))
		min_counter = len(list(filter(lambda owner: owner == "O", owners)))
		empty_counter = len(list(filter(lambda owner: owner == " " ,owners)))

		# If there exist squares such that one is owned by MAX and another by MIN, then no player can ever fully complete this line to win, so we skip it. We also skip the lines that are not owned by anyone, as they are worth nothing
		if (max_counter > 0 and min_counter > 0) or empty_counter==len(owners): 
			return False
		return True



	# Let X be a MAX player, i.e. he is trying to maximise the minimax value (us)
	# Let O be a MIN player, i.e. he is trying to minimise the minimax value
	# The function returns a two tuple, the 1st element of which is the minimax value and the 2nd is the index of the best column move
	def _run_minimax(self, game_board, encoded_game_board, encoded_symmetrical_game_board ,cur_player: str, cur_depth:int, max_depth:int, best_move_from_game_state: object ) -> tuple[int,int]:
		chosen_move=-1
		# The default best move is any valid move
		for column_idx in range(game_board.numColumns):
			if(game_board.checkSpace(game_board.numRows-1, column_idx ).value == " "):
				chosen_move=column_idx
				break

		max_score= -math.inf # The worst possible scenario for a MAX player X
		min_score = math.inf # The worst possible scenario for a MIN player O
		last_player: str = game_board.lastPlay[2] # The name of the last player - either 'X' or 'O'
		last_column_move: int = game_board.lastPlay[1] # The index of the column where we moved to
		if game_board.checkWin():
			if last_player == "X":
				return (math.inf, last_column_move)
			else:
				return (-math.inf, last_column_move)
		elif game_board.checkFull():
			return (0, last_column_move)  # If no one won AND the board is full, then it is a draw 

		if (cur_depth >= max_depth):
			return (self.eval_game_board(game_board,cur_player), last_column_move) # If we have reached the final depth, then we evaluate the current state and return the results


		# If we previously computed the score and best move for this game board, then return the stored results
		# encoded_game_board: tuple[int,int] = encode_game_board(game_board)
		if((encoded_game_board, cur_player) in best_move_from_game_state):
			return best_move_from_game_state[(encoded_game_board, cur_player)]
		
		# # If we previously computed the score and best move for the symmetrical game board, then we return the symmetrical column
		# symmetrical_game_board: tuple[int,int] = get_encoded_symmetrical_board(encoded_game_board, game_board.numRows, game_board.numColumns)
		if((encoded_symmetrical_game_board, cur_player) in best_move_from_game_state):
			expected_final_state =  best_move_from_game_state[(encoded_symmetrical_game_board, cur_player)]
			symmetrical_column = (game_board.numColumns-1) - expected_final_state[1] # Compute the symmetrical column
			return (expected_final_state[0],symmetrical_column)

		states = [] # Temporarily holds all the final states
		self.numExpanded+=1 # We expand the node and then check all of its children 
		for column in range(game_board.numColumns):
			if(game_board.checkSpace(game_board.numRows-1, column).value != " "): # If the column is full, then we try a different one
				continue

			game_board.addPiece(column, cur_player)
			encoded_game_board = add_piece_to_encoded_board(encoded_game_board, game_board.numRows, game_board.numColumns, column, cur_player)
			encoded_symmetrical_game_board = add_piece_to_encoded_board(encoded_symmetrical_game_board,  game_board.numRows, game_board.numColumns, (game_board.numColumns-1-column) , cur_player)


			if (cur_player == "X"):
				expected_final_state = self._run_minimax(game_board, encoded_game_board, encoded_symmetrical_game_board, "O", cur_depth+1, max_depth, best_move_from_game_state) # If the current player is X (i.e. MAX player), then the next move is O's move
				
				game_board.removePiece(column)
				encoded_game_board = remove_piece_from_encoded_board(encoded_game_board, game_board.numRows, game_board.numColumns, column)
				encoded_symmetrical_game_board = remove_piece_from_encoded_board(encoded_symmetrical_game_board, game_board.numRows, game_board.numColumns, (game_board.numColumns-1-column))

				expected_final_score = expected_final_state[0]
				states.append((expected_final_score, column))
				if (expected_final_score > max_score):  # MAX is trying to maximise the minimax value
					max_score = expected_final_score
					chosen_move = column
			else:
				expected_final_state = self._run_minimax(game_board, encoded_game_board, encoded_symmetrical_game_board, "X", cur_depth+1, max_depth, best_move_from_game_state)  # If the current player is O (i.e. MIN player), then the next move is X's move
				
				game_board.removePiece(column)
				encoded_game_board = remove_piece_from_encoded_board(encoded_game_board, game_board.numRows, game_board.numColumns, column)
				encoded_symmetrical_game_board = remove_piece_from_encoded_board(encoded_symmetrical_game_board, game_board.numRows, game_board.numColumns, (game_board.numColumns-1-column))


				expected_final_score = expected_final_state[0]
				states.append((expected_final_score, column))
				if (expected_final_score < min_score): #  MIN is trying to minimise the minimax value.
					min_score = expected_final_score
					chosen_move = column


		# If several states have the same best value, then the column that is closest to the center is chosen
		for state in states:
			score = state[0]
			move=state[1]
			center_col = game_board.numColumns // 2
			if(
				(cur_player == "X" and score == max_score and abs(center_col - move) < abs(center_col - chosen_move))
				or 
				(cur_player == "O" and score == min_score and abs(center_col - move) < abs(center_col - chosen_move))
			):
				chosen_move = move

		if(cur_player == "X"):
			best_value = max_score
		else:
			best_value = min_score

		best_move_from_game_state[(encoded_game_board, cur_player)] = (best_value, chosen_move) # Save the move in the TT
		return (best_value, chosen_move)



	def getMove(self, game_board):
		isEmpty=True
		best_move_from_game_state = {}

		for i in range(game_board.numColumns):
			if(game_board.checkSpace(0,i).value != " "):
				isEmpty=False
		# If it is our 1st move, then it is always the best strategy to go in the middle column
		if (isEmpty == True): 
			return math.floor(game_board.numColumns/2) 

		max_depth=-1
		if(game_board.numColumns >= 26):
			max_depth = 1
		elif(game_board.numColumns >= 13):
			max_depth=2
		elif(game_board.numColumns >= 8): 
			max_depth = 3
		elif(game_board.numColumns >= 5):
			max_depth = 4
		else:
			max_depth = 5


		encoded_game_board = encode_game_board(game_board)
		encoded_symmetrical_game_board: tuple[int,int]  = get_encoded_symmetrical_board(encoded_game_board, game_board.numRows, game_board.numColumns)
		best_expected_state = self._run_minimax(game_board, encoded_game_board, encoded_symmetrical_game_board, self.name, 0, max_depth, best_move_from_game_state)
		chosen_move = best_expected_state[1]
		return chosen_move



	


	def _run_minimax_alpha_beta(self, game_board, encoded_game_board, encoded_symmetrical_game_board ,cur_player: str, alpha: int, beta: int, cur_depth:int, max_depth:int, best_move_from_game_state: object, previous_iteration_best_move_from_game_state: object ) -> tuple[int,int]:
		chosen_move=-1
		# The default best move is any valid move
		for column_idx in range(game_board.numColumns):
			if(game_board.checkSpace(game_board.numRows-1, column_idx ).value == " "):
				chosen_move=column_idx
				break

		max_score= -math.inf # The worst possible scenario for a MAX player X
		min_score = math.inf # The worst possible scenario for a MIN player O
		last_player: str = game_board.lastPlay[2] # The name of the last player - either 'X' or 'O'
		last_column_move: int = game_board.lastPlay[1] # The index of the column where we moved to
		
		# Terminal checks
		if game_board.checkWin():
			if last_player == "X":
				return (math.inf, last_column_move)
			else:
				return (-math.inf, last_column_move)
		elif game_board.checkFull():
			return (0, last_column_move)  # If no one won AND the board is full, then it is a draw 
		if (cur_depth >= max_depth):
			return (self.eval_game_board(game_board,cur_player), last_column_move) # If we have reached the final depth, then we evaluate the current state and return the results

		alpha_orig = alpha
		beta_orig = beta
		key = (encoded_game_board, cur_player)

		# We check if the current encoded board is already in the table
		if key in best_move_from_game_state:
			t_val, t_move, t_flag = best_move_from_game_state[key]
			if (t_flag == EXACT):
				return (t_val, t_move)
			elif (t_flag == LOWER):
				alpha = max(alpha, t_val)
			elif(t_flag == UPPER):
				beta = min(beta, t_val)
			
			if(alpha >= beta):
				return (t_val, t_move)


		# We check if the current symmetrical encoded board is already in the table
		sym_key = (encoded_symmetrical_game_board, cur_player)
		if(sym_key in best_move_from_game_state):
			t_val, t_move, t_flag = best_move_from_game_state[sym_key]
			sym_move = (game_board.numColumns - 1) - t_move 

			if (t_flag == EXACT):
				return (t_val, sym_move)
			elif (t_flag == LOWER):
				alpha = max(alpha, t_val)
			elif(t_flag == UPPER):
				beta = min(beta, t_val)
			
			if(alpha >= beta):
				return (t_val, sym_move)

	

		states = [] # Temporarily holds all the final states
		sorted_columns = self._sort_columns_ID(game_board, cur_player, previous_iteration_best_move_from_game_state, encoded_game_board) # Sort the columns to improve move ordering
		self.numExpanded+=1 # We expand the node and then check all of its children 
		for column in sorted_columns:
			if(game_board.checkSpace(game_board.numRows-1, column).value != " "): # If the column is full, then we try a different one
				continue

			game_board.addPiece(column, cur_player)
			encoded_game_board = add_piece_to_encoded_board(encoded_game_board, game_board.numRows, game_board.numColumns, column, cur_player)
			encoded_symmetrical_game_board = add_piece_to_encoded_board(encoded_symmetrical_game_board,  game_board.numRows, game_board.numColumns, (game_board.numColumns-1-column) , cur_player)

			if (cur_player == "X"):
				expected_final_state = self._run_minimax_alpha_beta(game_board, encoded_game_board, encoded_symmetrical_game_board, "O", alpha, beta, cur_depth+1, max_depth, best_move_from_game_state, previous_iteration_best_move_from_game_state) # If the current player is X (i.e. MAX player), then the next move is O's move
				
				game_board.removePiece(column)
				encoded_game_board = remove_piece_from_encoded_board(encoded_game_board, game_board.numRows, game_board.numColumns, column)
				encoded_symmetrical_game_board = remove_piece_from_encoded_board(encoded_symmetrical_game_board, game_board.numRows, game_board.numColumns, (game_board.numColumns-1-column))

				expected_final_score = expected_final_state[0]
				states.append((expected_final_score, column))
				if (expected_final_score > max_score):  # MAX is trying to maximise the minimax value
					max_score = expected_final_score
					chosen_move = column

				alpha = max(alpha, max_score)
				if (beta <= alpha): 
					self.numPruned+=1
					break
			else:
				expected_final_state = self._run_minimax_alpha_beta(game_board, encoded_game_board, encoded_symmetrical_game_board, "X", alpha,beta, cur_depth+1, max_depth, best_move_from_game_state, previous_iteration_best_move_from_game_state)  # If the current player is O (i.e. MIN player), then the next move is X's move
				
				game_board.removePiece(column)
				encoded_game_board = remove_piece_from_encoded_board(encoded_game_board, game_board.numRows, game_board.numColumns, column)
				encoded_symmetrical_game_board = remove_piece_from_encoded_board(encoded_symmetrical_game_board, game_board.numRows, game_board.numColumns, (game_board.numColumns-1-column))

				expected_final_score = expected_final_state[0]
				states.append((expected_final_score, column))
				if (expected_final_score < min_score): #  MIN is trying to minimise the minimax value.
					min_score = expected_final_score
					chosen_move = column

				beta = min(beta, min_score)
				if (alpha >= beta):
					self.numPruned+=1
					break

		# If several states have the same best value, then the column that is closest to the center is chosen
		for state in states:
			score = state[0]
			move=state[1]
			center_col = game_board.numColumns // 2
			if(
				(cur_player == "X" and score == max_score and abs(center_col - move) < abs(center_col - chosen_move))
				or 
				(cur_player == "O" and score == min_score and abs(center_col - move) < abs(center_col - chosen_move))
			):
				chosen_move = move

		if(cur_player == "X"):
			best_value = max_score
		else:
			best_value = min_score

		flag = EXACT 
		if(best_value <= alpha_orig):
			flag= UPPER # We failed to beat alpha, and the true value is actually <= best_value
		elif(best_value >= beta_orig):
			flag = LOWER # We pruned, and the true value is >= best_value
		
		best_move_from_game_state[key] = (best_value, chosen_move, flag) # Save the move in the TT
		return (best_value, chosen_move)


	# Helps to improve move ordering
	# We first try columns that are closer to the center as they tend to lead to faster endings
	# The best move from this game state for the current max depth is likely to be the best move from this game state when we explored down to depth - 1
	# If some move is a winning one, then we try it first
	def _sort_columns_ID(self,game_board, cur_player:str, previous_iteration_best_move_from_game_state, encoded_game_board) -> list[int]:
		sorted_columns=[]
		i = 0
		j = game_board.numColumns-1
		midpoint_idx= math.floor(game_board.numColumns / 2)
		# Trying moves in the middle of the board is better, as it gives more opportunity to win
		while i < j: # We first add the columns that are farthest away from the midpoint and then reverse the list
			sorted_columns.append(i)
			sorted_columns.append(j)
			i+=1
			j-=1
		if game_board.numColumns % 2 == 1:
			sorted_columns.append(midpoint_idx)
		sorted_columns.reverse()

		# The best move from this game state for the current max depth is likely to be the best move from this game state when we explored down to depth - 1
		# So here the previous move means a move from exact same location, but in previous iteration of Iterative deepening
		if((encoded_game_board, cur_player) in previous_iteration_best_move_from_game_state): # It is False if we are at level 0
			previous_best_move = previous_iteration_best_move_from_game_state[(encoded_game_board, cur_player)]

			# Remove and place this column at the start of the list so that we try it first
			removed_column = sorted_columns.pop(sorted_columns.index(previous_best_move[1]))
			sorted_columns.insert(0, removed_column) 


		for i in range(len(sorted_columns)):
			cur_column = sorted_columns[i]
			is_added: bool = game_board.addPiece(cur_column, cur_player)
			if game_board.checkWin() == True and is_added == True: 
				# Remove and place this column at the start of the list so that we try it first
				removed_column=sorted_columns.pop(i)
				sorted_columns.insert(0, removed_column) 
			if(is_added == True):
				game_board.removePiece(cur_column)
		return sorted_columns

		


	# We use Iterative Deepening (ID) here in order to help ordering the columns, which in its turn will result in early cutoffs by alpha beta pruning
	# This is because a move that was best at depth 3 with (e.g.) cur_max_depth=5, is likely to be the best at the same depth 3 in the next iteration (i.e. with cur_max_depth=6) 
	def getMoveAlphaBeta(self, game_board):
		isEmpty=True
		for i in range(game_board.numColumns):
			if(game_board.checkSpace(0,i).value != " "):
				isEmpty=False
		# If it is our 1st move, then it is always the best strategy to go in the middle column
		if (isEmpty == True): 
			return math.floor(game_board.numColumns/2) 

		alpha= -math.inf # The worst case for a MAX player
		beta= math.inf # The worst case for a MIN player


		max_depth = -1 # The maximum depth down to which we search

		if(game_board.numColumns == 7 and game_board.numRows == 6 and game_board.winNum == 4):
			max_depth=9
		elif(game_board.numColumns >= 38):
			max_depth = 1
		elif(game_board.numColumns >=  20):
			max_depth = 2
		elif(game_board.numColumns >= 16):
			max_depth=3
		elif(game_board.numColumns  >= 10):
			max_depth = 4
		elif(game_board.numColumns  >= 8):
			max_depth = 5 
		elif(game_board.numColumns  >= 6):
			max_depth = 6
		elif(game_board.numColumns  >= 5):
			max_depth = 7
		else:
			max_depth=8


		cur_max_depth = 1  # Initial maximum depth
		previous_iteration_best_move_from_game_state = {} # Stores the best move from a praticular game state from previous iteration
		chosen_move = -1
		
		encoded_game_board: tuple[int,int] = encode_game_board(game_board)
		encoded_symmetrical_game_board: tuple[int,int]  = get_encoded_symmetrical_board(encoded_game_board, game_board.numRows, game_board.numColumns)

		while(cur_max_depth <= max_depth): 
			best_move_from_game_state = {} # Stores the best move from a particular game state in this iteration
			best_expected_state = self._run_minimax_alpha_beta(game_board, encoded_game_board, encoded_symmetrical_game_board, self.name, alpha, beta, 0, cur_max_depth, best_move_from_game_state, previous_iteration_best_move_from_game_state)
			previous_iteration_best_move_from_game_state = deep_copy(best_move_from_game_state) # This table will be used in the next iteration to improve move ordering
			chosen_move = best_expected_state[1]
			cur_max_depth+=1
		return chosen_move






# Returns a deep copy for some Python types
def deep_copy(obj):
	if obj is None or isinstance(obj, (int, float, bool, str, bytes)):
		return obj
	if isinstance(obj, tuple):
		return tuple(deep_copy(x) for x in obj)
	if isinstance(obj, dict):
		return {deep_copy(k): deep_copy(v) for k, v in obj.items()}
	raise TypeError("Unsupported type for deep copy")



# Encodes the game board into a tuple of two binary numbers, each of each is of length numColumns * numRows. The first one encodes the spaces occupied by X and the 2nd number encodes the spaces occupied by player O 
def encode_game_board(game_board) -> tuple[int,int]:
	xs=0b0
	os=0b0
	for r in range(game_board.numRows):
		for c in range(game_board.numColumns):
			xs = xs << 1
			os = os << 1
			cur_space_val = game_board.checkSpace(r,c).value 
			if (cur_space_val == "X"):
				xs = xs | 1
			elif(cur_space_val == "O"):
				os= os | 1
	return (xs,os)



# Returns the board symmetrical to a provided board
def get_encoded_symmetrical_board(encoded_game_board: tuple[int,int], numRows, numColumns) -> tuple[int,int]:
	xs = encoded_game_board[0]
	os = encoded_game_board[1]

	symmetrical_xs = 0b0
	symmetrical_os = 0b0

	for i in range(numRows):
		mask = (2**(numColumns)-1) << ((numRows-1-i) * numColumns) # The mask to extract a row (numRows-i). i.e. we start extraction from the end (i.e. we preserve representation defined in encode_game_board() function)

		xs_row = (xs & mask) >> ((numRows-1-i) * numColumns) # We extract a row of X's and move it right (so that the first numColumns bits in this number represent an extracted row)
		symmetrical_xs_row = reverse_bits(xs_row, numColumns) # We reverse the row of X's
		symmetrical_xs = symmetrical_xs << (numColumns) # Move the number left so that we can add this new symmetrical row to it
		symmetrical_xs = symmetrical_xs | symmetrical_xs_row # Add the symmetrical row to the number

		os_row = (os & mask) >> ((numRows-1-i) * numColumns) # We extract a row of O's and move it right (so that the first numColumns bits in this number represent an extracted row)
		symmetrical_os_row = reverse_bits(os_row, numColumns) # We reverse the row of O's
		symmetrical_os = symmetrical_os << (numColumns) # Move the number left so that we can add this new symmetrical row to it
		symmetrical_os = symmetrical_os | symmetrical_os_row # Add the symmetrical row to the number
	return (symmetrical_xs, symmetrical_os)


# Adds a piece to encoded game board and returns a modified board
def add_piece_to_encoded_board(encoded_game_board: tuple[int,int], numRows: int, numColumns:int, column:int,  cur_player: str) -> tuple[int,int]:
	xs = encoded_game_board[0]
	os = encoded_game_board[1]

	column_mask = 1 << (numColumns - 1 - column) # Extracts the column (we ensure that we always extract the first numColumns bits)

	# We extract rows from bottom to top
	for row in range(numRows):
		row_mask = (2**(numColumns)-1) << ((numRows-1-row) * numColumns)

		xs_row = (xs & row_mask) >> ((numRows-1-row) * numColumns) # We extract a row of X's and move it right (so that the first numColumns bits in this number represent an extracted row)
		os_row = (os & row_mask) >> ((numRows-1-row) * numColumns) # We extract a row of O's and move it right (so that the first numColumns bits in this number represent an extracted row)

		if (xs_row & column_mask == 0 and os_row & column_mask == 0): # Once we encounter the first row that has nothing in the destination column, we will add the piece to that column 
			bit_index = (numRows*numColumns - 1) - (row*numColumns + column)
			bit_mask = 1 << bit_index 

			if(cur_player == "X"): # If this is the move of player X, then add a piece to X's encoding
				xs = xs | bit_mask
			elif(cur_player == "O"): # If this is the move of player O, then add a piece to O's encoding
				os = os | bit_mask
			return (xs,os)
	return (xs,os)


# Removes a piece from an encoded game board and returns a modified board
def remove_piece_from_encoded_board(encoded_game_board: tuple[int,int], numRows: int, numColumns: int, column:int) -> tuple[int,int]:
	xs = encoded_game_board[0]
	os = encoded_game_board[1]

	column_mask = 1 << (numColumns - 1 - column) # Extracts the column (we ensure that we always extract the first numColumns bits)

	# We extract rows from top to bottom
	for row in range(numRows):
		row_mask = (2**(numColumns)-1) << (row * numColumns)

		xs_row = (xs & row_mask) >> (row * numColumns) # We extract a row of X's and move it right (so that the first numColumns bits in this number represent an extracted row)
		os_row = (os & row_mask) >> (row * numColumns) # We extract a row of O's and move it right (so that the first numColumns bits in this number represent an extracted row)

		bit_index =  (row*numColumns + (numColumns-1-column))
		bit_mask = 1 << bit_index

		# Once we encounter the first row that has some piece in the destination column, then we remove that piece from that column
		if(xs_row & column_mask != 0): 
			xs = xs & (~bit_mask) # Remove a piece from a column and leave the rest as it is
			return (xs,os)
		elif(os_row & column_mask != 0):
			os = os & (~bit_mask) # Remove a piece from a column and leave the rest as it is
			return (xs,os)
	return (xs,os)
		
		

		

		



			


		
# Reverses the first num_width bits of num
def reverse_bits(num, num_width):
	reversed_num=0
	for i in range(num_width):
		reversed_num = (reversed_num << 1) | (num & 1)
		num = num >> 1
	return reversed_num

