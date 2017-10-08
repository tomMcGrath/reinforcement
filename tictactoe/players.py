import boards
import numpy as np

class HumanPlayer(object):
	def __init__(self, symbol):
		self.symbol = symbol

	def move(self, board):
		state = board.get_state()
		print board.statestring()

		moves = board.legal_moves()
		movestrings = board.legal_movestrings(self.symbol)
		moveset = zip(moves, movestrings)
		print moveset

		moveloc = int(input('Which square do you want to move in? '))
		mymove = (moveloc, self.symbol)

		if board.is_valid_move(mymove):
			board.update(mymove)

		else:
			print 'Illegal move!'
			self.move(board)

class RandomPlayer(object):
	def __init__(self, symbol):
		self.symbol = symbol

	def move(self, board):
		moves = board.legal_moves()
		moveloc = np.random.choice(moves)
		mymove = (moveloc, self.symbol)

		if board.is_valid_move(mymove):
			board.update(mymove)
			print '\nMove by %s' %(self.symbol)
			print board.statestring()

		else:
			print 'Illegal move!'
			self.move(board)

class GreedyRLPlayer(object):
	def __init__(self, symbol, value_dict, eps, alpha):
		self.symbol = symbol
		self.value_dict = value_dict
		self.eps = eps
		self.alpha = alpha

	def move(self, board):
		## Get value dictionary, legal moves
		value_dict = self.value_dict
		moves = board.legal_moves()
		movestrings = board.legal_movestrings(self.symbol)
		known_states = value_dict.keys() # save getting this repeatedly

		## Populate the move dictionary
		## First add current state
		curr_state = board.get_state()
		if curr_state not in known_states:
			value_dict[curr_state] = 0.5

		## Now add all possible states from this position
		moveset = zip(moves, movestrings)
		for x in moveset:
			if x[1] not in known_states:
				value_dict[x[1]] = 0.5 # default

		## Now do the epsilon-greedy algorithm
		do_greedy = np.random.binomial(1, self.eps)

		if do_greedy:
			chosen_idx = np.random.randint(len(moveset))
			chosen_move = moveset[chosen_idx]

		else:
			## Find the highest-value move
			## TODO: make random choice if several have maxval
			sorted_moves = sorted(moveset, key=lambda x:value_dict[x[1]])
			chosen_move = sorted_moves[0]

		## Update the value dictionary
		newval = value_dict[chosen_move[1]]
		oldval = value_dict[board.get_state()]
		value_dict[board.get_state()] = oldval + self.alpha*(newval - oldval) # Temporal difference step

		## Store the value dictionary again
		self.value_dict = value_dict

		## Now take the move
		moveloc = chosen_move[0]
		mymove = (moveloc, self.symbol)

		if board.is_valid_move(mymove):
			board.update(mymove)

		else:
			print 'Illegal move!'
			self.move(board)

	def win_update(self, board):
		## Update when agent wins - set probability of winning in final state to be 1
		self.value_dict[board.get_state()] = 1

	def loss_update(self, board):
		## Update when agent loses - set probability of winning in final state to be 0
		self.value_dict[board.get_state()] = 0

	def draw_update(self, board):
		## Update when agent draws - set probability of winning in final state to be 0
		self.value_dict[board.get_state()] = 0

	def get_value_dict(self):
		return self.value_dict
