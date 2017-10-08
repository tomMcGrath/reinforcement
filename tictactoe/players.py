import boards
import numpy as np

class HumanPlayer(object):
	def __init__(self, symbol):
		self.symbol = symbol

	def move(self, board):
		state = board.get_state()
		print board.statestring()
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
	def __init__(self, symbol, value_dict, eps):
		self.symbol = symbol
		self.value_dict = value_dict
		self.eps = eps