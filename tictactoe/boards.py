import itertools
import re

class Board(object):
	"""
	The basic 3x3 tic-tac-toe board
	States are represented as 9 character strings:
	-: empty
	o: circles
	x: crosses

	e.g. --ox-o--x-- represents board:
	 | |o
	x| |o
	x| |
	"""

	def __init__(self, state='---------'):
		self.state = state

	def set_state(self, state):
		self.state = state

	def is_valid_move(self, move):
		"""
		Check if proposed move is valid. Moves are tuple of form (index, symbol).
		Moves are valid if the proposed square is empty
		"""
		return self.state[move[0]] == '-'

	def update(self, move):
		"""
		Update the board state if the move is valid
		"""
		if not self.is_valid_move(move):
			raise ValueError

		else:
			self.state = self.state[:move[0]] + move[1] + self.state[move[0]+1:]

	def get_state(self):
		"""
		Give the state of the board
		"""
		return self.state

	def statestring(self):
		"""
		Return the board state in a way that's nicer to print
		"""
		state = self.state
		return '%s\n%s\n%s' %(state[:3], state[3:6], state[6:])

	def winning_move(self):
		"""
		Check if the current state of the board is a winning configuration
		"""
		state = self.state
		#print '\n', self.statestring()

		## Vertical winning
		for i in range(0,3):
			if state[i] == state[i+3] and state[i+3] == state[i+6] and state[i] != '-':
				return True

		## Horizontal winning
		for i in range(0,3):
			if state[3*i] == state[3*i+1] and state[3*i+1] == state[3*i + 2] and state[3*i] != '-':
				return True

		## Diagonal winning
		if state[0] == state[4] and state[4] == state[8] and state[0] != '-':
			return True

		if state[2] == state[4] and state[4] == state[6] and state[2] != '-':
			return True

		## Otherwise not winning
		return False

	def draw_move(self):
		"""
		Check for draws
		"""
		return len(self.legal_moves()) == 0

	def legal_moves(self):
		"""
		Returns a list of legal moves in the form of a tuple of indices of blank spaces
		"""
		return [idx.start() for idx in re.finditer('-', self.state)]


	def legal_movestrings(self, symbol):
		"""
		Returns a list of all allowed board states
		"""
		movelist = self.legal_moves()
		state = self.state

		movestrings = []
		for move in movelist:
			movestrings.append(state[:move] + symbol + state[move+1:])

		return movestrings
