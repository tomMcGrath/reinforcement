import boards
import sys

class HumanPlayer(object):
	def __init__(self, symbol):
		self.symbol = symbol

	def move(self, board):
		state = board.get_state()
		print '%s\n%s\n%s' %(state[:3], state[3:6], state[6:])
		moveloc = int(input('Which square do you want to move in? '))
		mymove = (moveloc, self.symbol)

		if board.is_valid_move(mymove):
			board.update(mymove)

		else:
			print 'Illegal move!'
			self.move(board)