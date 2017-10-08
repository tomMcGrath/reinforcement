import boards
import players
import itertools

class Game(object):
	def __init__(self, board, player1, player2):
		self.board = board
		self.player1 = player1
		self.player2 = player2
		self.current_player = 0

	def play(self):
		## Check the board state
		is_won = board.winning_move()

		## Gameplay loop
		while not is_won:
			if self.current_player == 0:
				self.player1.move(self.board)
				is_won = board.winning_move()

				if is_won:
					return 0

				else:
					self.current_player = 1

			elif self.current_player == 1:
				self.player2.move(self.board)
				is_won = board.winning_move()
				
				if is_won:
					return 1

				else:
					self.current_player = 0

			else:
				print "ERROR in move"


board = boards.Board()
player1 = players.HumanPlayer('x')
player2 = players.HumanPlayer('o')

game = Game(board, player1, player2)
winner = game.play()