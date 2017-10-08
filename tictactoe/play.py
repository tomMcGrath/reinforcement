import boards
import players
import itertools
import matplotlib.pyplot as plt
import numpy as np

class Game(object):
	def __init__(self, board, player1, player2):
		self.board = board
		self.player1 = player1
		self.player2 = player2
		self.current_player = 0

	def play(self):
		## Check the board state
		is_won = board.winning_move()
		is_draw = board.draw_move()

		## Gameplay loop
		while not (is_won or is_draw):
			if self.current_player == 0:
				self.player1.move(self.board)
				is_won = board.winning_move()
				is_draw = board.draw_move()

				if is_won:
					return 0

				elif is_draw:
					return -1

				else:
					self.current_player = 1

			elif self.current_player == 1:
				self.player2.move(self.board)
				is_won = board.winning_move()
				is_draw = board.draw_move()
				
				if is_won:
					return 1

				elif is_draw:
					return -1

				else:
					self.current_player = 0

			else:
				print "ERROR in move"

## Run iterated trials
num_trials = 1000
eps = 0.1
alpha = 0.01

## Setup game
board = boards.Board()
init_state = board.get_state()
value_dict = {}
player1 = players.GreedyRLPlayer('o', value_dict, eps, alpha)
player2 = players.GreedyRLPlayer('x', value_dict, eps, alpha)
game = Game(board, player1, player2)

wins = []
for i in range(num_trials):
	board.set_state(init_state)
	winner = game.play()

	if winner == 0:
		player1.win_update(board)
		player2.loss_update(board)

	elif winner == 1:
		player1.loss_update(board)
		player2.win_update(board)

	elif winner == -1:
		player1.draw_update(board)
		player2.win_update(board) # test finding optimal p2 strategt

	wins.append(winner)

## Analyse the results
wins = np.array(wins)
p1_wins = np.cumsum(wins == 0)
p2_wins = np.cumsum(wins == 1)
draws = np.cumsum(wins == -1)
fig, axes = plt.subplots(1)
axes.plot(p1_wins, c='r')
axes.plot(p2_wins, c='b')
axes.plot(draws, c='k')
plt.show()

## Now let the human play!
board.set_state(init_state)
human = players.HumanPlayer('x')
game = Game(board, player1, human)
game.play()