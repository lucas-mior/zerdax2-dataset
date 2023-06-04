import numpy as np
import chess
import copy
import misc
import sys


def generate_random_fen(num_pieces):
    piece_types = ['P', 'N', 'B', 'R', 'Q',
                   'p', 'n', 'b', 'r', 'q']
    board = chess.Board("8/8/8/8/8/8/8/8")
    rules = copy.deepcopy(misc.AMOUNT)
    pieces = ['K', 'k']

    for _ in range(num_pieces):
        while True:
            piece = np.random.choice(piece_types)
            rule = rules[piece]
            if rule[0] < rule[1]:
                rule[0] += 1
                pieces.append(piece)
                break
            else:
                piece_types.remove(piece)

    for piece in pieces:
        while True:
            rank = np.random.randint(0, 8)
            file = np.random.randint(0, 8)
            square = chess.square(file, rank)
            if board.piece_at(square) is None:
                board.set_piece_at(square, chess.Piece.from_symbol(piece))
                break

    if not board.is_valid():
        return generate_random_fen(num_pieces)

    return str.split(board.fen())[0]


sys.setrecursionlimit(10000)

for n in range(1, 31):
    for i in range(400):
        print(generate_random_fen(n))
