#!/usr/bin/python

import numpy as np
import chess
import copy
import misc
import sys
from sys import stderr


def generate_random_board(num_pieces):
    piece_types = ['P', 'N', 'B', 'R', 'Q',
                   'p', 'n', 'b', 'r', 'q']
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

    board = chess.Board("8/8/8/8/8/8/8/8")
    for piece in pieces:
        while True:
            rank = np.random.randint(0, 8)
            file = np.random.randint(0, 8)
            square = chess.square(file, rank)
            if board.piece_at(square) is None:
                board.set_piece_at(square, chess.Piece.from_symbol(piece))
                break

    return board


for n in range(1, 31):
    stderr.write(f"npieces = {n}\n")
    for i in range(1000):
        if i % 100 == 0:
            stderr.write(f"got {i}\n")
        while True:
            board = generate_random_board(n)
            if board.is_valid():
                break
        print(str.split(board.fen())[0])
