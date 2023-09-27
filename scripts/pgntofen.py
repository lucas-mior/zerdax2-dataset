#!/usr/bin/python

import sys
import chess
import chess.pgn
import threading
import multiprocessing


def games_chunk(pgns, i=0):
    outfile = open(f"out_{i}.fen", "w")
    for game in pgns:
        board = game.board()

        for move in game.mainline_moves():
            board.push(move)
        fen = board.fen()

        outfile.write(fen + "\n")
    return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit(1)

    pgnFilePath = sys.argv[1]
    print('PGN File: ', pgnFilePath)
    pgn = open(pgnFilePath)

    pgns = []
    i = 0
    while game := chess.pgn.read_game(pgn):
        pgns.append(game)

    print("start processing...")
    nthreads = multiprocessing.cpu_count()
    chunk_size = max(1, len(pgns) / nthreads)

    if chunk_size > nthreads:
        threads = []
        for i in range(nthreads):
            start = chunk_size * i;
            end = chunk_size * (i+1);
            if i == (nthreads - 1):
                end = len(pgns)

            group = pgns[start:end]
            thread = threading.Thread(target=games_chunk, args=(group, i))
            threads.append(thread)
            thread.start()

        for t in threads:
            t.join()
    else:
        games_chunk(pgns)
