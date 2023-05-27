import random
import chess

def generate_random_chess_fen(piece_count):
    while True:
        # Initialize an empty chessboard
        chessboard = chess.Board("8/8/8/8/8/8/8/8")

        # Add the two kings
        chessboard.set_piece_at(random.randint(0, 63), chess.Piece(chess.KING, chess.WHITE))
        chessboard.set_piece_at(random.randint(0, 63), chess.Piece(chess.KING, chess.BLACK))

        # Add random pieces based on piece_count
        for _ in range(piece_count):
            piece = random.choice([chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT])
            square = random.randint(0, 63)
            chessboard.set_piece_at(square, chess.Piece(piece, chess.WHITE))

        # Convert the chessboard to FEN
        fen = chessboard.fen()

        # Validate the position
        if is_valid_chess_position(fen):
            return fen

def is_valid_chess_position(fen):
    board = chess.Board(fen)
    return board.is_valid()

# Generate random chess FENs for each piece count
piece_counts = [1, 2, 3, 4]
for count in piece_counts:
    for _ in range(700):
        random_fen = generate_random_chess_fen(count)
        print(count+2, random_fen)

