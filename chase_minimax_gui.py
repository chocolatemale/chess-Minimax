import chess
from IPython.display import display, clear_output
import random

# Enhanced evaluation function with positional values
def evaluate_board(board):
    piece_values = {
        'PAWN': 1, 'KNIGHT': 3, 'BISHOP': 3, 'ROOK': 5, 'QUEEN': 9, 'KING': 0
    }
    
    piece_position_values = {
        'PAWN': [
              0,  0,  0,  0,  0,  0,  0,  0,
              5, 10, 10,-20,-20, 10, 10,  5,
              5, -5,-10,  0,  0,-10, -5,  5,
              0,  0,  0, 20, 20,  0,  0,  0,
              5,  5, 10, 25, 25, 10,  5,  5,
             10, 10, 20, 30, 30, 20, 10, 10,
             50, 50, 50, 50, 50, 50, 50, 50,
              0,  0,  0,  0,  0,  0,  0,  0
        ],
        'KNIGHT': [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ],
        'BISHOP': [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ],
        'ROOK': [
              0,  0,  0,  5,  5,  0,  0,  0,
             -5,  0,  0,  0,  0,  0,  0, -5,
             -5,  0,  0,  0,  0,  0,  0, -5,
             -5,  0,  0,  0,  0,  0,  0, -5,
             -5,  0,  0,  0,  0,  0,  0, -5,
             -5,  0,  0,  0,  0,  0,  0, -5,
              5, 10, 10, 10, 10, 10, 10,  5,
              0,  0,  0,  0,  0,  0,  0,  0
        ],
        'QUEEN': [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  5,  5,  5,  5,  5,  0,-10,
              0,  0,  5,  5,  5,  5,  0, -5,
             -5,  0,  5,  5,  5,  5,  0, -5,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ],
        'KING': [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
             20, 20,  0,  0,  0,  0, 20, 20,
             20, 30, 10,  0,  0, 10, 30, 20
        ],
    }

    piece_symbol_to_name = {
        'P': 'PAWN', 'N': 'KNIGHT', 'B': 'BISHOP',
        'R': 'ROOK', 'Q': 'QUEEN', 'K': 'KING'
    }

    evaluation = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        piece_name = piece_symbol_to_name[chess.Piece(piece.piece_type, piece.color).symbol().upper()]
        value = piece_values[piece_name]
        
        if piece.color == chess.WHITE:
            evaluation += value
            if piece_name in piece_position_values:
                evaluation += piece_position_values[piece_name][square]
        else:
            evaluation -= value
            if piece_name in piece_position_values:
                mirrored_square = chess.square_mirror(square)
                evaluation -= piece_position_values[piece_name][mirrored_square]

    return evaluation

# Dynamic depth based on game phase
def dynamic_depth(board):
    if len(board.piece_map()) < 10:
        return 4
    return 3

# Minimax with alpha-beta pruning
def alphabeta(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = alphabeta(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = alphabeta(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# AI move generator using alpha-beta pruning
def ai_move_alphabeta(board, depth):
    best_moves = []  # List to store all moves with the best score
    alpha = float('-inf')
    beta = float('inf')
    best_score = float('-inf') if board.turn else float('inf')  # Initialize the best score accordingly

    for move in board.legal_moves:
        board.push(move)
        eval = alphabeta(board, depth - 1, alpha, beta, not board.turn)
        board.pop()

        # If we found a new best score, reset the list of best moves
        if (board.turn and eval > best_score) or (not board.turn and eval < best_score):
            best_score = eval
            best_moves = [move]
        # If the move has the same score as the current best score, add it to the list
        elif eval == best_score:
            best_moves.append(move)

        if board.turn:
            alpha = max(alpha, eval)
        else:
            beta = min(beta, eval)

    # 90% of the time, choose the first best move found, 10% of the time choose randomly from the best moves
    if random.random() < 0.85 or len(best_moves) == 1:
        return best_moves[0]
    else:
        return random.choice(best_moves)

# GUI to visualize board
def display_board_gui(board, round_num):
    clear_output(wait=True)
    print(f"Round {round_num}")
    
    # Display the board with rank numbers
    board_str = str(board)
    rows = board_str.split("\n")
    for i in range(8, 0, -1):
        print(f"{i} | {rows[8-i]}")
    print("   " + "-"*17)  # Add horizontal divider
    print("    a b c d e f g h")  # Display the file labels
    print()

def main():
    board = chess.Board()
    round_num = 1
    while not board.is_game_over():
        display_board_gui(board, round_num)
        if board.turn == chess.WHITE:
            print("White's Turn")
            # White makes random moves to demonstrate the black is capturing white's mistakes.
            # move = ai_move_alphabeta(board, dynamic_depth(board))
            move = random.choice(list(board.legal_moves))
            piece = board.piece_at(move.from_square)
            print(f"White's Random Move: {piece.symbol()} from {chess.SQUARE_NAMES[move.from_square]} to {chess.SQUARE_NAMES[move.to_square]}")
        else:
            print("Black's Turn")
            move = ai_move_alphabeta(board, dynamic_depth(board))
            piece = board.piece_at(move.from_square)
            print(f"AI's Move: {piece.symbol()} from {chess.SQUARE_NAMES[move.from_square]} to {chess.SQUARE_NAMES[move.to_square]}")
        board.push(move)
        print("\n" + "-"*40 + "\n")
        round_num += 1
    print("Game Over")
    result = board.result()
    if result == "1-0":
        print("White wins!")
        if board.is_checkmate():
            print("Reason: Black is checkmated.")
    elif result == "0-1":
        print("Black wins!")
        if board.is_checkmate():
            print("Reason: White is checkmated.")
    else:
        print("It's a draw!")
        if board.is_stalemate():
            print("Reason: Stalemate.")
        elif board.is_insufficient_material():
            print("Reason: Insufficient material.")
        elif board.is_seventyfive_moves():
            print("Reason: 75 moves without a capture or pawn move.")
        elif board.is_fivefold_repetition():
            print("Reason: Position repeated five times.")
    print(f"Total Rounds: {round_num - 1}")

if __name__ == "__main__":
    main()