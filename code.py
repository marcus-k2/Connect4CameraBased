import cv2
import numpy as np
import mediapipe as mp
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Connect Four game parameters
ROWS = 6
COLS = 7
CELL_SIZE = 80
RADIUS = int(CELL_SIZE / 2 - 5)
WIDTH = COLS * CELL_SIZE
HEIGHT = (ROWS + 1) * CELL_SIZE  # Extra row for dropping animation

# Colors
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

# Game state
board = np.zeros((ROWS, COLS))
game_over = False
turn = 0  # 0 for player 1 (red), 1 for player 2 (yellow)
last_drop_time = 0
drop_delay = 1.0  # seconds between drops

def create_board():
    return np.zeros((ROWS, COLS))

def draw_board(img, board):
    # Draw the board
    cv2.rectangle(img, (0, CELL_SIZE), (WIDTH, HEIGHT), BLUE, -1)
    
    for c in range(COLS):
        for r in range(ROWS):
            center = (c * CELL_SIZE + CELL_SIZE // 2, (r + 1) * CELL_SIZE + CELL_SIZE // 2)
            color = BLACK
            if board[r][c] == 1:
                color = RED
            elif board[r][c] == 2:
                color = YELLOW
            cv2.circle(img, center, RADIUS, color, -1)
    
    return img

def is_valid_location(board, col):
    return board[0][col] == 0

def get_next_open_row(board, col):
    for r in range(ROWS-1, -1, -1):
        if board[r][col] == 0:
            return r
    return -1

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def winning_move(board, piece):
    # Check horizontal locations
    for c in range(COLS-3):
        for r in range(ROWS):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Check vertical locations
    for c in range(COLS):
        for r in range(ROWS-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positively sloped diagonals
    for c in range(COLS-3):
        for r in range(ROWS-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Check negatively sloped diagonals
    for c in range(COLS-3):
        for r in range(3, ROWS):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
    return False

def process_hand_landmarks(img, landmarks, board, current_time):
    global turn, game_over, last_drop_time
    
    # Get the x-coordinate of the index finger tip (landmark 8)
    h, w = img.shape[:2]
    index_finger_tip = landmarks.landmark[8]
    cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
    
    # Draw the finger position
    cv2.circle(img, (cx, cy), 10, (0, 255, 0), -1)
    
    # Only allow drops every 'drop_delay' seconds
    if current_time - last_drop_time < drop_delay:
        return
    
    # If finger is in the board area
    if CELL_SIZE <= cy <= HEIGHT and 0 <= cx <= WIDTH:
        col = int(cx / CELL_SIZE)
        
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, turn + 1)
            
            if winning_move(board, turn + 1):
                game_over = True
                print(f"Player {turn + 1} wins!")
            
            turn = (turn + 1) % 2
            last_drop_time = current_time

def main():
    global board, game_over, turn
    
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    
    board = create_board()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip and resize the frame
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if not game_over:
                    process_hand_landmarks(frame, hand_landmarks, board, current_time)
        
        # Draw the board
        frame = draw_board(frame, board)
        
        # Show whose turn it is
        turn_text = f"Player {turn + 1}'s turn"
        color = RED if turn == 0 else YELLOW
        cv2.putText(frame, turn_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if game_over:
            winner_text = f"Player {(turn) % 2 + 1} wins! Press R to restart"
            cv2.putText(frame, winner_text, (WIDTH//4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        
        cv2.imshow('Connect Four in the Air', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            board = create_board()
            game_over = False
            turn = 0
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
