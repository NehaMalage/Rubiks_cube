import json
import os
from random import randint, choice
from tqdm import tqdm
from collections import deque

NUMBER_OF_BLOCKS = 3
MAX_SHUFFLE = 6    # maximum number of moves to shuffle the cube
MIN_SHUFFLE = 5
MAX_MOVES = 5     # maximum moves for heuristic database
NEW_HEURISTICS = False
HEURISTIC_FILE = f'heuristic_{NUMBER_OF_BLOCKS}.json'  
MAX_SOLVE_DEPTH = 20  # maximum depth for DFS solver

def solved(cube, colors, blocks):
    for i in range(6):
        face = []
        for row in range(blocks):
            face_row = []
            for col in range(blocks):
                face_row.append(colors[i])
            face.append(face_row)
        cube.append(face)
    # print(cube)
    return cube

def is_solved(cube, colors, blocks):
    # Checks if the cube is in a solved state
    for face_idx in range(6):
        for row in range(blocks):
            for col in range(blocks):
                if cube[face_idx][row][col] != colors[face_idx]:
                    return False
    return True

def rotate_face_clockwise(face, blocks):
    # Rotates a 2D face clockwise
    new_face = [[None]*blocks for _ in range(blocks)]
    for i in range(blocks):
        for j in range(blocks):
            new_face[j][blocks - 1 - i] = face[i][j]
    return new_face

def rotate_face_anticlockwise(face, blocks):
    # Rotates a 2D face anticlockwise
    new_face = [[None]*blocks for _ in range(blocks)]
    for i in range(blocks):
        for j in range(blocks):
            new_face[blocks - 1 - j][i] = face[i][j]
    return new_face

def horizontal_move(cube, direction, row, blocks):
    # Horizontal twist
    if direction == 0:  # left
        temp = [cell for cell in cube[1][row]]  # Make a copy
        cube[1][row] = [cell for cell in cube[2][row]]
        cube[2][row] = [cell for cell in cube[3][row]]
        cube[3][row] = [cell for cell in cube[4][row]]
        cube[4][row] = temp
    else:  # right
        temp = [cell for cell in cube[4][row]]  # Make a copy
        cube[4][row] = [cell for cell in cube[3][row]]
        cube[3][row] = [cell for cell in cube[2][row]]
        cube[2][row] = [cell for cell in cube[1][row]]
        cube[1][row] = temp

    # Rotate top/bottom blocks based on row
    if row == 0:  # Top face affected
        cube[0] = rotate_face_anticlockwise(cube[0], blocks) if direction == 0 else rotate_face_clockwise(cube[0], blocks)
    if row == blocks - 1:  # Bottom face affected
        cube[5] = rotate_face_anticlockwise(cube[5], blocks) if direction == 0 else rotate_face_clockwise(cube[5], blocks)

    return cube

def vertical_move(cube, direction, col, blocks):
    # Vertically rotating 
    for i in range(blocks):
        if direction == 0:  # down
            temp = cube[0][i][col]
            cube[0][i][col] = cube[4][blocks - 1 - i][blocks - 1 - col]
            cube[4][blocks - 1 - i][blocks - 1 - col] = cube[5][i][col]
            cube[5][i][col] = cube[2][i][col]
            cube[2][i][col] = temp
        else:  # up
            temp = cube[0][i][col]
            cube[0][i][col] = cube[2][i][col]
            cube[2][i][col] = cube[5][i][col]
            cube[5][i][col] = cube[4][blocks - 1 - i][blocks - 1 - col]
            cube[4][blocks - 1 - i][blocks - 1 - col] = temp

    # Rotate left/right blocks based on column
    if col == 0:  # Left face affected
        cube[1] = rotate_face_clockwise(cube[1], blocks) if direction == 0 else rotate_face_anticlockwise(cube[1], blocks)
    if col == blocks - 1:  # Right face affected
        cube[3] = rotate_face_clockwise(cube[3], blocks) if direction == 0 else rotate_face_anticlockwise(cube[3], blocks)

    return cube

def side_move(cube, direction, col, blocks):
    # this move is to rotate vertically such that the top, left, right and bottom are affected
    for i in range(blocks):
        if direction == 0:  # down
            temp = cube[0][col][i]
            cube[0][col][i] = cube[3][i][blocks - 1 - col]
            cube[3][i][blocks - 1 - col] = cube[5][blocks - 1 - col][blocks - 1 - i]
            cube[5][blocks - 1 - col][blocks - 1 - i] = cube[1][blocks - 1 - i][col]
            cube[1][blocks - 1 - i][col] = temp
        else:  # up
            temp = cube[0][col][i]
            cube[0][col][i] = cube[1][blocks - 1 - i][col]
            cube[1][blocks - 1 - i][col] = cube[5][blocks - 1 - col][blocks - 1 - i]
            cube[5][blocks - 1 - col][blocks - 1 - i] = cube[3][i][blocks - 1 - col]
            cube[3][i][blocks - 1 - col] = temp
    
    # Rotate front/back blocks based on column
    if col == 0:  # Back face affected
        cube[4] = rotate_face_anticlockwise(cube[4], blocks) if direction == 0 else rotate_face_clockwise(cube[4], blocks)
    if col == blocks - 1:  # Front face affected
        cube[2] = rotate_face_anticlockwise(cube[2], blocks) if direction == 0 else rotate_face_clockwise(cube[2], blocks)
    
    return cube

def stringify_cube(cube):
    # Converts cube state to string for heuristic database
    cube_str = ""
    for face in cube:
        for row in face:
            for cell in row:
                cube_str += str(cell)
    return cube_str

def create_cube_from_state(cube_str):
    cube = []
    blocks = NUMBER_OF_BLOCKS
    cells_per_face = blocks * blocks
    
    for face_idx in range(6):
        face = []
        start_idx = face_idx * cells_per_face
        for row in range(blocks):
            face_row = []
            for col in range(blocks):
                char_idx = start_idx + row * blocks + col
                face_row.append(cube_str[char_idx])
            face.append(face_row)
        cube.append(face)
    return cube

def build_heuristic_db(state, actions, max_moves, heuristic=None):

    if heuristic is None:
        heuristic = {state: 0}
    
    queue = deque([(state, 0)])
    node_count = sum([len(actions) ** (x + 1) for x in range(max_moves + 1)])
    
    with tqdm(total=node_count, desc='Heuristic DB') as pbar:
        while queue:
            s, d = queue.popleft()
            if d >= max_moves:
                continue
            for a in actions:
                cube = create_cube_from_state(s)
                # a[0] is either horizontal, vertical, side move
                # a[1] is the row/column
                # a[2] is top/down or left/right
                if a[0] == 'h':
                    cube = horizontal_move(cube, a[2], a[1], NUMBER_OF_BLOCKS)
                elif a[0] == 'v':
                    cube = vertical_move(cube, a[2], a[1], NUMBER_OF_BLOCKS)
                elif a[0] == 's':
                    cube = side_move(cube, a[2], a[1], NUMBER_OF_BLOCKS)
                a_str = stringify_cube(cube)
                if a_str not in heuristic or heuristic[a_str] > d + 1:
                    heuristic[a_str] = d + 1
                    queue.append((a_str, d + 1))
                pbar.update(1) #update the progress bar
    
    return heuristic

def shuffle(cube, min_moves, max_moves, blocks):
    # Shuffles the cube with random moves

    moves = randint(min_moves, max_moves)
    possible_moves = [
        ('h', 0),
        ('h', 1),
        ('v', 0),
        ('v', 1),
        ('s', 0),
        ('s', 1)
    ]
    print(moves)
    shuffle_moves = []
    for i in range(moves):
        num = randint(0, len(possible_moves) - 1)  # choose a random move
        curr_move = possible_moves[num]
        row_or_col = randint(0, blocks - 1)  # random row or column
        move_tuple = (curr_move[0], row_or_col, curr_move[1])
        shuffle_moves.append(move_tuple)
        
        if curr_move[0] == 'h':
            cube = horizontal_move(cube, curr_move[1], row_or_col, blocks)
        elif curr_move[0] == 'v':
            cube = vertical_move(cube, curr_move[1], row_or_col, blocks)
        elif curr_move[0] == 's':
            cube = side_move(cube, curr_move[1], row_or_col, blocks)
        print(f"Move {i+1}: {move_tuple}")
        print_cube(cube)
        # print("yyyyyyyyyyyyyyyyyyyyyyyyy")
    
    print(f"Total shuffle moves: {len(shuffle_moves)}")
    return cube

def print_cube(cube):
    # Prints the cube in a readable format
    for face_idx, face in enumerate(cube):
        face_names = ["Top", "Left", "Front", "Right", "Back", "Bottom"]
        print(f"{face_names[face_idx]} face:")
        for row in face:
            print("  ", row)
        print()

def load_or_build_heuristic_db(solved_cube):

    h_db = None
    
    if os.path.exists(HEURISTIC_FILE) and not NEW_HEURISTICS:
        try:
            #in case the file already exists
            with open(HEURISTIC_FILE, 'r') as f:
                h_db = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading heuristic file: {e}")
            h_db = None

    if h_db is None or NEW_HEURISTICS:
        # if it doesnt exist then it will start building
        actions = [(move_type, row_col, direction) 
                   for move_type in ['h', 'v', 's'] 
                   for direction in [0, 1] 
                   for row_col in range(NUMBER_OF_BLOCKS)]
        
        solved_state = stringify_cube(solved_cube)
        h_db = build_heuristic_db(
            solved_state,
            actions,
            max_moves=MAX_MOVES,
            heuristic=h_db if not NEW_HEURISTICS else None
        )
        
        print(f"Saving heuristic database to {HEURISTIC_FILE}")
        try:
            with open(HEURISTIC_FILE, 'w') as f:
                json.dump(h_db, f)
            print(f"Saved heuristic database with {len(h_db)} states")
        except IOError as e:
            print(f"Error saving heuristic file: {e}")
    
    return h_db

def apply_move(cube, move):
    # Applies a move to the cube

    move_type, row_or_col, direction = move
    if move_type == 'h':
        cube = horizontal_move(cube, direction, row_or_col, NUMBER_OF_BLOCKS)
    elif move_type == 'v':
        cube = vertical_move(cube, direction, row_or_col, NUMBER_OF_BLOCKS)
    elif move_type == 's':
        cube = side_move(cube, direction, row_or_col, NUMBER_OF_BLOCKS)
    return cube

def deep_copy_cube(cube):
    # Create a deep copy of the cube
    return [[[cell for cell in row] for row in face] for face in cube]

def dfs_search(cube, moves_so_far, remaining_depth, heuristic_db, colors, visited_current_path):
    # Recursive DFS search for solution
    
    if is_solved(cube, colors, NUMBER_OF_BLOCKS):
        return moves_so_far[:]  # Return a copy of the successful move sequence
    
    if remaining_depth <= 0:
        return None
    
    state_str = stringify_cube(cube)
    if state_str in visited_current_path:
        return None  # Avoid cycles in current search path
    
    heuristic_moves = heuristic_db.get(state_str, remaining_depth + 1)
    if heuristic_moves > remaining_depth:
        return None  # Heuristic says we can't reach solution in remaining moves
    #If database says this state needs X moves to solve and we only have Y moves left then its not possible

    
    visited_current_path.add(state_str)
    
    possible_moves = [(move_type, row_col, direction) 
                     for move_type in ['h', 'v', 's'] 
                     for row_col in range(NUMBER_OF_BLOCKS) 
                     for direction in [0, 1]]
    
    # Try moves in order of heuristic promise
    move_scores = []
    for move in possible_moves:
        test_cube = deep_copy_cube(cube)
        test_cube = apply_move(test_cube, move)
        test_state = stringify_cube(test_cube)
        score = heuristic_db.get(test_state, remaining_depth)
        move_scores.append((score, move))
    
    move_scores.sort(key=lambda x: x[0])
    
    for score, move in move_scores:
        new_cube = deep_copy_cube(cube)
        new_cube = apply_move(new_cube, move)
        new_moves = moves_so_far + [move]
        
        result = dfs_search(new_cube, new_moves, remaining_depth - 1, heuristic_db, colors, visited_current_path)
        if result is not None:
            visited_current_path.remove(state_str)
            return result
    
    visited_current_path.remove(state_str)
    return None

def solve_cube(cube, heuristic_db, colors, max_depth):
    # Solve the cube using DFS
    
    print("Finding solution using DFS ")    
    for depth in range(1, max_depth + 1):
        print(f"Trying depth: {depth}")
        visited_current_path = set()
        result = dfs_search(cube, [], depth, heuristic_db, colors, visited_current_path)
        if result is not None:
            print(f"Solution found at depth: {depth}")
            return result
        print(f"No solution found at depth: {depth}")
    
    print("No solution found within max depth")
    return None

def main():
    cube = []
    #we have 6 colors w- white, o-orange, g-green, r- red, b-blue, y-yellow
    colors = ["w", "o", "g", "r", "b", "y"]
    blocks = NUMBER_OF_BLOCKS #so basically for 3x3, NUMBER_OF_BLOCKS is 3
    solved_cube = solved(cube, colors, blocks) #to get the fully solved cube
    print("Solved cube:")
    # print(cube)
    print_cube(solved_cube) #just so that we can read it properly
     
    # Load or build heuristic database
    heuristic_db = load_or_build_heuristic_db(solved_cube)
    
    # Shuffle the cube
    print("Shuffling in progress: ")
    shuffled_cube = [[[cell for cell in row] for row in face] for face in solved_cube]
    shuffled_cube = shuffle(shuffled_cube, MIN_SHUFFLE, MAX_SHUFFLE, blocks)
    print("Shuffled cube:")
    print_cube(shuffled_cube)
    
    # Solve the cube
    solution = solve_cube(shuffled_cube, heuristic_db, colors, MAX_SOLVE_DEPTH)
    
    if solution:
        print(f"\nSolution found! Moves: {len(solution)}")
        print("Solution moves:", solution)
        
        test_cube = [[[cell for cell in row] for row in face] for face in shuffled_cube]
        print("\nApplying solution moves:")
        i = 0
        for move in solution:
            test_cube = apply_move(test_cube, move)
            print(f"After move {i+1} {move}:")
            print_cube(test_cube)
            if is_solved(test_cube, colors, blocks):
                print("CUBE IS SOLVED!")
                break
            i += 1

        
        print("\nFinal cube state:")
        print_cube(test_cube)
        
        if is_solved(test_cube, colors, blocks):
            print("Cube solved")
        else:
            print("Cube not Solved")
    else:
        print("No solution found within the depth limit.")

if __name__ == "__main__":
    main()