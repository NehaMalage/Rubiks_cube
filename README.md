# Rubik's Cube Solver

A Python implementation of a Rubik's Cube solver using heuristic search with Depth-First Search (DFS). Supports configurable cube sizes and uses precomputed heuristic databases for efficient solving.

## Features

- **Configurable cube size**: Default 3x3x3, easily adjustable
- **Heuristic-based solving**: Precomputed move databases for optimal pathfinding
- **Three move types**: Horizontal, vertical, and side rotations
- **Persistent heuristics**: Saves/loads databases to avoid recomputation

## Installation

```bash
git clone https://github.com/NehaMalage/Rubiks_cube.git
cd Rubiks_cube
python -m venv venv
venv\Scripts\activate # on Windows and source venv/bin/activate on MAC
pip install -r requirements.txt
python rubiks.py
```

## How It Works

1. Generates a solved cube state
2. Builds or loads a heuristic database of optimal move counts
3. Shuffles the cube with random moves
4. Uses DFS with heuristic pruning to find the solution
5. Verifies the solution by applying moves step-by-step

**Move Types:**
- **Horizontal (`h`)**: Rotate rows left/right
- **Vertical (`v`)**: Rotate columns up/down  
- **Side (`s`)**: Rotate faces affecting top/left/right/bottom

## Configuration

Modify constants at the top of `rubiks.py` as per your requirement:

```python
NUMBER_OF_BLOCKS = 3      # Cube size (3 for 3x3x3)
MAX_SHUFFLE = 6           # Maximum shuffle moves
MIN_SHUFFLE = 5           # Minimum shuffle moves  
MAX_MOVES = 5             # Heuristic database depth
MAX_SOLVE_DEPTH = 20      # Maximum solving depth
```

## Algorithm Details

**Heuristic Database:**
- Precomputes optimal move counts using BFS
- Provides lower bound estimates for pruning
- Stored as a dictionary(JSON) for persistence

**Search Strategy:**
- Iterative deepening DFS with heuristic pruning
- Cycle detection and move ordering optimization
- Moves represented as `(move_type, row_or_col, direction)`



## File Structure

```
├── rubiks.py          # Main implementation
├── heuristic_cubesize.json          # Auto-generated database on running the code
├── requirements.txt          # Dependencies
└── README.md                
```

## License

This project is for educational purposes.

