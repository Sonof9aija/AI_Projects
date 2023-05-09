from collections import deque



def solve_sudoku(grid):
    """
    Solves the given Sudoku grid using BFS.
    Returns the solved grid if possible, otherwise returns None.
    """
    queue = deque([(grid, 0, 0)])
    while queue:
        current_grid, row, col = queue.popleft()
        if row == 9:
            return current_grid
        next_row, next_col = (row + 1, 0) if col == 8 else (row, col + 1)
        if current_grid[row][col] != 0:
            queue.append((current_grid, next_row, next_col))
            continue
        for i in range(1, 10):
            if is_valid(current_grid, row, col, i):
                new_grid = [list(row) for row in current_grid] 
                new_grid[row][col] = i
                queue.append((new_grid, next_row, next_col))
    return None

def is_valid(grid, row, col, value):
    for i in range(9):
        if grid[row][i] == value or grid[i][col] == value:
            return False
    box_row, box_col = (row // 3) * 3, (col // 3) * 3 
    for i in range(box_row, box_row + 3):
        for j in range(box_col, box_col + 3):
            if grid[i][j] == value: 
                return False
    return True

Tgrid = [    [0, 0, 5, 0, 0, 4, 0, 0, 9],
    [0, 4, 0, 0, 2, 0, 0, 7, 0],
    [2, 0, 0, 1, 0, 0, 3, 0, 0],
    [8, 0, 0, 6, 0, 0, 5, 0, 0],
    [0, 6, 0, 0, 3, 0, 0, 2, 0],
    [0, 0, 3, 0, 0, 2, 0, 0, 8],
    [0, 0, 7, 0, 0, 1, 0, 0, 2],
    [0, 3, 0, 0, 5, 0, 0, 4, 0],
    [1, 0, 0, 3, 0, 0, 8, 0, 0]
]

answer = solve_sudoku(Tgrid)
for i in range(len(answer)):
    temp = ''
    for j in range(len(answer[i])):
        temp += str(answer[i][j])
        temp += " "
        if j == 2 or j == 5:
            temp += "|"
            temp += " "
    print(temp)
    if i == 2 or i == 5:
        print('______________________')
