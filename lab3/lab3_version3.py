import random
import time
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import numpy as np

class Maze:
    """
    Class representing a maze with methods for generation and solving.
    The maze is represented as a 2D array where:
    -1 = wall
    0 = passage
    1+ = visited by a specific thread
    """
    def __init__(self, width: int, height: int):
        """
        Initialize a new maze with the given dimensions.

        @param width: Width of the maze (number of cells)
        @param height: Height of the maze (number of cells)
        """
        self.width = width
        self.height = height
        self.maze = [[-1 for _ in range(width)] for _ in range(height)]
        self.start_pos: Tuple[int, int] = (0, 0)
        self.end_pos: Tuple[int, int] = (height - 1, width - 1)

    def generate_maze(self) -> None:
        """
        Generate a random maze using Depth-First Search algorithm.
        This ensures that every cell in the maze is reachable.
        """
        for i in range(self.height):
            for j in range(self.width):
                self.maze[i][j] = -1

        start_x, start_y = 1, 1
        self.maze[start_x][start_y] = 0

        stack: List[Tuple[int, int]] = [(start_x, start_y)]

        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]

        while stack:
            current_x, current_y = stack[-1]

            neighbors = []
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = current_x + dx, current_y + dy
                if 0 < nx < self.height - 1 and 0 < ny < self.width - 1 and self.maze[nx][ny] == -1:
                    neighbors.append((nx, ny, dx // 2, dy // 2))

            if neighbors:
                next_x, next_y, wall_x, wall_y = neighbors[0]

                self.maze[current_x + wall_x][current_y + wall_y] = 0

                self.maze[next_x][next_y] = 0

                stack.append((next_x, next_y))
            else:
                stack.pop()

        # Create entrance and exit
        self.start_pos = (0, 1)
        self.end_pos = (self.height - 1, self.width - 2)
        self.maze[self.start_pos[0]][self.start_pos[1]] = 0
        self.maze[self.end_pos[0]][self.end_pos[1]] = 0

    def solve_sequential(self) -> List[Tuple[int, int]]:
        """
        Solve the maze using a sequential breadth-first search algorithm.

        @return: List of coordinates representing the path from start to end
        """
        # BFS to find path
        queue: List[Tuple[int, int]] = [self.start_pos]
        visited: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {self.start_pos: None}

        while queue:
            current = queue.pop(0)

            if current == self.end_pos:
                break

            x, y = current
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                next_pos: Tuple[int, int] = (nx, ny)

                if (0 <= nx < self.height and 0 <= ny < self.width and
                    self.maze[nx][ny] == 0 and next_pos not in visited):
                    queue.append(next_pos)
                    visited[next_pos] = current
                    # Mark as visited
                    self.maze[nx][ny] = 2

        # Reconstruct the path
        path: List[Tuple[int, int]] = []
        current = self.end_pos

        while current:
            path.append(current)
            current = visited.get(current)

        # Mark the path with a different value
        for x, y in path:
            if (x, y) != self.start_pos and (x, y) != self.end_pos:
                self.maze[x][y] = 3

        # Return the path in reverse order
        return path[::-1]

    def visualize(self, title: str = "Maze") -> None:
        """
        Visualize the maze using matplotlib.

        @param title: Title for the visualization
        """
        plt.figure(figsize=(10, 10))

        # Create a colored representation
        colored_maze = np.zeros((self.height, self.width, 3))

        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i][j] == -1:  # Wall
                    colored_maze[i, j] = [0, 0, 0]  # Black
                elif self.maze[i][j] == 0:  # Passage
                    colored_maze[i, j] = [1, 1, 1]  # White
                elif self.maze[i][j] == 2:  # Visited
                    colored_maze[i, j] = [0.7, 0.7, 1]  # Light blue
                elif self.maze[i][j] == 3:  # Path
                    colored_maze[i, j] = [0, 1, 0]  # Green

        # Mark start and end
        colored_maze[self.start_pos] = [1, 0, 0]  # Red
        colored_maze[self.end_pos] = [1, 0.5, 0]  # Orange

        plt.imshow(colored_maze)
        plt.title(title)
        plt.axis('off')
        plt.show()

def main():
    """
    Main function to demonstrate maze generation and solving.
    """
    # Set maze dimensions
    width, height = 31, 31

    # Create and generate maze
    maze = Maze(width, height)
    maze.generate_maze()

    # Visualize initial maze
    maze.visualize("Generated Maze")

    # Solve maze and measure time
    start_time = time.time()
    path = maze.solve_sequential()
    end_time = time.time()

    # Visualize solved maze
    maze.visualize(f"Solved Maze (Sequential) - Time: {end_time - start_time:.4f}s")

    print(f"Sequential solving time: {end_time - start_time:.4f} seconds")
    print(f"Path length: {len(path)}")

if __name__ == "__main__":
    main()
