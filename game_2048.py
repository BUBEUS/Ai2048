import random
import numpy as np

class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.reset()

    def reset(self):
        """Resetuje grę i zwraca początkowy stan."""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self._add_random_tile()
        self._add_random_tile()
        return self.board.copy()

    def _add_random_tile(self):
        """Dodaje losowy kafelek (2 lub 4) na pustym polu."""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if not empty_cells:
            return False
        y, x = random.choice(empty_cells)
        self.board[y, x] = 4 if random.random() < 0.1 else 2
        return True

    def _compress(self, row):
        """Przesuwa niezerowe elementy na początek listy."""
        new_row = [i for i in row if i != 0]
        new_row += [0] * (self.size - len(new_row))
        return new_row

    def _merge(self, row):
        """Łączy klocki. Zwraca (zmieniony_wiersz, zdobyte_punkty)."""
        points = 0
        for i in range(self.size - 1):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                points += row[i]  # Tylko obliczamy, NIE dodajemy do self.score
                row[i + 1] = 0
        return row, points

    def _move_row_left(self, row):
        """Wykonuje pełną sekwencję: compress -> merge -> compress."""
        curr_row = list(row)
        curr_row = self._compress(curr_row)
        curr_row, points = self._merge(curr_row)
        curr_row = self._compress(curr_row)
        return curr_row, points

    def _calculate_move_result(self, direction):
        """Oblicza deterministyczny wynik ruchu (bez losowania)."""
        rotations = {'left': 0, 'up': 1, 'right': 2, 'down': 3}
        if direction not in rotations: raise ValueError("Błąd kierunku")
        k = rotations[direction]

        # Operujemy na widoku (szybkie, brak deepcopy)
        board_working = np.rot90(self.board, k=k)
        new_rows = []
        step_reward = 0

        for row in board_working:
            processed, points = self._move_row_left(row)
            new_rows.append(processed)
            step_reward += points

        new_board = np.array(new_rows)
        new_board = np.rot90(new_board, k=-k)
        changed = not np.array_equal(self.board, new_board)

        return new_board, step_reward, changed

    def move_without_random(self, direction):
        """Symulacja ruchu dla AI: przesuwa, ale NIE dodaje kafelka."""
        new_board, reward, changed = self._calculate_move_result(direction)
        self.board = new_board
        # W symulacji nie aktualizujemy self.score, bo to tylko 'wyobrażenie'
        return self.board, reward, changed

    def move(self, direction):
        """Ruch w prawdziwej grze: przesuwa + dodaje losowy kafelek."""
        new_board, reward, changed = self._calculate_move_result(direction)
        self.board = new_board

        if changed:
            self.score += reward
            self._add_random_tile()

        done = not self._can_move()
        return self.board.copy(), reward, done, changed

    def _can_move(self):
        """Sprawdza czy możliwy jest jakikolwiek ruch."""
        if np.any(self.board == 0):
            return True
        for y in range(self.size):
            for x in range(self.size - 1):
                if self.board[y, x] == self.board[y, x + 1]:
                    return True
                if self.board[x, y] == self.board[x + 1, y]:
                    return True
        return False


    def can_move_direction(self, direction):
        """Szybkie sprawdzenie czy ruch jest możliwy (bez kopiowania)."""
        rotations = {'left': 0, 'up': 1, 'right': 2, 'down': 3}
        k = rotations[direction]
        # rot90 zwraca widok (view) - operacja O(1), brak alokacji pamięci
        board_view = np.rot90(self.board, k=k)

        for row in board_view:
            prev = -1
            for x in row:
                if x == 0:
                    prev = 0 # Oznaczamy, że widzieliśmy puste pole
                elif prev == 0:
                    return True # Liczba po zerze -> przesunięcie
                elif x == prev:
                    return True # Dwie takie same obok siebie -> łączenie
                else:
                    prev = x
        return False

    def get_valid_moves(self):
        """Zwraca listę dostępnych ruchów używając szybkiego sprawdzania."""
        return [d for d in ['left', 'right', 'up', 'down']
                if self.can_move_direction(d)]

    def print_board(self):
        print(self.board)
        print(f"Score: {self.score}\n")




# ===== Przykładowe uruchomienie =====
if __name__ == "__main__":
    game = Game2048(size=4)
    game.print_board()

    while True:
        move = input("Ruch (w/s/a/d): ").strip().lower()
        mapping = {'w': 'up', 's': 'down', 'a': 'left', 'd': 'right'}
        if move not in mapping:
            print("Nieprawidłowy ruch!")
            continue

        # ZMIANA: Odbieramy 4 wartości zamiast 3
        board, reward, done, changed = game.move(mapping[move])
        game.print_board()
        if done:
            print("Koniec gry!")
            break

