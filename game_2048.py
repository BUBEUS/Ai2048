import random
import numpy as np

class Game2048:
    """
    Główna klasa logiki gry 2048.
    
    Zarządza stanem planszy, wykonuje ruchy, dodaje losowe kafelki
    i sprawdza warunki końca gry.

    Attributes:
        size (int): Rozmiar planszy (domyślnie 4x4).
        board (np.ndarray): Macierz NxN reprezentująca planszę gry.
        score (int): Aktualny wynik punktowy gry.
    """
    def __init__(self, size=4):
        self.size = size
        self.reset()

    def reset(self):
        """
        Resetuje grę do stanu początkowego.

        Zeruje planszę i wynik, a następnie dodaje dwa losowe kafelki startowe.

        Returns:
            np.ndarray: Kopia planszy (stan początkowy).
        """
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self._add_random_tile()
        self._add_random_tile()
        return self.board.copy()

    def _add_random_tile(self):
        """
        Dodaje losowy kafelek (2 lub 4) na losowym pustym polu.

        Prawdopodobieństwo: 90% na '2', 10% na '4'.

        Returns:
            bool: True jeśli dodano kafelek, False jeśli brak miejsca.
        """
        empty_cells = list(zip(*np.where(self.board == 0)))
        if not empty_cells:
            return False
        y, x = random.choice(empty_cells)
        self.board[y, x] = 4 if random.random() < 0.1 else 2
        return True

    def _compress(self, row):
        """
        Przesuwa niezerowe elementy na początek listy (implementacja ruchu).

        Args:
            row (list): Wiersz planszy.

        Returns:
            list: Nowy wiersz z przesuniętymi elementami.
        """
        new_row = [i for i in row if i != 0]
        new_row += [0] * (self.size - len(new_row))
        return new_row

    def _merge(self, row):
        """
        Łączy sąsiadujące identyczne klocki w wierszu.

        Args:
            row (list): Wiersz po operacji compress.

        Returns:
            tuple: (zmieniony_wiersz, zdobyte_punkty)
        """
        points = 0
        for i in range(self.size - 1):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                points += row[i]  
                row[i + 1] = 0
        return row, points

    def _move_row_left(self, row):
        """
        Wykonuje pełną sekwencję ruchu w lewo dla jednego wiersza.

        Sekwencja: compress -> merge -> compress.

        Args:
            row (np.ndarray): Pojedynczy wiersz planszy.

        Returns:
            tuple: (przetworzony_wiersz, punkty_za_ten_wiersz)
        """
        curr_row = list(row)
        curr_row = self._compress(curr_row)
        curr_row, points = self._merge(curr_row)
        curr_row = self._compress(curr_row)
        return curr_row, points

    def _calculate_move_result(self, direction):
        """
        Oblicza deterministyczny wynik ruchu (bez losowania nowego kafelka).
        
        Używana zarówno przez grę właściwą, jak i symulacje AI.

        Args:
            direction (str): Kierunek ruchu ('left', 'up', 'right', 'down').

        Returns:
            tuple: (nowa_plansza, nagroda_za_ruch, czy_zaszla_zmiana)
        
        Raises:
            ValueError: Jeśli podano nieznany kierunek.
        """
        rotations = {'left': 0, 'up': 1, 'right': 2, 'down': 3}
        if direction not in rotations: raise ValueError("Błąd kierunku")
        k = rotations[direction]

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
        """
        Symulacja ruchu dla AI: przesuwa planszę, ale NIE dodaje losowego kafelka.

        Metoda służy do "wyobrażenia sobie" stanu po ruchu (Lookahead).

        Args:
            direction (str): Kierunek ruchu.

        Returns:
            tuple: (nowa_plansza, nagroda, czy_zmiana)
        """
        new_board, reward, changed = self._calculate_move_result(direction)
        self.board = new_board
        return self.board, reward, changed

    def move(self, direction):
        """
        Wykonuje ruch w prawdziwej grze.
        
        1. Przesuwa klocki i łączy je.
        2. Aktualizuje wynik.
        3. Dodaje losowy kafelek (jeśli ruch był ważny).
        4. Sprawdza czy gra się skończyła.

        Args:
            direction (str): Kierunek ruchu (w/s/a/d lub nazwy pełne).

        Returns:
            tuple: (kopia_planszy, nagroda, czy_koniec, czy_zmieniono)
        """
        new_board, reward, changed = self._calculate_move_result(direction)
        self.board = new_board

        if changed:
            self.score += reward
            self._add_random_tile()

        done = not self._can_move()
        return self.board.copy(), reward, done, changed

    def _can_move(self):
        """
        Sprawdza czy możliwy jest jakikolwiek ruch na planszy.

        Returns:
            bool: True jeśli gra może trwać dalej, False jeśli Game Over.
        """
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
        """
        Szybkie sprawdzenie czy ruch w danym kierunku jest legalny.
        
        Zoptymalizowane pod kątem wydajności (używa widoków numpy bez kopiowania).

        Args:
            direction (str): Kierunek do sprawdzenia.

        Returns:
            bool: True jeśli ruch spowoduje zmianę stanu planszy.
        """
        rotations = {'left': 0, 'up': 1, 'right': 2, 'down': 3}
        k = rotations[direction]
        board_view = np.rot90(self.board, k=k)

        for row in board_view:
            prev = -1
            for x in row:
                if x == 0:
                    prev = 0 
                elif prev == 0:
                    return True 
                elif x == prev:
                    return True 
                else:
                    prev = x
        return False

    def get_valid_moves(self):
        """
        Zwraca listę wszystkich legalnych ruchów w danym stanie.

        Returns:
            list[str]: Lista kierunków np. ['left', 'up'].
        """
        return [d for d in ['left', 'right', 'up', 'down']
                if self.can_move_direction(d)]

    def print_board(self):
        """Wypisuje obecny stan planszy i wynik w konsoli."""
        print(self.board)
        print(f"Score: {self.score}\n")




if __name__ == "__main__":
    game = Game2048(size=4)
    game.print_board()

    while True:
        move = input("Ruch (w/s/a/d): ").strip().lower()
        mapping = {'w': 'up', 's': 'down', 'a': 'left', 'd': 'right'}
        if move not in mapping:
            print("Nieprawidłowy ruch!")
            continue

        board, reward, done, changed = game.move(mapping[move])
        game.print_board()
        if done:
            print("Koniec gry!")
            break