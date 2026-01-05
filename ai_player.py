import numpy as np
import math
import pickle
import os


class AIPlayer:
    """
    Agent AI grający w 2048 przy użyciu uczenia ze wzmocnieniem (TD-Learning).
    
    Implementuje architekturę 'Dual-Brain' (Dwa Mózgi):
    1. **Normal Mode**: Strategia do budowania wysokich klocków (spokojna).
    2. **Panic Mode**: Strategia ratunkowa, gdy plansza jest prawie pełna.

    Attributes:
        weights_normal (np.ndarray): Wagi cech dla trybu normalnego.
        weights_panic (np.ndarray): Wagi cech dla trybu paniki.
        alpha (float): Współczynnik uczenia.
        gradients (list): Prekalkulowane maski gradientów (Snake).
    """
    def __init__(self):

        self.weights_normal = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.weights_panic  = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])





        self.alpha = 0.00025

        base_gradient = np.array([
            [15, 14, 13, 12],
            [ 8,  9, 10, 11],
            [ 7,  6,  5,  4],
            [ 0,  1,  2,  3]
        ])


        self.gradients = []
        for k in range(4):
            rot = np.rot90(base_gradient, k=k)
            self.gradients.append(rot)
            self.gradients.append(np.fliplr(rot))

    def get_features(self, board):
        """
        Ekstrahuje wektor cech stanu planszy.

        Oblicza 6 kluczowych metryk znormalizowanych do zakresu używalnego przez sieć.

        Args:
            board (np.ndarray): Aktualna plansza gry.

        Returns:
            np.ndarray: Wektor 6 cech:
                0. Empty Cells (znormalizowane)
                1. Max Tile (logarytmicznie)
                2. Gradient Score (dopasowanie do węża)
                3. Merges Available
                4. Corner Position (czy max jest w rogu?)
                5. Neighbor Bonus (czy duzi sąsiedzi są obok?)
        """

        board_log = np.zeros_like(board, dtype=float)
        mask = board > 0
        board_log[mask] = np.log2(board[mask])


        empty = len(board[board == 0]) / 16.0

        max_val_norm = np.max(board_log) / 16.0 

        gradient_scores = [np.sum(board_log * g) for g in self.gradients]
        best_gradient = max(gradient_scores) / 1000.0

        merges_h = (board[:, :-1] == board[:, 1:]) & (board[:, :-1] != 0)
        merges_v = (board[:-1, :] == board[1:, :]) & (board[:-1, :] != 0)
        merges = np.sum(merges_h) + np.sum(merges_v)
        merges_norm = min(merges / 10.0, 1.0)


        max_pos = np.argmax(board)
        r, c = divmod(max_pos, 4)
        is_corner = 0.0
        if (r == 0 or r == 3) and (c == 0 or c == 3):
            is_corner = 1.0

        neighbor_bonus = 0.0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 4 and 0 <= nc < 4:
                if board[nr, nc] > 0:
                    neighbor_bonus += board_log[nr, nc]

        neighbor_norm = min(neighbor_bonus / 40.0, 1.0)

        return np.array([empty, max_val_norm, best_gradient, merges_norm, is_corner, neighbor_norm])

    def _calculate_smoothness(self, board):
        """
        Oblicza karę za brak gładkości (różnice między sąsiadami).
        
        Wersja ZWEKTORYZOWANA (Błyskawiczna). Zamiast pętli, używamy operacji na całych macierzach.
        
        Args:
            board (np.ndarray): Plansza gry.
            
        Returns:
            float: Ujemna wartość (kara), im mniejsza różnica, tym bliżej 0.
        """
        """
        Wersja ZWEKTORYZOWANA (Błyskawiczna).
        Zamiast pętli, używamy operacji na całych macierzach.
        """

        mask = board > 0
        if not np.any(mask):
            return 0

        board_log = np.zeros_like(board, dtype=float)

        board_log[mask] = np.log2(board[mask])

        smoothness = 0


        diff_x = np.abs(board_log[:, :-1] - board_log[:, 1:])

        mask_x = (board[:, :-1] > 0) & (board[:, 1:] > 0)
        smoothness -= np.sum(diff_x[mask_x])

        diff_y = np.abs(board_log[:-1, :] - board_log[1:, :])
        mask_y = (board[:-1, :] > 0) & (board[1:, :] > 0)
        smoothness -= np.sum(diff_y[mask_y])

        return smoothness

    def _calculate_isolation_penalty(self, board):
        """
        Oblicza karę za izolowane klocki (brak identycznego sąsiada).

        Wersja ZWEKTORYZOWANA. Sprawdza izolację bez ani jednej pętli for.

        Args:
            board (np.ndarray): Plansza gry.

        Returns:
            int: Liczba klocków, które nie mają żadnego pasującego sąsiada.
        """
        """
        Wersja ZWEKTORYZOWANA.
        Sprawdza izolację bez ani jednej pętli for.
        """
        has_neighbor = np.zeros(board.shape, dtype=bool)


        match_h = (board[:, :-1] == board[:, 1:]) & (board[:, :-1] != 0)


        has_neighbor[:, :-1] |= match_h  
        has_neighbor[:, 1:]  |= match_h  


        match_v = (board[:-1, :] == board[1:, :]) & (board[:-1, :] != 0)

        has_neighbor[:-1, :] |= match_v 
        has_neighbor[1:, :]  |= match_v

        isolated_mask = (board != 0) & (~has_neighbor)

        return np.sum(isolated_mask)

    def evaluate(self, board):
        """
        Główna funkcja oceny stanu planszy (V-Value).
        
        Dynamicznie przełącza się między wagami NORMAL i PANIC w zależności
        od liczby pustych pól (< 4 to tryb paniki).
        
        Dodaje ręczne kary (Instynkty):
        - Smoothness (Gładkość)
        - Isolation (Izolacja)

        Args:
            board (np.ndarray): Plansza gry.

        Returns:
            float: Wartość oceny stanu (Score).
        """
        features = self.get_features(board)
        empty_cells_count = len(board[board == 0])

        if empty_cells_count < 4:

            base_score = np.dot(self.weights_panic, features)

            smoothness_weight = 2

            isolation_weight = 10
        else:

            base_score = np.dot(self.weights_normal, features)

            smoothness_weight = 1
            isolation_weight = 5

        smoothness = self._calculate_smoothness(board)
        isolation = self._calculate_isolation_penalty(board)

        final_score = base_score + \
                      (smoothness * smoothness_weight) - \
                      (isolation * isolation_weight)

        return final_score

    def update_weights(self, features_state, td_error):
        """
        Aktualizuje wagi sieci (TD-Learning Update) w zależności od fazy gry.
        
        Decyduje, który zestaw wag (Normal czy Panic) przyczynił się do wyniku
        i aktualizuje tylko ten zestaw.

        Args:
            features_state (np.ndarray): Wektor cech stanu przed ruchem.
            td_error (float): Błąd predykcji czasowej (Target - Prediction).
        """
        empty_ratio = features_state[0]
        empty_count = empty_ratio * 16.0

        td_error_clipped = np.clip(td_error, -10, 10)
        delta = self.alpha * td_error_clipped * features_state

        if empty_count < 3.99:
            self.weights_panic += delta
            self.weights_panic = np.maximum(self.weights_panic, 0.0)
        else:
            self.weights_normal += delta
            self.weights_normal = np.maximum(self.weights_normal, 0.0)

    def get_expected_value(self, board):
        """
        Oblicza wartość oczekiwaną stanu (Expectimax 1-step).
        
        Symuluje losowe pojawienie się kafelka (2 lub 4) w wolnych miejscach
        i uśrednia wynik oceny dla tych możliwości.

        Args:
            board (np.ndarray): Stan planszy PO ruchu gracza (przed pojawieniem się losowego kafelka).

        Returns:
            float: Uśredniona wartość oceny stanu.
        """
        empty_cells = list(zip(*np.where(board == 0)))
        if not empty_cells:
            return self.evaluate(board)

        if len(empty_cells) > 3:
            indices = np.random.choice(len(empty_cells), 3, replace=False)
            sample_cells = [empty_cells[i] for i in indices]
        else:
            sample_cells = empty_cells

        total_val = 0
        for r, c in sample_cells:
            # 2 (90%)
            board[r, c] = 2
            v2 = self.evaluate(board)
            # 4 (10%)
            board[r, c] = 4
            v4 = self.evaluate(board)
            board[r, c] = 0 

            total_val += (0.9 * v2 + 0.1 * v4)

        return total_val / len(sample_cells)


    def save_model(self, filename, episode_count):
        """
        Zapisuje stan AI (wagi obu mózgów) do pliku pickle.

        Args:
            filename (str): Ścieżka do pliku.
            episode_count (int): Numer aktualnego epizodu treningu.
        """
        data = {
            'weights_normal': self.weights_normal,
            'weights_panic': self.weights_panic,
            'episode': episode_count
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"--> Zapisano checkpoint (Epizod: {episode_count})")

    def load_model(self, filename):
        """
        Wczytuje stan AI z pliku. Obsługuje wsteczną kompatybilność.

        Args:
            filename (str): Ścieżka do pliku.

        Returns:
            int: Numer wczytanego epizodu (lub 0 jeśli błąd/brak pliku).
        """
        if not os.path.exists(filename):
            return 0
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

                if 'weights_normal' in data:
                    self.weights_normal = data['weights_normal']
                    self.weights_panic = data['weights_panic']
                else:
                    old_weights = data['weights']
                    self.weights_normal = old_weights.copy()
                    self.weights_panic = old_weights.copy()
                    print("Konwersja starego zapisu na Dual-Weights...")

                return data['episode']
        except Exception as e:
            print(f"Błąd odczytu zapisu: {e}")
            return 0