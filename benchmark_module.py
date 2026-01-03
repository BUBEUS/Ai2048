import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import os
import time
from collections import Counter
from game_2048 import Game2048
from ai_player import AIPlayer
import concurrent.futures

# --- KOLORY DO WIZUALIZACJI PLANSZ ---
CELL_COLORS = {
    0: '#cdc1b4', 2: '#eee4da', 4: '#ede0c8', 8: '#f2b179',
    16: '#f59563', 32: '#f67c5f', 64: '#f65e3b', 128: '#edcf72',
    256: '#edcc61', 512: '#edc850', 1024: '#edc53f', 2048: '#edc22e',
    'super': '#3c3a32'
}
TEXT_COLORS = { 2: '#776e65', 4: '#776e65', 'other': 'white'}

def run_single_game(weights_normal, weights_panic, log_table):
    """
    Uruchamia grę i zbiera heatmapę z KAŻDEGO RUCHU.
    """
    ai = AIPlayer()
    ai.weights_normal = weights_normal
    ai.weights_panic = weights_panic
    ai.log_table = log_table
    ai.epsilon = 0
    ai.alpha = 0

    game = Game2048()
    sim_game = Game2048()

    done = False
    state = game.board.copy()

    # --- NOWOŚĆ: Lokalna heatmapa dla tej jednej gry ---
    local_heatmap = np.zeros((4, 4), dtype=float)
    moves_in_game = 0

    while not done:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            break

        best_move, best_v = None, -float('inf')
        for move in valid_moves:
            sim_game.board = state
            next_s, _, _ = sim_game.move_without_random(move)
            v = ai.get_expected_value(next_s)
            if v > best_v:
                best_v = v
                best_move = move

        state, _, done, _ = game.move(best_move)

        # --- REJESTRACJA STANU (Snapshot strategii) ---
        # Dodajemy planszę do heatmapy PO wykonaniu ruchu.
        # Dzięki temu widzimy "zdrowy" stan gry, a nie tylko game over.
        local_heatmap += game.board
        moves_in_game += 1

    # Zwracamy też local_heatmap i liczbę ruchów
    return game.score, np.max(game.board), game.board, local_heatmap, moves_in_game

class Benchmark:
    def __init__(self, ai_player):
        self.ai = ai_player
        self.games_to_run = 1000
        self.output_prefix = "avg1k"
        self.output_folder = "benchmarks"

        if not os.path.exists(self.output_folder):
            try:
                os.makedirs(self.output_folder)
            except OSError as e:
                print(f"Błąd tworzenia folderu: {e}")

    def get_next_base_filename(self):
        i = 1
        while True:
            base = os.path.join(self.output_folder, f"{self.output_prefix}-{i:02d}")
            if not os.path.exists(base + ".png"):
                return base
            i += 1

    def run(self, update_gui_callback=None):
        scores = []
        max_tiles = []

        # Globalna suma ze wszystkich ruchów we wszystkich grach
        global_heatmap_sum = np.zeros((4, 4), dtype=float)
        total_moves_count = 0

        min_score = float('inf')
        worst_board = None
        max_score = -float('inf')
        best_board = None

        print(f"--> Rozpoczynam benchmark {self.games_to_run} gier (ANALIZA RUCH PO RUCHU)...")
        start_time = time.time()

        w_norm = self.ai.weights_normal
        w_panic = self.ai.weights_panic

        try:
            l_table = self.ai.log_table
        except AttributeError:
            l_table = np.zeros(66000, dtype=float)
            for i in range(1, 17):
                l_table[2**i] = float(i)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_single_game, w_norm, w_panic, l_table) for _ in range(self.games_to_run)]

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                # Odbieramy rozszerzone dane
                score, max_val, final_board, local_heatmap, moves_cnt = future.result()

                scores.append(score)
                max_tiles.append(max_val)

                # Agregacja danych "Live"
                global_heatmap_sum += local_heatmap
                total_moves_count += moves_cnt

                if score > max_score:
                    max_score = score
                    best_board = final_board.copy()

                if score < min_score:
                    min_score = score
                    worst_board = final_board.copy()

                if update_gui_callback and i % 10 == 0:
                    update_gui_callback(final_board, i + 1, self.games_to_run, score)

        duration = time.time() - start_time
        avg_score = sum(scores) / len(scores)

        # Obliczamy średnią wartość klocka NA RUCH
        # (Suma wartości klocków przez całą historię) / (Całkowita liczba ruchów)
        heatmap_avg = global_heatmap_sum / total_moves_count if total_moves_count > 0 else global_heatmap_sum

        print(f"--> Benchmark zakończony w {duration:.2f}s. Średnia: {avg_score:.0f}")
        print(f"--> Przeanalizowano łącznie {total_moves_count} ruchów.")

        base_filename = self.get_next_base_filename()
        main_plot_file = base_filename + ".png"
        best_plot_file = base_filename + "-BEST.png"
        worst_plot_file = base_filename + "-WORST.png"

        self._generate_main_plot(scores, max_tiles, heatmap_avg, min_score, max_score, main_plot_file)

        print("--> Zapisywanie plansz ekstremalnych...")
        self._save_board_image(best_board, max_score, "NAJLEPSZY WYNIK", best_plot_file)
        self._save_board_image(worst_board, min_score, "NAJGORSZY WYNIK", worst_plot_file)

        try:
            if os.name == 'nt': os.startfile(main_plot_file)
            else: os.system(f"xdg-open {main_plot_file}")
        except: pass

    def _save_board_image(self, board, score, title_prefix, filename):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_axis_off()

        table = plt.table(cellText=board, loc='center', cellLoc='center', bbox=[0, 0, 1, 1])

        for i in range(4):
            for j in range(4):
                val = int(board[i, j])
                cell = table[i, j]
                bg_color = CELL_COLORS.get(val, CELL_COLORS['super'])
                cell.set_facecolor(bg_color)
                text_color = TEXT_COLORS.get(val, TEXT_COLORS['other'])
                cell.get_text().set_color(text_color)

                font_size = 20
                if val > 1000: font_size = 16
                if val > 10000: font_size = 14
                cell.get_text().set_fontsize(font_size)
                cell.get_text().set_fontweight('bold')

                if val == 0: cell.get_text().set_text("")

        plt.title(f"{title_prefix}\nWynik: {score}", fontsize=16, fontweight='bold', pad=20)
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)

    def _generate_main_plot(self, scores, max_tiles, heatmap_avg, min_score, max_score, filename):
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 3)

        # 1. MAPA CIEPŁA (AVERAGE PER MOVE)
        ax_map = fig.add_subplot(gs[0:2, 0:2])

        # Logarytmowanie dla czytelności (wartości będą mniejsze niż wcześniej, bo dzielimy przez ilość ruchów)
        # ale relacje między polami pozostaną te same.
        heatmap_log = np.log2(heatmap_avg + 1)

        im = ax_map.imshow(heatmap_log, cmap='magma', interpolation='nearest')
        ax_map.set_title("Średnie Wartości Klocków (ANALIZA WSZYSTKICH RUCHÓW)", fontsize=14, fontweight='bold')
        ax_map.axis('off')

        for i in range(4):
            for j in range(4):
                val = heatmap_avg[i, j]
                # Formatowanie wartości (teraz mogą być mniejsze, więc jedno miejsce po przecinku)
                val_str = f"{val:.1f}"

                text = ax_map.text(j, i, val_str, ha="center", va="center",
                                   color='white', fontsize=12, fontweight='bold')
                text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])

        cbar = plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
        cbar.set_label('Skala Logarytmiczna', rotation=270, labelpad=15)

        # 2. STATYSTYKI
        ax_stats = fig.add_subplot(gs[0:2, 2])
        ax_stats.axis('off')

        counts = Counter(max_tiles)
        total = len(max_tiles)
        sorted_tiles = sorted(counts.keys())

        text_str = "Dystrybucja MaxTile:\n"
        text_str += "-" * 20 + "\n"
        for tile in sorted_tiles:
            if tile >= 256:
                count = counts[tile]
                perc = (count / total) * 100
                text_str += f"{tile}:  {count} ({perc:.1f}%)\n"

        avg_score = sum(scores) / len(scores)

        text_str += "\n\nSTATYSTYKI PUNKTOWE:\n"
        text_str += "-" * 20 + "\n"
        text_str += f"Średnia: {avg_score:.0f}\n"
        text_str += f"Max:     {max_score}\n"
        text_str += f"Min:     {min_score}\n"

        ax_stats.text(0.1, 0.9, text_str, transform=ax_stats.transAxes, fontsize=12, verticalalignment='top', fontfamily='monospace')

        # 3. WAGI
        ax_weights = fig.add_subplot(gs[2, 0:2])
        ax_weights.axis('off')

        w_norm = self.ai.weights_normal
        w_panic = self.ai.weights_panic

        labels = ["Empty", "MaxTile", "Gradient", "Merge", "Corner", "Neighbor"]
        def format_weights(weights):
            s = ""
            for i, val in enumerate(weights):
                name = labels[i] if i < len(labels) else f"W{i}"
                s += f"{name}: {val:.2f} | "
            return s.strip(" | ")

        weights_text = "AKTUALNE WAGI:\n"
        weights_text += f"NORMAL: {format_weights(w_norm)}\n"
        weights_text += f"PANIC : {format_weights(w_panic)}"

        ax_weights.text(0.5, 0.5, weights_text, transform=ax_weights.transAxes,
                        ha="center", va="center", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))

        plt.tight_layout()
        print(f"--> Zapisywanie głównego wykresu do: {filename}")
        plt.savefig(filename, dpi=120)
        plt.close(fig)