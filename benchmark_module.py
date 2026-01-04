import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
import numpy as np
import os
import time
from collections import Counter
from game_2048 import Game2048
from ai_player import AIPlayer
import concurrent.futures

try:
    plt.style.use('seaborn-v0_8-white')
except:
    pass

GRID_COLOR = '#bbada0' 
CELL_COLORS = {
    0: '#cdc1b4', 2: '#eee4da', 4: '#ede0c8', 8: '#f2b179',
    16: '#f59563', 32: '#f67c5f', 64: '#f65e3b', 128: '#edcf72',
    256: '#edcc61', 512: '#edc850', 1024: '#edc53f', 2048: '#edc22e',
    'super': '#3c3a32'
}
TEXT_COLORS = { 2: '#776e65', 4: '#776e65', 'other': '#f9f6f2'}

def run_single_game(weights_normal, weights_panic, log_table):
    """
    Uruchamia pojedynczą grę w izolowanym procesie.
    
    Zbiera heatmapę (częstotliwość odwiedzin pól) dla każdego ruchu.

    Args:
        weights_normal (np.ndarray): Wagi dla trybu normalnego.
        weights_panic (np.ndarray): Wagi dla trybu paniki.
        log_table (np.ndarray): Tablica prekomputowanych logarytmów (nieużywana w tej wersji, ale zachowana).

    Returns:
        tuple: (wynik, max_kafelek, plansza_końcowa, lokalna_heatmapa, liczba_ruchów)
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

        local_heatmap += game.board
        moves_in_game += 1

    return game.score, np.max(game.board), game.board, local_heatmap, moves_in_game

class Benchmark:
    """
    Moduł testujący wydajność AI na dużej próbie gier.
    
    Generuje raporty graficzne z:
    1. Heatmapą (które pola są najważniejsze).
    2. Statystykami (rozkład max klocka).
    3. Wizualizacją najlepszej i najgorszej gry.

    Attributes:
        ai (AIPlayer): Instancja agenta AI do przetestowania.
        games_to_run (int): Liczba gier do symulacji (domyślnie 1000).
        output_folder (str): Folder na wyniki.
    """
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
        """Generuje unikalną nazwę pliku wyjściowego (inkrementacja licznika)."""
        i = 1
        while True:
            base = os.path.join(self.output_folder, f"{self.output_prefix}-{i:02d}")
            if not os.path.exists(base + ".png"):
                return base
            i += 1

    def run(self, update_gui_callback=None):
        """
        Uruchamia główną pętlę benchmarku przy użyciu wielowątkowości (ProcessPoolExecutor).

        Args:
            update_gui_callback (function, optional): Funkcja zwrotna do aktualizacji paska postępu w GUI.
        """
        scores = []
        max_tiles = []

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
                score, max_val, final_board, local_heatmap, moves_cnt = future.result()

                scores.append(score)
                max_tiles.append(max_val)

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

        heatmap_avg = global_heatmap_sum / total_moves_count if total_moves_count > 0 else global_heatmap_sum

        print(f"--> Benchmark zakończony w {duration:.2f}s. Średnia: {avg_score:.0f}")
        print(f"--> Przeanalizowano łącznie {total_moves_count} ruchów.")

        base_filename = self.get_next_base_filename()
        main_plot_file = base_filename + ".png"
        best_plot_file = base_filename + "-BEST.png"
        worst_plot_file = base_filename + "-WORST.png"

        self._generate_main_plot(scores, max_tiles, heatmap_avg, min_score, max_score, main_plot_file)

        print("--> Zapisywanie plansz ekstremalnych (High Quality)...")
        self._save_board_image(best_board, max_score, "NAJLEPSZY WYNIK", best_plot_file)
        self._save_board_image(worst_board, min_score, "NAJGORSZY WYNIK", worst_plot_file)

        try:
            if os.name == 'nt': os.startfile(main_plot_file)
            else: os.system(f"xdg-open {main_plot_file}")
        except: pass

    def _save_board_image(self, board, score, title_prefix, filename):
        """
        Rysuje planszę używając prostokątów (patches), aby imitować wygląd gry.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_axis_off()

        padding = 0.1
        cell_size = 1.0
        grid_width = 4 * cell_size + 5 * padding

        ax.set_xlim(0, grid_width)
        ax.set_ylim(0, grid_width)
        bg_rect = patches.Rectangle((0, 0), grid_width, grid_width, linewidth=0, facecolor=GRID_COLOR)
        ax.add_patch(bg_rect)

        for row in range(4):
            for col in range(4):
                val = int(board[row, col])

                x_pos = padding + col * (cell_size + padding)
                y_pos = padding + (3 - row) * (cell_size + padding)

                bg_color = CELL_COLORS.get(val, CELL_COLORS['super'])

    
                rect = patches.Rectangle((x_pos, y_pos), cell_size, cell_size,
                                         linewidth=0, facecolor=bg_color)
                ax.add_patch(rect)

                if val > 0:
                    text_color = TEXT_COLORS.get(val, TEXT_COLORS['other'])

                    font_size = 35
                    if val > 100: font_size = 30
                    if val > 1000: font_size = 24
                    if val > 10000: font_size = 20

                    ax.text(x_pos + cell_size/2, y_pos + cell_size/2, str(val),
                            ha='center', va='center', color=text_color,
                            fontsize=font_size, fontweight='bold', fontfamily='sans-serif')

        plt.title(f"{title_prefix}\nWynik: {score}", fontsize=18, fontweight='bold', color='#776e65', pad=20)

        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#faf8ef')
        plt.close(fig)

    def _generate_main_plot(self, scores, max_tiles, heatmap_avg, min_score, max_score, filename):
        """Generuje główny raport zbiorczy (GridSpec: Heatmapa, Statystyki, Wagi)."""
        fig = plt.figure(figsize=(16, 12), facecolor='#faf8ef')
        gs = fig.add_gridspec(3, 3, wspace=0.3, hspace=0.3)

        ax_map = fig.add_subplot(gs[0:2, 0:2])

        heatmap_log = np.log2(heatmap_avg + 1)

        im = ax_map.imshow(heatmap_log, cmap='inferno', interpolation='nearest')
        ax_map.set_title("Analiza Strategii (Średnia Wartość Pola na Ruch)", fontsize=16, fontweight='bold', color='#776e65', pad=15)

        ax_map.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax_map.set_yticks(np.arange(-.5, 4, 1), minor=True)
        ax_map.grid(which='minor', color='white', linestyle='-', linewidth=2)
        ax_map.tick_params(which='minor', bottom=False, left=False)
        ax_map.axis('off')

        for i in range(4):
            for j in range(4):
                val = heatmap_avg[i, j]
                val_str = f"{val:.1f}"
                text = ax_map.text(j, i, val_str, ha="center", va="center",
                                   color='white', fontsize=13, fontweight='bold')
                text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

        cbar = plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
        cbar.set_label('Skala Logarytmiczna (Log2)', rotation=270, labelpad=20, fontsize=12)
        cbar.outline.set_linewidth(0)


        ax_stats = fig.add_subplot(gs[0:2, 2])
        ax_stats.axis('off')

        counts = Counter(max_tiles)
        total = len(max_tiles)
        sorted_tiles = sorted(counts.keys(), reverse=True)

        text_str = "DYSTRYBUCJA MAX KLOCKA:\n"
        text_str += "━" * 25 + "\n"
        for tile in sorted_tiles:
            if tile >= 128: 
                count = counts[tile]
                perc = (count / total) * 100

                text_str += f"{tile:<5} : {count:>4} gier ({perc:>5.1f}%)\n"

        avg_score = sum(scores) / len(scores)

        text_str += "\n\nSTATYSTYKI PUNKTOWE:\n"
        text_str += "━" * 25 + "\n"
        text_str += f"Średnia : {avg_score:>8.0f}\n"
        text_str += f"Max     : {max_score:>8}\n"
        text_str += f"Min     : {min_score:>8}\n"

 
        props = dict(boxstyle='round,pad=1', facecolor='#f9f6f2', edgecolor='#bbada0', linewidth=2)
        ax_stats.text(0.5, 0.5, text_str, transform=ax_stats.transAxes, fontsize=13,
                      verticalalignment='center', horizontalalignment='center',
                      fontfamily='monospace', bbox=props, color='#776e65')

        ax_weights = fig.add_subplot(gs[2, 0:3]) 
        ax_weights.axis('off')

        w_norm = self.ai.weights_normal
        w_panic = self.ai.weights_panic

        labels = ["Empty", "MaxTile", "Gradient", "Merge", "Corner", "Neighbor"]

        def format_weights_nice(weights):
            parts = []
            for i, val in enumerate(weights):
                name = labels[i] if i < len(labels) else f"W{i}"
                parts.append(f"{name}: {val:.2f}")
            return "   ".join(parts)

        weights_text = "AKTUALNA KONFIGURACJA WAG AI\n"
        weights_text += "━" * 60 + "\n"
        weights_text += f"Tryb NORMAL (Budowanie) :  {format_weights_nice(w_norm)}\n"
        weights_text += f"Tryb PANIC  (Ratunek)   :  {format_weights_nice(w_panic)}"

        props_w = dict(boxstyle='round,pad=0.8', facecolor='#edf2f7', edgecolor='#cbd5e0', linewidth=2)
        ax_weights.text(0.5, 0.5, weights_text, transform=ax_weights.transAxes,
                        ha="center", va="center", fontsize=12, fontfamily='monospace',
                        bbox=props_w, color='#2d3748')

        plt.tight_layout()
        print(f"--> Zapisywanie głównego wykresu do: {filename}")
        plt.savefig(filename, dpi=120, facecolor='#faf8ef')
        plt.close(fig)