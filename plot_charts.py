import pandas as pd
import matplotlib.pyplot as plt
import os

# --- KONFIGURACJA ---
CSV_FILE = "training_history.csv"
WINDOW_SIZE = 50  # Jak bardzo wygładzać wykres (średnia z 50 gier)
DPI = 120  # Rozdzielczość wykresu

def generate_charts():
    if not os.path.exists(CSV_FILE):
        print(f"Błąd: Nie znaleziono pliku {CSV_FILE}")
        return

    print("Wczytywanie danych...")
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Błąd odczytu CSV: {e}")
        return

    if df.empty:
        print("Plik CSV jest pusty.")
        return

    # Ustawienie stylu wykresu na "ładny"
    plt.style.use('seaborn-v0_8-darkgrid')

    # Tworzymy płótno z 3 wykresami jeden pod drugim
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

    # --- WYKRES 1: WYNIKI I MAX KLOCEK (Dual Axis) ---
    # Obliczamy średnią kroczącą, żeby wykres był gładki i czytelny
    rolling_score = df['Score'].rolling(window=WINDOW_SIZE).mean()
    rolling_max = df['MaxTile'].rolling(window=WINDOW_SIZE).mean()

    color = 'tab:blue'
    ax1.set_title(f'Postęp Treningu (Średnia z {WINDOW_SIZE} gier)', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Średni Wynik (Score)', color=color, fontsize=12)
    ax1.plot(df['Episode'], rolling_score, color=color, linewidth=2, label='Avg Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)

    # Druga oś Y dla MaxTile
    ax1_twin = ax1.twinx()
    color = 'tab:orange'
    ax1_twin.set_ylabel('Średni Max Klocek', color=color, fontsize=12)
    ax1_twin.plot(df['Episode'], rolling_max, color=color, linewidth=2, linestyle='--', label='Avg MaxTile')
    ax1_twin.tick_params(axis='y', labelcolor=color)

    # Dodanie legendy dla obu osi
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    # --- WYKRES 2: EWOLUCJA MÓZGU (FAZA NORMALNA) ---
    # Pokazujemy tylko najważniejsze wagi strategiczne
    ax2.set_title('Ewolucja Strategii (Faza NORMAL)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Wartość Wagi', fontsize=12)

    # Wybieramy kluczowe wagi
    weights_to_plot = {
        'N_Snake': ('green', 'Snake (Wąż)'),
        'N_Corner': ('purple', 'Corner (Róg)'),
        'N_Neigh': ('brown', 'Neighbor (Sąsiedzi)'),
        'N_Merge': ('red', 'Merge (Łączenie)'),
        'N_Empty': ('gray', 'Empty (Miejsce)')
    }

    for col, (color, label) in weights_to_plot.items():
        if col in df.columns:
            ax2.plot(df['Episode'], df[col], color=color, label=label, linewidth=2, alpha=0.9)

    ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # --- WYKRES 3: EWOLUCJA INSTRYNKTU PRZETRWANIA (FAZA PANIKI) ---
    ax3.set_title('Instynkt Przetrwania (Faza PANIC)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epizod', fontsize=12)
    ax3.set_ylabel('Wartość Wagi', fontsize=12)

    panic_weights_to_plot = {
        'P_Max': ('orange', 'MaxTile (Obrona Króla)'),
        'P_Snake': ('green', 'Snake (Struktura)'),
        'P_Empty': ('gray', 'Empty (Ucieczka)')
    }

    for col, (color, label) in panic_weights_to_plot.items():
        if col in df.columns:
            ax3.plot(df['Episode'], df[col], color=color, label=label, linewidth=2, alpha=0.9)

    ax3.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    ax3.grid(True, linestyle='--', alpha=0.7)

    # Zapis i pokazanie
    plt.tight_layout()
    output_file = "wykres_treningu.png"
    plt.savefig(output_file, dpi=DPI)
    print(f"--> Wykres zapisano jako: {output_file}")
    plt.show()

if __name__ == "__main__":
    generate_charts()