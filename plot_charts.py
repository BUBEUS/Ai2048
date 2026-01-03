import pandas as pd
import matplotlib.pyplot as plt
import os

# --- KONFIGURACJA ---
CSV_FILE = "training_history.csv"
WINDOW_SIZE = 50  # Wygładzanie (średnia krocząca)
DPI = 120         # Jakość obrazka

def setup_style():
    """Ustawia ładny styl wykresów"""
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('ggplot') # Fallback jeśli seaborn nie jest dostępny

def plot_scores(df):
    """Generuje Wykres 1: Wyniki i MaxTile"""
    print("Generowanie wykresu wyników...")

    # Przygotowanie danych (średnia krocząca)
    rolling_score = df['Score'].rolling(window=WINDOW_SIZE).mean()
    rolling_max = df['MaxTile'].rolling(window=WINDOW_SIZE).mean()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Tytuł
    ax1.set_title(f'Historia Wyników (Średnia z {WINDOW_SIZE} gier)', fontsize=14, fontweight='bold')

    # Oś X - Epizody (wymagane przez Ciebie)
    ax1.set_xlabel('Liczba Epizodów', fontsize=12, fontweight='bold')

    # Oś Y1 - Score (Lewa)
    color = 'tab:blue'
    ax1.set_ylabel('Średni Wynik (Score)', color=color, fontsize=12)
    ax1.plot(df['Episode'], rolling_score, color=color, linewidth=2, label='Avg Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)

    # Oś Y2 - MaxTile (Prawa)
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Średni Max Klocek', color=color, fontsize=12)
    ax2.plot(df['Episode'], rolling_max, color=color, linewidth=2, linestyle='--', label='Avg MaxTile')
    ax2.tick_params(axis='y', labelcolor=color)

    # Wspólna legenda
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    filename = "wykres_wynikow.png"
    plt.savefig(filename, dpi=DPI)
    plt.close() # Zwalniamy pamięć
    print(f"--> Zapisano: {filename}")

def plot_weights(df):
    """Generuje Wykres 2: Wagi (Normal i Panic)"""
    print("Generowanie wykresu wag...")

    # Tworzymy 2 podpłótna (jedno pod drugim) na jednym obrazku
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- CZĘŚĆ GÓRNA: WAGI NORMAL ---
    ax1.set_title('Ewolucja Wag - Tryb NORMAL (Budowanie)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Wartość Wagi', fontsize=12)

    weights_normal = {
        'N_Snake': ('green', 'Snake (Wąż)'),
        'N_Corner': ('purple', 'Corner (Róg)'),
        'N_Neigh': ('brown', 'Neighbor (Sąsiedzi)'),
        'N_Merge': ('red', 'Merge (Łączenie)'),
        'N_Empty': ('gray', 'Empty (Miejsce)'),
        'N_Max':   ('orange', 'MaxTile')
    }

    for col, (color, label) in weights_normal.items():
        if col in df.columns:
            ax1.plot(df['Episode'], df[col], color=color, label=label, linewidth=2, alpha=0.8)

    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, ncol=2)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # --- CZĘŚĆ DOLNA: WAGI PANIC ---
    ax2.set_title('Ewolucja Wag - Tryb PANIC (Ratunek)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Wartość Wagi', fontsize=12)

    # Oś X - Epizody (na samym dole)
    ax2.set_xlabel('Liczba Epizodów', fontsize=12, fontweight='bold')

    weights_panic = {
        'P_Snake': ('green', 'Snake'),
        'P_Corner': ('purple', 'Corner'),
        'P_Merge': ('red', 'Merge'),
        'P_Empty': ('gray', 'Empty'),
        'P_Max':   ('orange', 'MaxTile')
    }

    for col, (color, label) in weights_panic.items():
        if col in df.columns:
            ax2.plot(df['Episode'], df[col], color=color, label=label, linewidth=2, alpha=0.8)

    ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, ncol=2)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    filename = "wykres_wag.png"
    plt.savefig(filename, dpi=DPI)
    plt.close()
    print(f"--> Zapisano: {filename}")

def generate_charts():
    if not os.path.exists(CSV_FILE):
        print(f"Błąd: Nie znaleziono pliku {CSV_FILE}")
        return

    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Błąd odczytu CSV: {e}")
        return

    if df.empty:
        print("Plik CSV jest pusty.")
        return

    setup_style()

    # Generowanie dwóch oddzielnych plików
    plot_scores(df)
    plot_weights(df)

    # Otwieranie plików (opcjonalne)
    try:
        if os.name == 'nt':
            os.startfile("wykres_wynikow.png")
            os.startfile("wykres_wag.png")
    except:
        pass

if __name__ == "__main__":
    generate_charts()