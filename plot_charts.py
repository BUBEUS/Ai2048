import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_FILE = "training_history.csv"
WINDOW_SIZE = 50
DPI = 120

def setup_style():
    """Ustawia styl wykresów matplotlib."""
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('ggplot')

def clean_data(df):
    """
    Czyści nazwy kolumn ze spacji i konwertuje dane na liczby.
    """
    # 1. Usuń spacje z nazw kolumn (np. " Moves" -> "Moves")
    df.columns = df.columns.str.strip()

    # 2. Upewnij się, że kluczowe kolumny są liczbami
    numeric_cols = ['Score', 'MaxTile', 'Moves', 'Episode']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def plot_just_scores(df):
    print("Generowanie: Wykres Wyników...")
    if 'Score' not in df.columns:
        print("-> Brak kolumny Score.")
        return

    rolling_score = df['Score'].rolling(window=WINDOW_SIZE, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f'Historia Średniego Wyniku (Okno: {WINDOW_SIZE})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epizody', fontsize=12)
    ax.set_ylabel('Średni Wynik (Score)', fontsize=12)

    ax.plot(df['Episode'], rolling_score, color='tab:blue', linewidth=2, label='Avg Score')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("wykres_wyniki.png", dpi=DPI)
    plt.close()

def plot_weights(df):
    print("Generowanie: Wykres Wag...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- NORMAL ---
    ax1.set_title('Wagi NORMAL (Budowanie)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Wartość', fontsize=12)

    weights_normal = {
        'N_Snake': ('green', 'Snake'), 'N_Corner': ('purple', 'Corner'),
        'N_Neigh': ('brown', 'Neighbor'), 'N_Merge': ('red', 'Merge'),
        'N_Empty': ('gray', 'Empty'), 'N_Max': ('orange', 'MaxTile')
    }

    for col, (color, label) in weights_normal.items():
        if col in df.columns:
            ax1.plot(df['Episode'], df[col], color=color, label=label, linewidth=2, alpha=0.8)

    ax1.legend(loc='upper left', ncol=3, fontsize='small')
    ax1.grid(True)

    # --- PANIC ---
    ax2.set_title('Wagi PANIC (Ratunek)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Wartość', fontsize=12)
    ax2.set_xlabel('Epizody', fontsize=12)

    # POPRAWKA: Dodano brakujący P_Neigh, który jest w Twoich danych
    weights_panic = {
        'P_Snake': ('green', 'Snake'), 'P_Corner': ('purple', 'Corner'),
        'P_Merge': ('red', 'Merge'), 'P_Empty': ('gray', 'Empty'),
        'P_Max': ('orange', 'MaxTile'), 'P_Neigh': ('brown', 'Neighbor')
    }

    for col, (color, label) in weights_panic.items():
        if col in df.columns:
            ax2.plot(df['Episode'], df[col], color=color, label=label, linewidth=2, alpha=0.8)

    ax2.legend(loc='upper left', ncol=3, fontsize='small')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("wykres_wagi.png", dpi=DPI)
    plt.close()

def plot_tiles_split(df):
    print("Generowanie: Wykres Klocków (2 ploty)...")
    if 'MaxTile' not in df.columns:
        print("-> Brak kolumny MaxTile.")
        return

    rolling_peak_max = df['MaxTile'].rolling(window=WINDOW_SIZE, min_periods=1).max()
    rolling_avg_max = df['MaxTile'].rolling(window=WINDOW_SIZE, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax1.set_title(f'Rekordowy Max Klocek (Najlepszy w oknie {WINDOW_SIZE})', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Wartość Klocka', fontsize=12)
    ax1.plot(df['Episode'], rolling_peak_max, color='tab:red', linewidth=2, label='Rekord MaxTile')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.set_title(f'Średni Max Klocek (Stabilność)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Wartość Klocka', fontsize=12)
    ax2.set_xlabel('Epizody', fontsize=12)
    ax2.plot(df['Episode'], rolling_avg_max, color='tab:orange', linewidth=2, label='Średni MaxTile')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("wykres_klocki.png", dpi=DPI)
    plt.close()

def plot_moves_only(df):
    """
    Rysuje średnią liczbę ruchów na grę.
    """
    print("Generowanie: Wykres Ruchów...")

    # Sprawdzenie po wyczyszczeniu nazw kolumn
    if 'Moves' not in df.columns:
        print(f"BŁĄD: Brak kolumny 'Moves'. Dostępne kolumny: {list(df.columns)}")
        return

    # Użycie min_periods=1 sprawia, że wykres rysuje się od samego początku,
    # zamiast czekać na wypełnienie okna 50 epizodów.
    rolling_moves = df['Moves'].rolling(window=WINDOW_SIZE, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f'Średnia Liczba Ruchów (Przetrwanie - średnia z {WINDOW_SIZE})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epizody', fontsize=12)
    ax.set_ylabel('Liczba Ruchów', fontsize=12)

    ax.plot(df['Episode'], rolling_moves, color='purple', linewidth=2, label='Avg Moves')

    # Dodanie surowych danych w tle (jasne), żeby zobaczyć wariancję
    ax.plot(df['Episode'], df['Moves'], color='purple', linewidth=1, alpha=0.15, label='Raw Moves')

    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("wykres_ruchy.png", dpi=DPI)
    plt.close()

def generate_charts():
    if not os.path.exists(CSV_FILE):
        print(f"Brak pliku {CSV_FILE}")
        return

    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            print("Plik CSV jest pusty.")
            return

        # Kluczowa poprawka: czyszczenie danych
        df = clean_data(df)

    except Exception as e:
        print(f"Błąd odczytu CSV: {e}")
        return

    setup_style()

    plot_just_scores(df)
    plot_weights(df)
    plot_tiles_split(df)
    plot_moves_only(df)

    print("--> Zakończono generowanie wszystkich 4 wykresów.")

    try:
        if os.name == 'nt': # Tylko dla Windows
            os.startfile("wykres_ruchy.png")
    except:
        pass

if __name__ == "__main__":
    generate_charts()