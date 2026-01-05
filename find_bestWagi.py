import pandas as pd
import os

CSV_FILE = "training_history.csv"
WINDOW_SIZE = 50  

def analyze_best_performance():
    """
    Analizuje plik CSV w poszukiwaniu najlepszego momentu treningu.
    
    Znajduje epizod z najwyższą średnią kroczącą wyników (window=50),
    a następnie wypisuje gotowy kod Pythona z wagami do wklejenia w `ai_player.py`.
    """
    if not os.path.exists(CSV_FILE):
        print(f"Błąd: Nie znaleziono pliku {CSV_FILE}")
        return

    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Błąd odczytu pliku: {e}")
        return

    if df.empty:
        print("Plik jest pusty.")
        return

    print(f"--> Analiza {len(df)} epizodów... Szukam najlepszej konfiguracji...")


    df['Rolling_Score'] = df['Score'].rolling(window=WINDOW_SIZE).mean()

    best_idx = df['Rolling_Score'].idxmax()

    if pd.isna(best_idx):
        best_idx = df['Score'].idxmax()
        best_val = df.loc[best_idx, 'Score']
        desc = "Najlepszy pojedynczy wynik"
    else:
        best_val = df.loc[best_idx, 'Rolling_Score']
        desc = f"Najlepsza średnia z {WINDOW_SIZE} gier"

    best_row = df.loc[best_idx]
    episode = int(best_row['Episode'])

    cols_norm = ['N_Empty', 'N_Max', 'N_Snake', 'N_Merge', 'N_Corner', 'N_Neigh']
    cols_panic = ['P_Empty', 'P_Max', 'P_Snake', 'P_Merge', 'P_Corner', 'P_Neigh']

    vals_norm = [best_row[c] for c in cols_norm if c in df.columns]
    vals_panic = [best_row[c] for c in cols_panic if c in df.columns]

    if len(vals_norm) < 6: vals_norm = [0.0] * 6
    if len(vals_panic) < 6: vals_panic = [0.0] * 6

    def format_list(lst):
        return ", ".join([f"{x:.4f}" for x in lst])

    str_norm = format_list(vals_norm)
    str_panic = format_list(vals_panic)

    print("\n" + "="*60)
    print(f"🏆 ZNALEZIONO OPTIMUM (Epizod: {episode})")
    print(f"📊 {desc}: {best_val:.2f}")
    print("="*60)

    print("\n--- KOD DO WKLEJENIA (ai_player.py) ---\n")

    print(f"self.weights_normal = np.array([{str_norm}])")
    print(f"self.weights_panic  = np.array([{str_panic}])")

    print("\n" + "-"*60)

if __name__ == "__main__":
    analyze_best_performance()