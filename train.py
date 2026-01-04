from game_2048 import Game2048
from ai_player import AIPlayer
import numpy as np
import math
import random
import time
import os
import csv

ALPHA_START = 0.001
ALPHA_END = 0.0001
GAMMA = 0.99
EPISODES = 5000
LOG_FILE = "training_history.csv"

def get_shaped_reward(game_reward, board):
    """
    Oblicza nagrodę ukształtowaną (Reward Shaping) dla uczenia ze wzmocnieniem.
    
    Logarytmuje surową nagrodę punktową i dodaje bonus za aktywność,
    aby zachęcić AI do wykonywania ruchów łączących klocki.

    Args:
        game_reward (int): Punkty zdobyte w ruchu (z silnika gry).
        board (np.ndarray): Stan planszy po ruchu.

    Returns:
        float: Zmodyfikowana wartość nagrody.
    """
    if game_reward == 0:
        reward = 0
    else:
        reward = math.log2(game_reward)


    if game_reward > 0:
        reward += 1.0

    return reward


def save_logs_to_csv(buffer, filename):
    """
    Zapisuje bufor logów treningowych do pliku CSV.

    Tworzy nagłówek, jeśli plik nie istnieje.
    Zapisuje: Epizod, Wynik, MaxTile, Czas, oraz wszystkie wagi (Normal/Panic).

    Args:
        buffer (list): Lista list z danymi wierszy.
        filename (str): Nazwa pliku wyjściowego.
    """
    file_exists = os.path.exists(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = [
                "Episode", "Score", "MaxTile", "Moves", "Duration_Sec",
                "N_Empty", "N_Max", "N_Snake", "N_Merge", "N_Corner", "N_Neigh",
                "P_Empty", "P_Max", "P_Snake", "P_Merge", "P_Corner", "P_Neigh"
            ]
            writer.writerow(header)
        writer.writerows(buffer)
        print(f"--> Zapisano {len(buffer)} wpisów.")

def train():
    """
    Główna pętla treningowa AI.
    
    Proces:
    1. Inicjalizacja gry i AI (ładowanie checkpointu).
    2. Pętla po epizodach (grach).
    3. Wybór ruchu (1-step Lookahead).
    4. Aktualizacja wag (TD-Learning).
    5. Logowanie wyników i zapisywanie modelu.
    """
    ai = AIPlayer()
    scores_history = []
    max_tiles_history = []
    csv_buffer = []


    CHECKPOINT_FILE = "ai_2048_save.pkl"
    start_episode = ai.load_model(CHECKPOINT_FILE)

    if start_episode > 0:
        print(f"Wznowiono trening od epizodu: {start_episode}")
    else:
        print("Rozpoczynam nowy trening...")

    start_time = time.time()

    current_episode = start_episode
    target_episode = start_episode + EPISODES

    while current_episode < target_episode:
        game = Game2048()
        state = game.board.copy()
        done = False

        game_start_time = time.time()
        moves_count = 0

   
        if current_episode < 2000:
            progress = current_episode / 2000
            base_alpha = ALPHA_START - ((ALPHA_START - ALPHA_END) * progress)
        else:
            base_alpha = ALPHA_END

      
        current_avg_score = 0
        if len(scores_history) > 0:
            current_avg_score = sum(scores_history) / len(scores_history)

        ai.alpha = 0.00005
        epsilon = 0 


     

        sim_game = Game2048(game.size) 

        while not done:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break

        
            if random.random() < epsilon:
                best_move = random.choice(valid_moves)
            else:
                best_move, best_v = None, -float('inf')

                for move in valid_moves:
                    sim_game.board = state
                    next_s_sim, _, _ = sim_game.move_without_random(move)

                    v = ai.get_expected_value(next_s_sim)

                    if v > best_v:
                        best_v = v
                        best_move = move

        
            next_state_real, raw_reward, done, _ = game.move(best_move)
            moves_count += 1

            reward_shaped = get_shaped_reward(raw_reward, next_state_real)
            features_state = ai.get_features(state)
            current_v = ai.evaluate(state)
          

            if done:
                target = reward_shaped - 30.0
            else:
                next_v = ai.evaluate(next_state_real)
                target = reward_shaped + GAMMA * next_v

            td_error = target - current_v

          
            ai.update_weights(features_state, td_error)

            state = next_state_real.copy()


        current_episode += 1  

        game_duration = time.time() - game_start_time
        log_entry = [
            current_episode,
            game.score,
            np.max(game.board),
            moves_count,
            round(game_duration, 4),
            round(ai.weights_normal[0], 5), round(ai.weights_normal[1], 5),
            round(ai.weights_normal[2], 5), round(ai.weights_normal[3], 5),
            round(ai.weights_normal[4], 5), round(ai.weights_normal[5], 5),
            round(ai.weights_panic[0], 5), round(ai.weights_panic[1], 5),
            round(ai.weights_panic[2], 5), round(ai.weights_panic[3], 5),
            round(ai.weights_panic[4], 5), round(ai.weights_panic[5], 5)
        ]
        csv_buffer.append(log_entry)

        scores_history.append(game.score)
        max_tiles_history.append(np.max(game.board))

        if len(scores_history) > 100:
            scores_history.pop(0)
            max_tiles_history.pop(0)

        
        if current_episode % 200 == 0:
            ai.save_model(CHECKPOINT_FILE, current_episode)
            save_logs_to_csv(csv_buffer, LOG_FILE) 
            csv_buffer = [] 

        if current_episode % 50 == 0:
            avg_score = sum(scores_history) / len(scores_history)
            avg_max = sum(max_tiles_history) / len(max_tiles_history)

            end_time = time.time()
            duration = end_time - start_time
            start_time = time.time()

            print(f"Ep: {current_episode} | Avg Score: {avg_score:.0f} | Avg MaxTile: {avg_max:.0f} | Time (50 ep): {duration:.2f}s")
            print(f"Wagi NORMAL: E={ai.weights_normal[0]:.2f}, M={ai.weights_normal[1]:.2f}, S={ai.weights_normal[2]:.2f}, Mrg={ai.weights_normal[3]:.2f}, Crn={ai.weights_normal[4]:.2f}, Ngh={ai.weights_normal[5]:.2f}")
            print(f"Wagi PANIC : E={ai.weights_panic[0]:.2f}, M={ai.weights_panic[1]:.2f}, S={ai.weights_panic[2]:.2f}, Mrg={ai.weights_panic[3]:.2f}, Crn={ai.weights_panic[4]:.2f}, Ngh={ai.weights_panic[5]:.2f}")

            print("Ostatnia plansza:")
            print(game.board)
            print("-" * 40)

    if csv_buffer:
        save_logs_to_csv(csv_buffer, LOG_FILE)

    ai.save_model(CHECKPOINT_FILE, current_episode)

if __name__ == "__main__":
    train()