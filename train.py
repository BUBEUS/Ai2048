from game_2048 import Game2048
from ai_player import AIPlayer
import numpy as np
import math
import random
import time

# Parametry
ALPHA_START = 0.001
ALPHA_END = 0.0001
GAMMA = 0.99
EPISODES = 5000

def get_shaped_reward(game_reward, board):
    if game_reward == 0:
        return 0
    return math.log2(game_reward)

def train():
    ai = AIPlayer()
    scores_history = []
    max_tiles_history = []

    start_time = time.time()

    print("Start treningu (Normalized Features)...")

    for episode in range(EPISODES):
        game = Game2048()
        state = game.board.copy()
        done = False

        # Annealing Alpha i Epsilon
        progress = episode / EPISODES
        ai.alpha = ALPHA_START - (ALPHA_START - ALPHA_END) * progress
        epsilon = max(0.01, 0.2 - progress * 0.2) # Zmniejszona losowość na start

        sim_game = Game2048(game.size) # 1.1 tu wyciagniecie

        while not done:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break

            # Epsilon-Greedy
            if random.random() < epsilon:
                best_move = random.choice(valid_moves)
            else:
                best_move, best_v = None, -float('inf')
                #sim_game = Game2048(game.size) 1.1 wyciagniecie przed petle dla optymalizacji

                for move in valid_moves:
                    sim_game.board = state.copy()
                    next_s_sim, _, _ = sim_game.move_without_random(move)

                    # 1-step Lookahead z Expectimaxem
                    v = ai.get_expected_value(next_s_sim)

                    if v > best_v:
                        best_v = v
                        best_move = move

            # Wykonanie
            next_state_real, raw_reward, done, _ = game.move(best_move)

            # Nauka
            reward_shaped = get_shaped_reward(raw_reward, next_state_real)
            features_state = ai.get_features(state)
            current_v = np.dot(ai.weights, features_state)

            if done:
                target = reward_shaped
            else:
                next_v = ai.evaluate(next_state_real)
                target = reward_shaped + GAMMA * next_v

            td_error = target - current_v
            ai.update_weights(features_state, td_error)

            state = next_state_real.copy()

        scores_history.append(game.score)
        max_tiles_history.append(np.max(game.board))

        if len(scores_history) > 100:
            scores_history.pop(0)
            max_tiles_history.pop(0)

        # Output co 50 epok
        if episode % 50 == 0:
            avg_score = sum(scores_history) / len(scores_history)
            avg_max = sum(max_tiles_history) / len(max_tiles_history)

            end_time = time.time()
            duration = end_time - start_time
            start_time = time.time()

            print(f"Ep: {episode} | Avg Score: {avg_score:.0f} | Avg MaxTile: {avg_max:.0f} | Time (50 ep): {duration:.2f}s")
            print(f"Wagi: Empty={ai.weights[0]:.2f}, Max={ai.weights[1]:.2f}, Snake={ai.weights[2]:.2f}, Merge={ai.weights[3]:.2f}")
            print("Ostatnia plansza:")
            print(game.board)
            print("-" * 40)

if __name__ == "__main__":
    train()