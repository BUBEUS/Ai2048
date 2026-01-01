from game_2048 import Game2048
from ai_player import AIPlayer
import numpy as np
import random

# Hiperparametry
ALPHA = 0.0005  # Szybkość uczenia
GAMMA = 0.99    # Współczynnik dyskontowania (przyszłe nagrody)
EPISODES = 10000  # Liczba gier do rozegrania

def train():
    ai = AIPlayer()
    ai.alpha = ALPHA

    scores_history = []

    for episode in range(EPISODES):
        game = Game2048()
        state = game.board.copy()
        done = False

        # Epsilon maleje liniowo: od 100% losowości do 10%
        epsilon = max(0.1, 1.0 - episode / 2000)

        while not done:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                break

            # === Wybór ruchu (Epsilon-Greedy + Expectimax) ===
            if random.random() < epsilon:
                best_move = random.choice(valid_moves)
            else:
                best_move, best_v = None, -float('inf')
                sim_game = Game2048(game.size)

                for move in valid_moves:
                    sim_game.board = state.copy()
                    # Ruch deterministyczny (bez kafelka)
                    next_s, _, _ = sim_game.move_without_random(move)

                    # ZMIANA: Oceniamy E[V] zamiast surowego V
                    # "Jak dobry średnio będzie ten stan po pojawieniu się kafelka?"
                    v = ai.get_expected_value(next_s)
                    if v > best_v:
                        best_move, best_v = move, v

            # === Wykonanie i Nauka ===
            next_state, reward, done, _ = game.move(best_move)

            # TD Update: Target też musi uwzględniać oczekiwaną przyszłość
            current_val = ai.evaluate(state)  # Tu nadal surowe V(s) - oceniamy obecny stan

            if done:
                target = reward
            else:
                # ZMIANA: Gamma * E[V(next_state)]
                # Ponieważ next_state z game.move() ma już nowy kafelek,
                # technicznie jest to s''. Ale dla spójności TD:
                # Target = r + gamma * Evaluate(stan_z_nowym_kafelkiem)
                target = reward + GAMMA * ai.evaluate(next_state)

            td_error = target - current_val
            ai.update_weights(ai.get_features(state), td_error)

            state = next_state.copy()

        scores_history.append(game.score)
        if len(scores_history) > 100:
            scores_history.pop(0)

        if episode % 100 == 0:
            avg_score = sum(scores_history) / len(scores_history)
            print(f"Ep: {episode} | Score: {game.score} | Max: {np.max(game.board)} | Avg Score: {avg_score:.2f} | Wagi: {ai.weights.round(2)}")
            print(f"Max Tile: {np.max(game.board)}")
            print(game.board)

if __name__ == "__main__":
    train()