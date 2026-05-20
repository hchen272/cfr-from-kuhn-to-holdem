from game import *
from cfr import *
import random
from utils import *


class Trainer:

    def train(self, iterations):
        # Clean the logs directory before training
        save_strategy_txt(
            node_map,
            0,
            0,
            iterations
        )
        util = 0

        for i in range(iterations):

            # shuffle cards
            cards = CARDS.copy()

            random.shuffle(cards)

            cards = (cards[0], cards[1])

            # run CFR
            util += cfr(cards, "", 1, 1)

            # print progress
            if (i + 1) % 100000 == 0:

                avg_game_value = round(util / (i + 1), 4)

                print(
                    f"Iteration {i+1}"
                )

                print(
                    f"Average game value: "
                    f"{avg_game_value}"
                )

                print("-" * 40)

                save_strategy_txt(node_map, i + 1, avg_game_value, iterations)

        print("\n=== FINAL STRATEGIES ===\n")

        for infoset in sorted(node_map):

            node = node_map[infoset]

            avg_strategy = node.get_average_strategy()

            print(
                f"{infoset}: "
                f"{avg_strategy}"
            )

        save_model(node_map, iterations)

if __name__ == "__main__":

    trainer = Trainer()

    trainer.train(10000000000)