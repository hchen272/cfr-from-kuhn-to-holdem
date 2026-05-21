import numpy as np


# Default number of actions (Kuhn Poker); overridden when passing to constructor
NUM_ACTIONS = 2


class Node:

    def __init__(self, num_actions=None):

        if num_actions is None:
            num_actions = NUM_ACTIONS
        self.num_actions = num_actions

        # cumulative regrets
        self.regret_sum = np.zeros(num_actions)

        # previous iteration's instant regret (used by PDCFR+ prediction)
        self.last_inst_regret = np.zeros(num_actions)

        # current strategy
        self.strategy = np.zeros(num_actions)

        # cumulative strategy
        self.strategy_sum = np.zeros(num_actions)

    def get_strategy(self, realization_weight, strategy_discount=1.0,
                     linear_weight=1.0):
        """
        Compute current strategy using regret matching.

        Args:
            realization_weight (float):
                reach probability

            strategy_discount (float):
                discount factor applied to strategy_sum before accumulation
                (used by DCFR; defaults to 1.0 for standard CFR / CFR+)

            linear_weight (float):
                iteration weight for linear averaging (CFR+).
                ``strategy_sum += linear_weight * reach_prob * strategy``.
                Defaults to 1.0 (uniform averaging) for standard CFR.

        Returns:
            np.array:
                current mixed strategy
        """

        normalizing_sum = 0

        # regret matching
        for a in range(self.num_actions):

            self.strategy[a] = max(self.regret_sum[a], 0)

            normalizing_sum += self.strategy[a]

        # normalize
        for a in range(self.num_actions):

            if normalizing_sum > 0:
                self.strategy[a] /= normalizing_sum
            else:
                # uniform random strategy
                self.strategy[a] = 1.0 / self.num_actions

            # accumulate average strategy (with optional DCFR discount
            # and CFR+ linear weight)
            self.strategy_sum[a] = (
                strategy_discount * self.strategy_sum[a]
                + (linear_weight * realization_weight) * self.strategy[a]
            )

        return self.strategy

    def get_average_strategy(self):
        """
        Compute average strategy over all iterations.
        """

        avg_strategy = np.zeros(self.num_actions)

        normalizing_sum = np.sum(self.strategy_sum)

        for a in range(self.num_actions):

            if normalizing_sum > 0:
                avg_strategy[a] = (
                    self.strategy_sum[a] / normalizing_sum
                )
            else:
                avg_strategy[a] = 1.0 / self.num_actions

        return avg_strategy

    def __str__(self):

        return (
            f"Regrets: {self.regret_sum} | "
            f"Avg Strategy: {self.get_average_strategy()}"
        )


if __name__ == "__main__":

    node = Node()

    print("Initial strategy:")
    print(node.get_strategy(1.0))

    print()

    # fake regrets
    node.regret_sum = np.array([3.0, -1.0])

    print("After regret update:")
    print(node.get_strategy(1.0))

    print()

    print(node)
