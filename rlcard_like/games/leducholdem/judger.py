from rlcard_like.utils.utils import rank2int

class LeducholdemJudger:
    def __init__(self, np_random):
        self.np_random = np_random

    @staticmethod
    def judge_game(players, public_card):
        winners = [0] * len(players)
        fold_count = sum(1 for p in players if p.status == 'folded')
        alive_idx = None
        for idx, p in enumerate(players):
            if p.status == 'alive':
                alive_idx = idx
        if fold_count == len(players) - 1:
            winners[alive_idx] = 1

        if sum(winners) < 1:
            for idx, p in enumerate(players):
                if p.status == 'alive' and p.hand.rank == public_card.rank:
                    winners[idx] = 1
                    break

        if sum(winners) < 1:
            ranks = [rank2int(p.hand.rank) if p.status == 'alive' else -1 for p in players]
            max_rank = max(ranks)
            for idx, r in enumerate(ranks):
                if r == max_rank:
                    winners[idx] = 1

        total = sum(p.in_chips for p in players)
        each_win = total / sum(winners)
        payoffs = []
        for i, p in enumerate(players):
            payoffs.append((each_win - p.in_chips) if winners[i] else float(-p.in_chips))
        return payoffs
