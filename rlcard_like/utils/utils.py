import numpy as np

def rank2int(rank):
    ''' Map card rank to integer. '''
    return {'J': 0, 'Q': 1, 'K': 2}.get(rank, -1)

def remove_illegal(action_probs, legal_actions):
    ''' Zero out illegal actions and re-normalise. '''
    probs = np.zeros_like(action_probs)
    for a in legal_actions:
        probs[a] = action_probs[a]
    s = probs.sum()
    if s > 0:
        probs /= s
    else:
        probs[legal_actions] = 1.0 / len(legal_actions)
    return probs

def set_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)
