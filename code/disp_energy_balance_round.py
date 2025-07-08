import numpy as np

def round_half_away_from_zero(x, decimals=0):
    factor = 10**decimals
    # sign(x) * floor(abs(x) * factor + 0.5) / factor
    return np.sign(x) * np.floor(np.abs(x) * factor + 0.5) / factor

