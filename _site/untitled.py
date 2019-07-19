
import numpy as np
def check_pal(s):
    chars = [0]*128
    for c in s:
        chars[ord(c)] += 1

    if len(s) % 2 == 0:
        if np.sum(chars) % 2 == 0:
            return True
        else:
            return False
    else:
        num_odd = 0
        for el in chars:
            if el % 2 == 0:
                pass
            else:
                num_odd += 1
        if num_odd > 1:
            return False
        return True