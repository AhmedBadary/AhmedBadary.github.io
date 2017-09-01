import numpy as np

data_input = ['24:59','02:00']

def solution(data_input):
    differences = []
    # 1
    new_arr = [[int(inp.split(":")[0]),int(inp.split(":")[1])] for inp in data_input]
    # 2
    new_arr = sorted(new_arr, key = lambda x: x[0])
    # 3
    new_arr = sorted(new_arr, key = lambda x: x[1])
    # 4
    for i in range(len(new_arr)):
        differences.append(abs(new_arr[i-1] - new_arr[i]))

    return np.argmax(np.array(differences))
    


