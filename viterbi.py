import numpy as np
import sys

def read_input(file_path):
    with open(file_path, 'r') as f:
        size = list(map(int, f.readline().strip().split()))
        map_data = [list(f.readline().strip().replace(' ', '')) for _ in range(size[0])]
        N = int(f.readline().strip())
        O = [f.readline().strip() for _ in range(N)]
        error_rate = float(f.readline().strip())
    # print('size: ', size)
    # print('map_data: ', map_data)
    # print('(1, 1): ', map_data[1][1])
    # print('N: ', N)
    # print('O: ', O)
    # print('error_rate: ', error_rate)
    return size, map_data, N, O, error_rate

def viterbi(map_data, size, N, O, error_rate):
    K = sum(x.count('0') for x in map_data)
    # print('K: ', K)
    if K != 0:
        prob = 1 / K
    # print('prob: ', prob)
    Q = [prob if map_data[i][j] == '0' else 0 for i in range(size[0]) for j in range(size[1])]
    Q = np.array(Q)
    Q = Q[Q != 0].reshape(K, 1)
    # print('Q: ', Q)
    Q = np.array(Q)
    # print('Q: \n', Q)

    traversable_cells = [(i, j) for i in range(size[0]) for j in range(size[1]) if map_data[i][j] == '0']
    # print('traversable_cells: ', traversable_cells)
    Tm = np.zeros((K, K))
    for idx1, cell1 in enumerate(traversable_cells):
        possible_transitions = 0
        for idx2, cell2 in enumerate(traversable_cells):
            if (abs(cell1[0]-cell2[0]) + abs(cell1[1]-cell2[1])) == 1:
                Tm[idx1][idx2] = 1
                possible_transitions += 1
        if possible_transitions > 0:
            Tm[idx1] /= possible_transitions
    # print('Tm: \n', Tm)

    sensor_readings = ['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111', '1000', '1001', '1010','1011', '1100', '1101', '1110', '1111']
    
    Em = np.zeros((K, len(sensor_readings)))
    for i, cell in enumerate(traversable_cells):
        x, y = cell
        actual_reading = ''
        # check north
        # print('x: ', x)
        # print('y: ', y)
        if x == 0 or map_data[x-1][y] == 'X':
            # print('map_data[x-1][y]: ', map_data[x-1][y])
            actual_reading += '1'
        else:
            actual_reading += '0'
        # print('actual_reading north: ', actual_reading)
        # check south
        if x == size[0]-1 or map_data[x+1][y] == 'X':
            # print('map_data[x+1][y]: ', map_data[x+1][y])
            actual_reading += '1'
        else:
            actual_reading += '0'
        # print('actual_reading south: ', actual_reading)
        # check west
        if y == 0 or map_data[x][y-1] == 'X':
            # print('map_data[x][y-1]: ', map_data[x][y-1])
            actual_reading += '1'
        else:
            actual_reading += '0'
        # print('actual_reading west: ', actual_reading)
        # check east
        if y == size[1]-1 or map_data[x][y+1] == 'X':
            # print('map_data[x][y+1]: ', map_data[x][y+1])
            actual_reading += '1'
        else:
            actual_reading += '0'
        # print('actual_reading east: ', actual_reading)

        # print('actual_reading: ', actual_reading)
        for j, reading in enumerate(sensor_readings):
            # print('________________________________')
            # print('cell: ', cell)
            # print('x: ', x)
            # print('y: ', y)
            # print('reading: ', reading)
            # print('actual_reading: ', actual_reading)
            # print('dit: ', sum(a != b for a, b in zip(actual_reading, reading)))
            dit = sum(a != b for a, b in zip(actual_reading, reading))  # count the number of differences
            Em[i][j] = (1 - error_rate)**(4 - dit) * error_rate**dit
            # print('Em[i][j]: ', Em[i][j])
    Em = np.array(Em)
    # Em = np.transpose(Em) 
    # print('Em: \n', Em)

    trellis = np.zeros((K, N))
    # trellis = np.zeros((size[0], size[1]))
    # print('trellis: \n', trellis)
    # print('Q[1]: ', Q[0])
    # print('Q.flatten(): ', Q.flatten())
    # print('Em[0]: ', '\n', Em[0])

    for i in range(K):
        # print('i: ', Q[i])
        # print('Em[i][int("".join(str(x) for x in O[0]), 2): ',Em[i][int("".join(str(x) for x in O[0]), 2)])
        trellis[i][0] = Q[i] * Em[i][int("".join(str(x) for x in O[0]), 2)]
    # print('trellis: \n', trellis)

    for j in range(1, N):
        for i in range(K):

            trellis[i][j] = max([trellis[k][j-1] * Tm[k][i] * Em[i][int("".join(str(x) for x in O[j]), 2)] for k in range(K)])
    # print('trellis: ','\n', trellis)


    maps = []
    for t in range(N):
        m = np.zeros((size[0], size[1]))
        for state, pos in enumerate(traversable_cells):
            m[pos] = trellis[state, t]
        maps.append(m)
    print(maps)

    np.savez("output.npz", *maps)

def main():
    input_file = sys.argv[1]
    size, map_data, N, O, error_rate = read_input(input_file)
    viterbi(map_data, size, N, O, error_rate)

if __name__ == '__main__':
    main()
