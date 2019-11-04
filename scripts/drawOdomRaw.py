#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
from matplotlib import pyplot as plt

if __name__=='__main__':
    if len(sys.argv) > 1:
        file = sys.argv[1]
    else :
        print('Usage: run_exe odomVI_file')
        print('odomVI_file format: timestamp x y theta')
        sys.exit(0)

    odomData = open(file, "r")
    x = []
    y = []

    for line in odomData:
        # timestamp x y theta
        value = [float(s) for s in line.split()]
        if len(value) == 4:
            x.append(value[1])
            y.append(value[2])
        else:
            continue

    p1, = plt.plot(x, y)
    p2, = plt.plot(x[0], y[0], 'ro')
    plt.legend(handles=[p1, p2], labels=['Trajectory', 'Start point'])
    plt.show()
