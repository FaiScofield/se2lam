#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
from matplotlib import pyplot as plt

if __name__=='__main__':
    if len(sys.argv) > 2:
        fileVI = sys.argv[1]
        fileOdo = sys.argv[2]
    else :
        print('Usage: run_exe odomVI_file odo_raw_file')
        print('odomVI_file format: timestamp x y theta')
        print('odo_raw_file format: x y theta')
        sys.exit(0)

    odomVIData = open(fileVI, "r")
    x = []
    y = []
    for line in odomVIData:
        # timestamp x y theta
        value = [float(s) for s in line.split()]
        if len(value) == 4:
            x.append(value[1])
            y.append(value[2])
        else:
            continue


    odomData = open(fileOdo, "r")
    x1 = []
    y1 = []
    for line in odomData:
        # timestamp x y theta
        value = [float(s) for s in line.split()]
        if len(value) == 3:
            x1.append(value[0])
            y1.append(value[1])
        else:
            continue

    p1, = plt.plot(x, y, 'b-')
    p2, = plt.plot(x[0], y[0], 'ro')
    p3, = plt.plot(x1, y1, 'y.')
    p4, = plt.plot(x1[0], y1[0], 'bo')
    plt.legend(handles=[p1, p2, p3, p4], \
               labels=['Trajectory_VI', 'SP VI', 'Trajectory_Odo', 'SP Odo'])
    plt.show()
