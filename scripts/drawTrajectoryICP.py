#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
from matplotlib import pyplot as plt
import numpy as np


def solvICP(odomVI, odomRaw):
    center_e = np.array([[0.], [0.]])
    center_g = np.array([[0.], [0.]])
    for i in range(len(odomVI)):
        center_e = center_e + odomVI[i]
        center_g = center_g + odomRaw[i]

    # print 'tatal sum:', center_e, center_g
    # line vector
    center_e = center_e / len(odomVI)
    center_g = center_g / len(odomVI)
    # print 'ave:', center_e, center_g

    W = np.mat(np.zeros((2, 2)))
    for i in range(len(odomVI)):
        de = odomVI[i] - center_e
        dg = odomRaw[i] - center_g
        W = W + np.dot(dg, np.transpose(de))

    # print 'W = ', W
    U, S, VT = np.linalg.svd(W)
    # print 'U, S, V = ', U, S, VT

    R = np.dot(U, VT)
    t = center_g - np.dot(R, center_e)

    return R, t


if __name__ == '__main__':
    if len(sys.argv) > 2:
        fileVI = sys.argv[1]
        fileOdo = sys.argv[2]
    else:
        print('Usage: run_exe <trajectory> <odo_raw>')
        print('trajectory format: frameID x y theta')
        print('odo_raw_file format: x y theta')
        sys.exit(0)

    odomVIData = open(fileVI, "r")
    odomVI = []
    for line in odomVIData:
        # frameID x y theta
        value = [float(s) for s in line.split()]
        if len(value) == 4:
            odomVI.append(np.array([[value[1] / 1000.], [value[2] / 1000.]]))
        else:
            print 'skip one line'
            continue

    odomData = open(fileOdo, "r")
    odomRaw = []
    for line in odomData:
        # x y theta
        value = [float(s) for s in line.split()]
        if len(value) == 3:
            odomRaw.append(np.array([[value[0] / 1000.], [value[1] / 1000.]]))
        else:
            print 'skip one line'
            continue

    if len(odomVI) != len(odomRaw):
        print('Wrong size of trajectory and gt! need to drop same poses of them!')
        print('length of estimate: ', len(odomVI))
        print('length of odom_raw: ', len(odomRaw))
        if len(odomVI) < len(odomRaw):
            dm = len(odomRaw) - len(odomVI)
            dm_2 = round(dm / 2.)
            del odomRaw[0:dm_2]
            del odomRaw[dm_2 - dm:]
        else:
            dm = len(odomVI) - len(odomRaw)
            dm_2 = round(dm / 2.)
            del odomVI[0:dm_2]
            del odomVI[dm_2 - dm:]

    print len(odomVI), "/", len(odomRaw), ' size of poses.'

    R, t = solvICP(odomVI, odomRaw)
    print('R:', R)
    print('t:', t)

    x = []
    y = []
    x1 = []
    y1 = []
    for i in range(len(odomVI)):
        odomVI[i] = np.dot(R, odomVI[i]) + t
        x.append(float(odomVI[i][0]))
        y.append(float(odomVI[i][1]))
        x1.append(float(odomRaw[i][0]))
        y1.append(float(odomRaw[i][1]))

    p1, = plt.plot(x, y, 'b.')
    p2, = plt.plot(x[0], y[0], 'ro')
    p3, = plt.plot(x1, y1, 'y.')
    p4, = plt.plot(x1[0], y1[0], 'bo')
    plt.legend(handles=[p1, p2, p3, p4],
               labels=['Trajectory_VI', 'SP VI', 'Odom', 'SP Odo'])
    plt.show()
