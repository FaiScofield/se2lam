#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
from matplotlib import pyplot as plt
import numpy as np

# R21 and t21
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
        print('Usage: run_exe <trajectoryKF> <odo_raw>')
        print('trajectoryKF format: frameID x y z theta')
        print('odo_raw_file format: x y theta')
        sys.exit(0)

    odomVIData = open(fileVI, "r")
    odomVI = []
    odomVIidx = []
    for line in odomVIData:
        # frameID x y theta
        value = [float(s) for s in line.split()]
        if len(value) == 5:
            odomVIidx.append(int(value[0]))
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

    odomRawKF = []
    for i in range(len(odomVIidx)):
        odomRawKF.append(odomRaw[odomVIidx[i]])

    print len(odomVI), "/", len(odomRawKF), ' size of poses.'

    R, t = solvICP(odomVI, odomRawKF)
    print('R21:', R)
    print('t21:', t)
    print('yaw:', np.arccos(R[0][0]))
    R12 = np.linalg.inv(R)
    print('R12:', R12)
    print('yaw12:', np.arccos(R12[0][0]))

    x = []
    y = []
    x1 = []
    y1 = []
    for i in range(len(odomVI)):
        odomVI[i] = np.dot(R, odomVI[i]) + t
        x.append(float(odomVI[i][0]))
        y.append(float(odomVI[i][1]))
        x1.append(float(odomRawKF[i][0]))
        y1.append(float(odomRawKF[i][1]))

    p1, = plt.plot(x, y, 'k-')
    p2, = plt.plot(x[0], y[0], 'ro')
    p3, = plt.plot(x1, y1, 'r-')
    p4, = plt.plot(x1[0], y1[0], 'bo')
    plt.legend(handles=[p1, p2, p3, p4],
               labels=['Trajectory_VI', 'SP VI', 'Odom', 'SP Odo'])
    plt.show()
