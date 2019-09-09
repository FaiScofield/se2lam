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
    if len(sys.argv) == 5:
        fileVI = sys.argv[1]
        fileOdo = sys.argv[2]
        startIndex = int(sys.argv[3])
        endIndex = int(sys.argv[4])
    elif len(sys.argv) == 3:
        fileVI = sys.argv[1]
        fileOdo = sys.argv[2]
        startIndex = 0
        endIndex = 3000
    else:
        print('Usage: run_exe <trajectoryKF> <odo_raw> [startIndex=0] [endIndex=len(XX)]')
        print('trajectoryKF format: frameID x y z yaw')
        print('odo_raw_file format: x y yaw')
        sys.exit(0)
    print 'Set startIndex and endIndex to: ', startIndex, endIndex

    # Read VI
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
    print 'Read all odomVI size: ', len(odomVI)

    # Read odomRaw All
    odomRawAll = []
    odomData = open(fileOdo, "r")
    for line in odomData:
        # x y theta
        value = [float(s) for s in line.split()]
        if len(value) == 3:
            odomRawAll.append(np.array([[value[0] / 1000.], [value[1] / 1000.]]))
        else:
            print 'skip one line! this should not be happened!!'
            continue
    print 'Read all odom size: ', len(odomRawAll)

    # Get odomRaw between [startIndex, endIndex]
    index = 0
    odomRaw = []
    endIndex = min(len(odomRawAll), endIndex)
    print 'Final startIndex and endIndex is: ', startIndex, endIndex
    for index in range(len(odomRawAll)):
        if (index < startIndex):
            continue
        if index > endIndex:
            break
        odomRaw.append(odomRawAll[index])
    print 'Get odom frame size: ', len(odomRaw)

    # Get odomRaw with VI KF pose
    odomRawKF = []
    for i in range(len(odomVIidx)):
        odomRawKF.append(odomRaw[odomVIidx[i]])
    print len(odomVI), "/", len(odomRawKF), ' Final size of poses for ICP.'

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
    for i in range(len(odomRawAll)):
        x1.append(float(odomRawAll[i][0]))
        y1.append(float(odomRawAll[i][1]))

    plt.plot(x1, y1, color='red', linewidth=1.0, linestyle = '-', label='Trajectory_VI')
    plt.plot(x1[0], y1[0], 'ro', label='SP_VI')
    plt.plot(x, y, color='black', linewidth=2.0, linestyle = '-', label='Trajectory_Odom')
    plt.plot(x[0], y[0], 'ko', label='SP_Odom')
    plt.legend(loc='best')
    # plt.xlim(-1, 2)
    # plt.ylim(-1, 2)
    plt.axis('equal')
    plt.show()
