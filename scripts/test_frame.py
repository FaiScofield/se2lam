#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def test_frame():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim((-1, 3))
    ax.set_ylim((-1, 3))
    ax.set_zlim((-1, 3))

    # F1
    O1 = np.matrix([[0], [0], [0]])
    X1 = np.matrix([[1], [0], [0]])
    Y1 = np.matrix([[0], [1], [0]])
    Z1 = np.matrix([[0], [0], [1]])
    ax.plot3D([0, 1], [0, 0], [0, 0], 'r')
    ax.plot3D([0, 0], [0, 1], [0, 0], 'g')
    ax.plot3D([0, 0], [0, 0], [0, 1], 'b')
    ax.scatter(0, 0, 0, c='k')

    # R and t
    R = np.matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    print("R = \n", R)
    t = np.matrix([[2], [1], [0]])
    print("t = ", t.transpose())

    # F2
    O2 = np.matmul(R, O1) + t
    X2 = np.matmul(R, X1) + t
    Y2 = np.matmul(R, Y1) + t
    Z2 = np.matmul(R, Z1) + t
    ax.plot3D([O2[0, 0], X2[0, 0]], [O2[1, 0], X2[1, 0]], [O2[2, 0], X2[2, 0]], 'r')
    ax.plot3D([O2[0, 0], Y2[0, 0]], [O2[1, 0], Y2[1, 0]], [O2[2, 0], Y2[2, 0]], 'g')
    ax.plot3D([O2[0, 0], Z2[0, 0]], [O2[1, 0], Z2[1, 0]], [O2[2, 0], Z2[2, 0]], 'b')
    ax.scatter(2, 1, 0, c='k')

    print("O2 = ", O2.transpose())
    print("X2 = ", X2.transpose())
    print("Y2 = ", Y2.transpose())
    print("Z2 = ", Z2.transpose())

    P_o1 = np.matrix([[3], [0], [1]])
    P_o2 = np.matrix([[1], [1], [1]])
    print("P_o1 = ", (np.matmul(R, P_o2) + t).transpose())

    plt.show()


#
# def test_Affine:
#     fx = 

