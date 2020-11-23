#!/usr/bin/env python

import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA
from scipy.spatial import distance as dist


def is_contour_bad(c):

    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.1 * peri, True)

    # the contour is 'bad' if it is not a rectangle
    return not len(approx) == 4 and cv2.arcLength(approx, True) > 200


def order_points(pts):

    # from PyImageSearch
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


ll = 0

for k in np.sort(glob.glob("frames/*.png")):

    img = cv2.imread(k)
    size = img.shape

    print(size)

    mask = np.zeros(img.shape[0:2], dtype=np.uint8)

    # define ROI
    X1 = 475
    X2 = 900
    Y1 = 125
    Y2 = 425

    DX = X2-X1
    DY = Y2-Y1

    points = np.array([[[X1, Y1], [X2, Y1], [X2, Y2], [X1, Y2]]])

    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

    res = cv2.bitwise_and(img, img, mask=mask)
    rect = cv2.boundingRect(points)  # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    wbg = np.ones_like(img, np.uint8)*255
    cv2.bitwise_not(wbg, wbg, mask=mask)

    dst = wbg+res

    # Canny
    edges = cv2.Canny(res, 170, 200)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)

    wcanvas = np.ones(cropped.shape[:2], dtype="uint8") * 255

    _, contours, _ = cv2.findContours(
        edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    angles = []
    sizes = []

    for k in contours:
        if is_contour_bad(k):

            rotrect = cv2.minAreaRect(k)
            pointsrect = cv2.boxPoints(rotrect)

            ordered = order_points(pointsrect)

            box = np.int0(pointsrect)

            edge1 = np.array([np.array([ordered[1][0], ordered[1][1]]), np.array(
                [ordered[2][0], ordered[2][1]])])
            edge2 = np.array([np.array([ordered[0][0], ordered[0][1]]), np.array(
                [ordered[1][0], ordered[1][1]])])

            norm1 = np.sqrt((edge1[0][0]-edge1[1][0]) **
                            2 + (edge1[0][1]-edge1[1][1])**2)
            norm2 = np.sqrt((edge2[0][0]-edge2[1][0]) **
                            2 + (edge2[0][1]-edge2[1][1])**2)

            longedge = np.copy(edge1)

            # retrieve largest edge
            if(norm2 > norm1):
                longedge = edge2

            # AR filtering
            if np.amin(np.array([norm1, norm2]))/np.max(np.array([norm1, norm2])) < 0.1:

                cv2.drawContours(wcanvas, [box], 0, (0, 0, 255), 2)

                # origin to bottom left
                x1 = longedge[0][0]
                y1 = DY-longedge[0][1]

                x2 = longedge[1][0]
                y2 = DY-longedge[1][1]

                angle = 180.0/np.pi*(np.arctan2(y2-y1, x2-x1))

                angles = np.append(angles, angle)
                sizes = np.append(sizes, rotrect[1][0]*rotrect[1][1])

    if(len(angles) != 0):
        a = angles[np.argmax(sizes)]

        # remove the contours from the image and show the resulting images
        image = cropped

        textimg = "Tilt angle: "+str(np.round(a, 1))+" degrees"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, textimg, (50, 50), font,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        FILENAME = "frame_"+str(ll).zfill(3)+".png"

        ll += 1

        cv2.imwrite(FILENAME, img)
