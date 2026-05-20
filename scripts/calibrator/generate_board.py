import cv2
import cv2.aruco as aruco
import numpy as np

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard((5, 7), 0.04, 0.02, aruco_dict)

# 2480x3508 = A4 300 DPI
img = board.generateImage((2480, 3508), marginSize=150, borderBits=1)
cv2.imwrite("charuco_board.png", img)


# kağıt: A4
# Yönlendirme: Portrait
# Ölçek: %100
# "Sayfaya Sığdır":KAPALI  
# "Fit to page":KAPALI
#içteki siyah marker 2cm, dıştaki kare 4cm