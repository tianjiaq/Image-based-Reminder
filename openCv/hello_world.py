#Load up a color image in grey scale and save it as black&white

import numpy as np
import cv2

#   cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
#   cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
#   cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
#   Instead of these three flags, you can simply pass integers 1, 0 or -1 respectively.
img = cv2.imread('images/HelloWorld.png',0)
cv2.imshow('image',img)

k = cv2.waitKey(0) & 0xFF
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('images/HelloWorld.jpg',img)
    cv2.destroyAllWindows()
