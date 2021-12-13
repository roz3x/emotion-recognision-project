import  warping
import glob
import cv2


for f in glob.glob("Neutral/*"):
    img, t = warping.warp(f)
    print(t)
    if t:
        cv2.imwrite("output/"+f, img)
