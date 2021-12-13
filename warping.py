from imutils import face_utils
import cv2
import dlib
import numpy as np

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


opts = np.float32([[96, 9],
                [163,26],
                [88, 130],
                [14, 97]])

def warp(f):
    a  = cv2.imread(f, 0)
    a = cv2.resize(a, (600,600))
    rects = detector(a, 0)
    mx,Mx,my,My = 1000,-1,1000,-1
    if len(rects) == 0:
        return a, False

    pts = np.array([])
    for (i , rect) in enumerate(rects):
        shape = predictor(a, rect)
        shape = face_utils.shape_to_np(shape)
        for (x ,y) in shape[50:]:
            mx, Mx = min(mx, x) , max(Mx, x)
            my, My = min(my, y) , max(My, y)

        for k in [51, 53, 57, 59]:
            pts = np.append(pts, [shape[k][0]-mx, shape[k][1]-my])
        

    pts = pts[:8]
    pts = np.reshape(pts,(4, 2))
    pts = pts.astype(np.float32)

    #if np.array_equal(pts, opts):
    #    print(f)

    m = cv2.getPerspectiveTransform(pts, opts)
    r = cv2.warpPerspective(a[my:My, mx:Mx], m, (Mx-mx, My-my),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
            borderValue=(255))

    return r, True


# img = warp("./Happy/Training_20866.jpg")
# cv2.imshow("img", img)
# cv2.waitKey(0)
