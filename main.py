from imutils import face_utils
import dlib
import cv2
 
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


cap = cv2.VideoCapture(0)
 
while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    rects = detector(gray, 0)
    
    mx,Mx,my,My = 1000,-1,1000,-1
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # len(shape) == 68
        # the points for mouth  50..

        for (x ,y) in shape[50:]:
            mx, Mx = min(mx, x) , max(Mx, x)
            my, My = min(my, y) , max(My, y)

        for (x, y) in shape[50:]:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    cv2.imshow("Output", image[my-2:My+2, mx-2:Mx+2])
    k = cv2.waitKey(5) & 0xFF
    if k == 27: # esc
        break

cv2.destroyAllWindows()
cap.release()
