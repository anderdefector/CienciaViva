import cv2
import numpy as np
#import modelo
import joblib
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier

def trackChaned(x):
    pass



nombre = 'MiModelo.sav'
red_optimizada = joblib.load(nombre)

hh='Max'
wnd = 'Colorbars'

cv2.namedWindow("Camara")
cv2.namedWindow("Clasificados")
cv2.namedWindow("Binarizacion")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 0.5

thickness = 2

e=0
vc = cv2.VideoCapture(0)
cv2.createTrackbar("Max", "Camara",0,255,trackChaned)
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hul=cv2.getTrackbarPos("Max", "Camara")
    ret,thresh1 = cv2.threshold(gray,hul,255,cv2.THRESH_BINARY_INV)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    #ret,thresh4 = cv2.threshold(img,hul,huh,cv2.THRESH_TOZERO_INV)
 
    MCor=frame
    

    key = cv2.waitKey(20)
    n, Labels, stats, centroids = cv2.connectedComponentsWithStats( thresh1 )

    print( n )
    
    i = 1
    while i < n :
        x1 =  stats[i, cv2.CC_STAT_LEFT]
        y1 =  stats[i, cv2.CC_STAT_TOP]
        x2 = x1 + stats[i,cv2.CC_STAT_WIDTH]
        y2 = y1 + stats[i,cv2.CC_STAT_HEIGHT]
        Img2 = Labels[y1:y2, x1:x2] == i
        Img3 = 1 * Img2
        mu = cv2.moments( Img3, True )
        hu = cv2.HuMoments(mu)
        hur = np.transpose(hu)

        ypredicha = red_optimizada.predict(hu.T)
        if(ypredicha == 0):
            color = (255, 0, 0)
            MCor = cv2.putText(frame, 'L', (x1+5,y1+5), font, fontScale, color, thickness, cv2.LINE_AA) 
        if(ypredicha == 1):
            color = (0, 255, 0)
            MCor = cv2.putText(frame, 'O', (x1+5,y1+5), font, fontScale, color, thickness, cv2.LINE_AA)
        if(ypredicha== 2):
            color = (0, 0, 255)
            MCor = cv2.putText(frame, 'Cir', (x1+5,y1+5), font, fontScale, color, thickness, cv2.LINE_AA)
        if(ypredicha == 3):
            color = (255, 0, 255)
            MCor = cv2.putText(frame, 'Cruz', (x1+5,y1+5), font, fontScale, color, thickness, cv2.LINE_AA)
        start_point = (x1, y1)
        end_point = (x2, y2)
            
        thickness = 2
        MCor = cv2.rectangle(frame, start_point, end_point, color, thickness)
            
        i += 1

 
    cv2.imshow("Clasificados", MCor)  
    cv2.imshow("Binarizacion",thresh1)
    cv2.imshow("Camara", gray)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyAllWindows()

 
