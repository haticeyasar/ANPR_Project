
# Script to detect license plates.

import numpy as np
import cv2
import argparse
import imutils
import pytesseract


pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

CASCADE_PATH = 'data/cascade.xml'
RESIZE_HEIGHT = 600

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# Initiate cascade classifer.
plate_cascade = cv2.CascadeClassifier(CASCADE_PATH)

img = cv2.imread(args["image"])

img   = imutils.resize(img, height = RESIZE_HEIGHT)
img = cv2.resize(img, (620,480) )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect plates in img
plates = plate_cascade.detectMultiScale(gray,5,20)
    
print("Found %d plates:" %len(plates))
for (x,y,w,h) in plates:
	print (x,y,w,h)

	# Draw rectangle around found plates.
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


# Show image.
cv2.imshow('img',img)
cv2.waitKey(1000)
cv2.destroyAllWindows()

#image preprocess
"""
(thresh, gray) = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
gray = cv2.bilateralFilter(gray, 11, 17, 17) 
edged = cv2.Canny(gray, 30, 200)""" 


gray= cv2.bilateralFilter(gray, 13, 15, 15)
(thresh,output2) = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
edged = cv2.GaussianBlur(output2, (5, 5), 3)
edged = cv2.Canny(gray, 30, 200)

cv2.imshow('img',gray)
cv2.waitKey(10000)
cv2.destroyAllWindows()

contours= cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

for c in contours:
    
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
 
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print ("plaka bulunamadi!")
else:
     detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255),3)

    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1)    
    new_image = cv2.bitwise_and(img,img,mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]

    text = pytesseract.image_to_string(Cropped, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPRSTUVYZ --psm 8')
    print("Detected license plate Number is:",text)


    img = cv2.resize(img,(500,300))
    Cropped = cv2.resize(Cropped,(400,200))
    cv2.imshow('car',img)
    cv2.imshow('Cropped',Cropped)

cv2.waitKey()
cv2.destroyAllWindows()

