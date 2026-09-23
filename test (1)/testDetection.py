import os
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

## OLD
# img = cv2.imread('img\\1.png')

## NEW
dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = os.path.join(dir_path, 'img', 'question.png')
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Imagem não encontrada em: {img_path}")

# tess -> only rgb && open -> only bcr
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

## DETECTING CHARACTERS
""" hImg,wImg,cImg = img.shape
cong = r' --oem 3 --psm 6 outputbase digits'
# print(pytesseract.image_to_boxes(img))
boxes = pytesseract.image_to_boxes(img, lang='por',config=cong)
for b in boxes.splitlines():
    b = b.split(' ')
    print(b)
    x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
    cv2.rectangle(img,(x,hImg-y),(w,hImg-h),(0,0,255),1)
    cv2.putText(img,b[0],(x,hImg-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)  """

## DETECTING WORDS
""" hImg,wImg,cImg = img.shape
boxes = pytesseract.image_to_data(img, lang='eng')
for x,b in enumerate(boxes.splitlines()):
    if x != 0:
        b = b.split()
        print(b)
        if len(b) == 12:
            x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
            cv2.rectangle(img,(x,y),(w+x,h+y),(0,0,255),1)
            cv2.putText(img,b[11],(x,y),cv2.FONT_HERSHEY_COMPLEX,0.4,(50,50,255),1)  """

## DETECTING DIGITS
""" hImg,wImg,cImg = img.shape
cong = r' --oem 3 --psm 6 outputbase digits'
boxes = pytesseract.image_to_data(img, lang='por',config=cong)
for x,b in enumerate(boxes.splitlines()):
    if x != 0:
        b = b.split()
        print(b)
        if len(b) == 12:
            x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
            cv2.rectangle(img,(x,y),(w+x,h+y),(0,0,255),1)
            cv2.putText(img,b[11],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)  """

## !! BETTER RESOLUCTION !!
img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

data = pytesseract.image_to_data(thresh, lang='eng', config=r'--oem 3 --psm 6')
for i, line in enumerate(data.splitlines()):
    if i == 0:
        continue
    b = line.split()
    if len(b) == 12 and float(b[10]) > 50:
        x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(img, b[11], (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (50, 50, 255), 1)

cv2.imshow('Result',img)
cv2.waitKey(0)