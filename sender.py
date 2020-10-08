from __future__ import print_function
import requests
import json
global check
import cv2
from testtext import texti
from logosend import logosend
import pytesseract
from crop2 import crop2
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Srivatsan\AppData\Local\Tesseract-OCR\tesseract.exe'
check=''
global dock
dock=list()
global left,right,top,bottom
lef=[]
righ=[]
tp=[]
botto=[]
global textres
textres=list()
global logtitres
logtitres=list()
def canny(image):
    return cv2.Canny(image, 100, 200)
def sendi(imgi):
    textres=list()
    test_url = 'https://srivatsan.cognitiveservices.azure.com/customvision/v3.0/Prediction/09b6cec3-e166-44be-8830-b7c0804228d2/detect/iterations/checkboxacc88re80map80%2B/image'
    #test_url = addr + '/api/test'
    # prepare headers for http request
    headers = {'Prediction-Key': "015257f7093647ad9b57ac05e3b6c085","Content-Type":"application/octet-stream"}
    logtitres=list()
    img_file = "uploads/"+ imgi
    #crop2(img_file)

    # encode image as jpe
    #_, img_encoded = cv2.imencode('
    img = open(img_file, 'rb').read()
    immg=cv2.imread(img_file,0)
    out=cv2.imread(img_file,0)
    # send http request with image and receive response
    response = requests.post(test_url,img, headers=headers)
    # decode response
    tq=list()
    #print(json.loads(response.text))
    check=json.loads(response.text)
    lic=len(check['predictions'])
    for i in range(0,lic):
        if(check['predictions'][i]['probability']>0.4):
            dock.append(check['predictions'][i])
            lef.append(check['predictions'][i]['boundingBox']['left'])
            righ.append(check['predictions'][i]['boundingBox']['width'])
            tp.append(check['predictions'][i]['boundingBox']['top'])
            botto.append(check['predictions'][i]['boundingBox']['height'])
    h,w=immg.shape
    for i in range(0,len(dock)):
        lt=int(lef[i]*w)
        t=int(tp[i]*h)
        r=int(righ[i]*w)+80
        g=int(botto[i]*h)
        pt1=(lt,t)
        pt2=((lt+r),(t+g))
        color=(120,115,60)
        thick=2
        roi=[lt,t,r,g]
        #gray = get_grayscale(immg)
        #canny = cv2.Canny(immg, 100, 200)
        roi_cropped = immg[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        height,width=roi_cropped.shape
        print(height)
        print(width)
        if(height<50):
            height=55
        if (width<50):
            width=55
        print(height)
        print(width)
        dim=(width,height)
        roi_cropped=cv2.resize(roi_cropped, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(("savedimg"+str(imgi)+".jpg"),roi_cropped)
        out=cv2.rectangle(out,pt1,pt2,color,thick)
        #print(pytesseract.image_to_string(("savedimg"+str(imgi)+".jpg")))
        textres=texti(("savedimg"+str(imgi)+".jpg"))

        #img_crop=img_raw[y1:y1+y2,x1:x1+x2]
        #cv2.imwrite("cropped.jpg", img_raw)
       # tic=out[pt1,pt2]
        #cv2.imwrite("tic.jpg",tic)
    path="outputs/"+imgi
    cv2.imwrite(path,out)
    teck=check['predictions'][0]
    wreck=teck['boundingBox']
    left=wreck['left']
    right=wreck['width']
    top=wreck['top']
    bottom=wreck['height']
    
    left=int(left*w)+30
    top=int(top*h)
    rw=int(right*w)+40
    hw=int(bottom*h)
    
    p1=(left,top)
    p2=((left+rw),(top+hw))
    color = (0, 0, 0)
    thick=0
    image = cv2.rectangle(immg, p1, p2, color, thick) 
    #cv2.imwrite("rect",img)
   # print(img)
    cv2.imwrite("uploads/image.jpg",image)
    tq.extend([left,right,top,bottom])
    logtitres=logosend(img_file)
    return [logtitres,textres]
