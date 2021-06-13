import cv2
import numpy as np
def mapp(h):
    h=h.reshape((4,2))
    hnew=np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0]=h[np.argmin(add)]
    hnew[2]=h[np.argmin(add)]

    diff = np.diff(h,axis=1)
    hnew[1]=h[np.argmin(diff)]
    hnew[3]=h[np.argmin(diff)]
    return hnew

image = cv2.imread("test_img.jpg")
image = cv2.resize(image, (600,700))
orig = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("testing", gray)

blurred = cv2.GaussianBlur(gray, (5,5), 0)
cv2.imshow("blur testing",blurred)

edged = cv2.Canny(blurred,30,50)
cv2.imshow("Canny", edged)

#
contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  #retrieve the contours as a list, with simple apprximation model
contours=sorted(contours,key=cv2.contourArea,reverse=True)

for c in contours:
    p=cv2.arcLength(c , True)
    approx = cv2.approxPolyDP(c,0.02*p , True)
    if len(approx)==4:
        target = approx
        break
approx = mapp(target)

pts = np.float32([[0,0],[800,0],[800,800],[0,800]])
op = cv2.getPerspectiveTransform(approx,pts)
dst=cv2.warpPerspective(orig,op,(800,800))

cv2.imshow("scanned",dst)
cv2.waitKey()
