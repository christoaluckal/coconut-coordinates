rects = open('outputs/text_files/bounding_box.txt')
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2

img = cv2.imread('DBCAoutput.jpg')
dup = img

for x in rects.readlines():
    line = x.split()
    ymin,xmin,ymax,xmax = int(line[0]),int(line[1]),int(line[2]),int(line[3])
    cv2.rectangle(dup,(ymin,xmin),(ymax,xmax),(255,255,0),thickness=1)

cv2.imwrite("op_bb.jpg",dup)