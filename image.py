import numpy as np
import cv2

image_map = {}

def cropImage(img,h_split_num,v_split_num):
    for i in range(0,img.shape[0]//v_split_num):
        for j in range(0,img.shape[1]//h_split_num):
            x_min,y_min = image_map[(i,j)][0][0],image_map[(i,j)][0][1]
            x_max,y_max = image_map[(i,j)][1][0],image_map[(i,j)][1][1]
            temp_img = img[x_min:x_max,y_min:y_max]
            cv2.imwrite("images/"+str(i)+str(j)+".jpg",temp_img) 

def splitImage(img):
    height,width = img.shape[0],img.shape[1]
    v_split_num = int(0.1*height)
    h_split_num = int(0.1*width)
    # v_padding = 0
    # h_padding = 0
    # if(height%10 is not 0):
    #     temp_height = height
    #     while(temp_height%10!=0):
    #         v_padding+=1
    #         temp_height+=1
    # if(width%10 is not 0):
    #     temp_width = width
    #     while(temp_width%10!=0):
    #         h_padding+=1
    #         temp_width+=1
    for i in range(0,height//v_split_num):
        for j in range(0,width//h_split_num):
            image_map[j,i] = [[j*v_split_num,i*h_split_num],[(j+1)*v_split_num,(i+1)*h_split_num]]

    print(image_map[(0,8)],image_map[(0,9)])
    cropImage(img,h_split_num,v_split_num)



    

    

# Load an color image in grayscale
img = cv2.imread('iris.jpg')
# imS = cv2.resize(img, (960, 540))
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
splitImage(img)

