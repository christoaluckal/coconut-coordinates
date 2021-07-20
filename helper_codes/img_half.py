import os
import cv2
import sys
args = sys.argv[1:]

input_dir = args[0]
img_list = os.listdir(input_dir)
output_dir = args[1]

width = 5472
height = 3648

for x in img_list:
    count = 0
    print(input_dir+x)
    full_img = cv2.imread(input_dir+x)
    tl_img = full_img[0:0+height//2,0:0+2736]
    print(tl_img.shape)
    tr_img = full_img[0:height//2,(width//2)+1:width]
    bl_img = full_img[(height//2)+1:height,0:width//2]
    br_img = full_img[(height//2)+1:height,(width//2)+1:width]
    # print(count)
    cv2.imwrite(output_dir+x[:-4]+"_"+str(count)+".JPG",tl_img)
    count+=1
    print(count)
    cv2.imwrite(output_dir+x[:-4]+"_"+str(count)+".JPG",tr_img) #2
    count+=1
    print(count)
    cv2.imwrite(output_dir+x[:-4]+"_"+str(count)+".JPG",bl_img)
    count+=1
    print(count)
    cv2.imwrite(output_dir+x[:-4]+"_"+str(count)+".JPG",br_img)
    