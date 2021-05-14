import numpy as np
import cv2

image_map = {} # This data structure is used to store coordinates of each sub-part
# The format of image_map is: for (i,j)th part [(y1,x1),(y2,x2)] ie standard image format where (y1,x1) are top left and (y2,x2) are bottom right coordinates

def sampleImage(img):
    height,width,channels = img.shape
    n_white_pix = np.sum(img == 255)
    n_white_pix = n_white_pix //3
    total_pixels = height*width
    if(n_white_pix/total_pixels > 0.96):
        return False
    else:
        return True

def padImage(img,name,default_pad):
    height,width,channels = img.shape
    # print(height,width)
    aspect_ratio = float(width/height)
    if aspect_ratio > 2.0 or aspect_ratio < 0.5:
        print("Image is not close to standard aspect ratio\n")
        exit()
    else:
        temp_width = width
        width_pad = 0
        temp_height = height
        height_pad = 0
        if width%default_pad[1] !=0:
            # print("Padding width")
            while(temp_width%default_pad[1]!=0):
                temp_width+=1
                width_pad+=1
        if height%default_pad[0] !=0:
            # print("Padding height")
            while(temp_height%default_pad[0]!=0):
                temp_height+=1
                height_pad+=1

        # print(height_pad,width_pad)
        final_width = width+width_pad
        final_height = height+height_pad
        print("Image Padded to :"+str(final_height)+"x"+str(final_width))
        color = (255,255,255)
        result = np.full((final_height,final_width,channels), color, dtype=np.uint8)

        # compute center offset
        # xx = (final_width - width) // 2
        # yy = (final_height - height) // 2

        # copy img image into center of result image
        # result[yy:yy+height, xx:xx+width] = img
        result[:height,:width] = img
        # save result
        return result,name
        
# def getSplitFactor(padded_img,name):
#     height,width,channels = padded_img.shape
#     base_size = 0
#     pad_factor = 0
#     if width < 10000:
#         base_size = 10
#         pad_factor = 1
#     elif width < 20000:
#         base_size = 5
#         pad_factor = 2
#     else:
#         base_size = 2
#         pad_factor = 3
    
#     return 3648,5472

def cropImage(img,h_split_num,v_split_num):
    white = 0
    sum_total = 0
    height,width,channel = img.shape
    # print(height,v_split_num,width,h_split_num)
    # print(height//v_split_num)
    # print(width//h_split_num)
    total = str((height//v_split_num)*(width//h_split_num))
    print("Splitting image into "+total+" parts")
    for i in range(0,height//v_split_num):
        for j in range(0,width//h_split_num):
            # print(i,j)
            x_min,y_min = image_map[(i,j)][0][1],image_map[(i,j)][0][0]
            x_max,y_max = image_map[(i,j)][1][1],image_map[(i,j)][1][0]
            # print([i,j],image_map[(i,j)],x_min,y_min,x_max,y_min) #IMPORTANT  image_map stores into standard image format while XY min/max prints as human readable ie row first then column
            temp_img = img[y_min:y_max,x_min:x_max]
            if sampleImage(temp_img):
            # print("images/"+str(j)+str(i)+".jpg")
                cv2.imwrite("images/"+str(i)+str(j)+".jpg",temp_img)
            else:
                white+=1
                # print("Popping",(i,j))
                image_map.pop((i,j))
            # print("End") 
            sum_total+=1
    print("Total white percentage:",str((white/sum_total)*100))
    # print(image_map)


def splitImage(img,name,default_size): # Notice the inversion of notations
    height,width = img.shape[0],img.shape[1]
    v_split_num = default_size[0]
    h_split_num = default_size[1]
    for i in range(0,height//v_split_num): # i is the row iterator ie ith row of a particular column
        for j in range(0,width//h_split_num): # j is the column iterator ie jth row of a particular IMAGE
            # image_map[j,i] = [[j*h_split_num,i*v_split_num],[(j+1)*h_split_num,(i+1)*v_split_num]]
            image_map[i,j] = [[i*v_split_num,j*h_split_num],[(i+1)*v_split_num,(j+1)*h_split_num]]
            # print([i,j],image_map[i,j])
        # print(image_map)
    # print(image_map)
    cropImage(img,h_split_num,v_split_num)
    # sampleImage(img)



    

    
choice = int(input("1:Blank\t2:Half\t3:Color\t4:Small Iris"))
if choice == 1:
    name = 'top_right_blank_Mavic_Full_PNG.png'
elif choice == 2:
    name = 'mid_right_half_Mavic_Full_PNG.png'
elif choice == 3:
    name = 'coconut_Mavic_Full_PNG.png'
elif choice == 4:
    name = '416_435.jpg'
else:
    print("Error")
    exit()

img = cv2.imread('big.png',-1)
# imS = cv2.resize(img, (960, 540))
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print("Standard image height and width?")
default_size = int(input()),int(input())
padded_img,name = padImage(img,'big.png',default_size)
# print("FINAL PAD:",padded_img.shape)
splitImage(padded_img,name,default_size)



