import numpy as np
import cv2

image_map = {}

def sampleImage(img):
    height,width,channels = img.shape
    n_white_pix = np.sum(img == 255)
    n_white_pix = n_white_pix //3
    total_pixels = height*width
    if(n_white_pix/total_pixels > 0.95):
        return False
    else:
        return True

def padImage(img,name):
    height,width,channels = img.shape
    aspect_ratio = float(width/height)
    if aspect_ratio > 2.0 or aspect_ratio < 0.5:
        print("Image is not close to standard aspect ratio\n")
        exit()
    else:
        base_size = 0
        pad_factor = 0
        if width < 10000:
            base_size = 10
            pad_factor = 1
        elif width < 20000:
            base_size = 5
            pad_factor = 2
        else:
            base_size = 2
            pad_factor = 3
        
        mod_factor = pow(10,pad_factor)
        temp_width = width
        width_pad = 0
        temp_height = height
        height_pad = 0
        if width%mod_factor !=0:
            print("Padding width")
            while(temp_width%mod_factor!=0):
                temp_width+=1
                width_pad+=1
        if height%mod_factor !=0:
            print("Padding height")
            while(temp_height%mod_factor!=0):
                temp_height+=1
                height_pad+=1

        
        final_width = width+width_pad
        final_height = height+height_pad
        color = (255,255,255)
        result = np.full((final_height,final_width,channels), color, dtype=np.uint8)

        # compute center offset
        # xx = (final_width - width) // 2
        # yy = (final_height - height) // 2

        # copy img image into center of result image
        # result[yy:yy+height, xx:xx+width] = img
        result[:height,:width] = img

        # save result
        new_name = "corner_"+name
        cv2.imwrite(new_name, result)
        


def cropImage(img,h_split_num,v_split_num):
    for i in range(0,img.shape[0]//v_split_num):
        for j in range(0,img.shape[1]//h_split_num):
            x_min,y_min = image_map[(i,j)][0][0],image_map[(i,j)][0][1]
            x_max,y_max = image_map[(i,j)][1][0],image_map[(i,j)][1][1]
            temp_img = img[x_min:x_max,y_min:y_max]
            cv2.imwrite("images/"+str(i)+str(j)+".jpg",temp_img) 

def splitImage(img,name):
    height,width = img.shape[0],img.shape[1]
    v_split_num = int(0.1*height)
    h_split_num = int(0.1*width)
    for i in range(0,height//v_split_num):
        for j in range(0,width//h_split_num):
            image_map[j,i] = [[j*v_split_num,i*h_split_num],[(j+1)*v_split_num,(i+1)*h_split_num]]

    # print(image_map[(0,8)],image_map[(0,9)])
    # cropImage(img,h_split_num,v_split_num)
    padImage(img,name)
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

img = cv2.imread(name,-1)
# imS = cv2.resize(img, (960, 540))
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
splitImage(img,name)

