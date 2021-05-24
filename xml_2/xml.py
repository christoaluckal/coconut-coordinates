import cv2
import numpy as np
# def drawImage(rects):
#     result = np.full((1500,1500,3),(255,255,255),dtype=np.uint8)
#     height,width,channels = result.shape
#     cv2.line(result,(750,0),(750,1500),(0,0,0),thickness=1)
#     cv2.line(result,(0,750),(1500,750),(0,0,0),thickness=1)

#     cv2.line(result,(750//2-25,0),(750//2-25,1500),(0,255,0),thickness=1)
#     cv2.line(result,(0,750//2-25),(1500,750//2-25),(0,255,0),thickness=1)
#     # cv2.line(result,(750*3//2-25,0),(750*3//2-25,1500),(0,255,0),thickness=1)
#     # cv2.line(result,(0,750),(1500,740),(0,255,0),thickness=1)
#     cv2.line(result,(750*3//2+25,0),(750*3//2+25,1500),(0,255,0),thickness=1)
#     cv2.line(result,(0,750*3//2+25),(1500,750*3//2+25),(0,255,0),thickness=1)
#     for x in rects:
#         cv2.rectangle(result,x,(0,0,255))

#     cv2.imwrite("test.jpg",result)
rects = []
box_0_0 = []
box_0_1 = []
box_1_0 = []
box_1_1 = []
def Split(box_list,tly,tlx,bry,brx):
    print("entered split")
    for i in box_list:
        # print(i)
        y_min,x_min,y_max,x_max = i[0],i[1],i[2],i[3]
        if(y_min < (bry-tly)//2 and x_min < (brx-tlx)//2 and y_max < (bry-tly)//2 and x_max < (brx-tlx)//2):
            box_0_0.append([y_min,x_min,y_max,x_max])
        elif(y_min < (bry-tly)//2 and x_min < (brx-tlx)//2 and y_max < (bry-tly)//2 and not (x_max < (brx-tlx)//2)):
            pass
        elif(y_min < (bry-tly)//2 and x_min < (brx-tlx)//2 and not(y_max < (bry-tly)//2) and (x_max < (brx-tlx)//2)):
            pass
        elif(y_min < (bry-tly)//2 and x_min < (brx-tlx)//2 and not (y_max < (bry-tly)//2) and not (x_max < (brx-tlx)//2)):
            pass
        elif(y_min < (bry-tly)//2 and not(x_min < (brx-tlx)//2) and y_max < (bry-tly)//2):
            box_0_1.append([y_min,x_min,y_max,x_max])
        elif(y_min < (bry-tly)//2 and not(x_min < (brx-tlx)//2) and not(y_max < (bry-tly)//2)):
            pass
        elif(not(y_min < (bry-tly)//2) and x_min < (brx-tlx)//2 and x_max < (brx-tlx)//2):
            box_1_0.append([y_min,x_min,y_max,x_max])
        elif(y_min < (bry-tly)//2 and x_min < (brx-tlx)//2 and not(x_max < (brx-tlx)//2)):
            pass
        else:
            box_1_1.append([y_min,x_min,y_max,x_max])

    print(box_0_0)
    print(box_0_1)
    print(box_1_0)
    print(box_1_1)
    avg_y = (tly+bry)//2
    avg_x = (tlx+brx)//2
    # if(len(box_0_0)!=1):
    #     Split(box_0_0,tly,tlx,avg_y,avg_x)
    # if(len(box_0_1)!=1):
    #     Split(box_0_1,tly,avg_x,avg_y,brx)
    # if(len(box_1_0)!=1):
    #     Split(box_1_0,avg_y,tlx,bry,avg_x)
    # if(len(box_1_1)!=1):
    #     Split(box_1_1,avg_y,avg_x,bry,brx)

    

for i in range(100,1500,200):
    for j in range(100,1500,200):
        # print([i,j,i+100,j+100])
        rects.append([i,j,100,100])

# drawImage(rects)
Split(rects,0,0,1500,1500)