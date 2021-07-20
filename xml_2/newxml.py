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

def intersection(rect1,rect2):
    y1,x1,y2,x2 = rect1
    y3,x3,y4,x4 = rect2

    ymin = max(y1,y3)
    xmin = max(x1,x3)
    ymax = min(y2,y4)
    xmax = min(x2,x4)
    # print(rect1,rect2,[ymin,xmin,ymax,xmax])
    if(xmin > xmax or ymin > ymax or xmax-xmin < 40 or ymax-ymin < 40):
        return []
    else:
        # print("True")
        return [ymin,xmin,ymax,xmax]

def normalizeBBImgs(box_key,width,height):
    for x,y in box_key.items():
        temp = x.split("_")
        yfactor,xfactor = int(temp[0]),int(temp[1])
        for bb in y:
            bb[0] = bb[0] - int((yfactor/2)*height/2)
            bb[1] = bb[1] - int((xfactor/2)*width/2)
            bb[2] = bb[2] - int((yfactor/2)*height/2)
            bb[3] = bb[3] - int((xfactor/2)*width/2)
            if(bb[0] == 1 or bb[1] == 1 or bb[2] == height/2 or bb[3] == width/2):
                bb.append(1)
            else:
                bb.append(0)
    

def newSplit(rects,max_height,max_width):
    # for i in range(100,1500,200):
    #     for j in range(100,1500,200):
    #         # print([i,j,i+100,j+100])
    #         rects.append([i,j,i+100,j+100])
    box_map = {}
    # drawImage(rects)
    # Split(rects,0,0,1500,1500)
    # max_width = 1500
    # max_height = 1500
    for i in range(0,3):
        for j in range(0,3):
            xfactor = str(j)
            yfactor = str(i)

            box_map[yfactor+"_"+xfactor] = [i*max_height//4,j*max_width//4,i*max_height//4+max_height//2,j*max_width//4+max_width//2]
    # intersection([0+750,0,1500,750],[300,300,400,400])
    rect_map = {}
    for i in box_map.keys():
        rect_map[i] = []

    for x in rects:
        # print(x)
        for i,j in box_map.items():
            # print(i,j)
            intersected = intersection(x,j)
            if(len(intersected)>0):
                rect_map[i].append(intersected)
                # print(rect_map)

    # for x,y in rect_map.items():
    #     print(x,y)
    #     print("\n\n")

    normalizeBBImgs(rect_map,max_width,max_height)
    # print("___________________________________")
    # for x,y in rect_map.items():
    #     print(x,y)
    #     print("\n\n")
    return rect_map
