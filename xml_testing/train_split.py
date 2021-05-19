import numpy as np
# import xml.etree.cElementTree as ET
from lxml.etree import tostring
import lxml.etree as ET
    
import cv2
# box_list = [[20,20,40,40],[20,50,40,70],[20,80,40,100],[50,20,70,40],[50,50,70,70],[50,80,70,100],[80,20,100,40],[80,50,100,70],[80,80,100,100]]
# final_height = 120
# final_width = 120
color = (255,255,255)
channels = 3
# result = np.full((final_height,final_width,channels), color, dtype=np.uint8)
# import cv2

# for x in box_list:
#     x1,y1,x2,y2 = x[0],x[1],x[2],x[3]
#     cv2.rectangle(result, (x1, y1), (x2, y2), (255,0,0), 2)

# cv2.imwrite("image.jpg",result)
# # cv2.waitKey(0)
# box_0_0 = []
# box_0_1 = []
# box_1_0 = []
# box_1_1 = []
# box_0_2 = []
# box_1_2 = []
# box_2_1 = []
# box_2_0 = []
# box_2_2 = []

# height,width = 120,120
# color = (255,255,255)
# channels = 3

def testDraw(bblist,op):
    result = np.full((3648//2,5472//2,channels), color, dtype=np.uint8)
    print(bblist)
    for x in bblist:
        x1,y1,x2,y2 = x[1],x[0],x[3],x[2]
        cv2.rectangle(result, (x1, y1), (x2, y2), (255,0,0), 2)
    cv2.imwrite(op+".JPG",result)

def getBBKeys(box_list,width,height):
    box_key = {}
    for i in range(0,3):
        for j in range(0,3):
            box_key[str(i)+"_"+str(j)] = []
    for x in box_list:
        y_min,x_min,y_max,x_max = int(x[0]),int(x[1]),int(x[2]),int(x[3])
        if(y_min < height//2):
            if(x_min < width//2):
                if(y_max < height//2):
                    if(x_max < width//2):
                        box_key["0_0"].append([y_min,x_min,y_max,x_max])
                    else:
                        box_key["0_1"].append([y_min,x_min,y_max,x_max])
                else:
                    if(x_max < width //2):
                        box_key["1_0"].append([y_min,x_min,y_max,x_max])
                    else:
                        box_key["1_1"].append([y_min,x_min,y_max,x_max])
            else:
                if(y_max < height//2):
                    box_key["0_2"].append([y_min,x_min,y_max,x_max])
                else:
                    box_key["1_2"].append([y_min,x_min,y_max,x_max])
        else:
            if(x_min < width//2):
                if(x_max < width//2):
                    box_key["2_0"].append([y_min,x_min,y_max,x_max])
                else:
                    box_key["2_1"].append([y_min,x_min,y_max,x_max])
            else:
                box_key["2_2"].append([y_min,x_min,y_max,x_max])
    return box_key

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


def getAnnotatedData(tree,tag):
    xpath_dict = {}
    xpath_dict["annotation"] = '/annotation'
    xpath_dict["folder"] = '/annotation/folder'
    xpath_dict["filename"] = '/annotation/filename'
    xpath_dict["path"] = '/annotation/path'
    xpath_dict["source"] = '/annotation/source'
    xpath_dict["size"] = '/annotation/size'
    xpath_dict["width"] = '/annotation/size/width'
    xpath_dict["height"] = '/annotation/size/height'
    xpath_dict["depth"] = '/annotation/size/depth'
    xpath_dict["segmentation"] = '/annotation/segmentation'
    xpath_dict["object"] = '/annotation/object'
    xpath_dict["name"] = '/annotation/object/name'
    xpath_dict["pose"] = '/annotation/object/pose'
    xpath_dict["truncated"] = '/annotation/object/truncated'
    xpath_dict["difficulty"] = '/annotation/object/difficulty'
    xpath_dict["bndbox"] = '/annotation/object/bndbox'
    xpath_dict["xmin"] = '/annotation/object/bndbox/xmin'
    xpath_dict["ymin"] = '/annotation/object/bndbox/ymin'
    xpath_dict["xmax"] = '/annotation/object/bndbox/xmax'
    xpath_dict["ymax"] = '/annotation/object/bndbox/ymax'
    op = {}
    # tree = ET.parse(xml_path)
    for x in tag:
        statlist = tree.xpath(xpath_dict[x])
        op[x] = statlist[0].text
    return op


def createAnnotationXML(cfolder,cfilename,cpath,cwidth,cheight,objects,region):
    if len(objects)!=0:
        result = np.full((cheight,cwidth,channels), color, dtype=np.uint8)
        y_min,x_min,y_max,x_max = region
        img = cv2.imread(cfilename[:-6]+".JPG")
        result = img[y_min:y_max,x_min:x_max]
        annotation = ET.Element("annotation")
        folder = ET.SubElement(annotation,"folder").text= cfolder
        filename = ET.SubElement(annotation,"filename").text = cfilename
        path = ET.SubElement(annotation,"path").text = cpath
        source = ET.SubElement(annotation,"source")
        database = ET.SubElement(source,"database").text = "Unknown"
        size = ET.SubElement(annotation,"size")
        width = ET.SubElement(size,"width").text = str(cwidth)
        height = ET.SubElement(size,"height").text = str(cheight)
        depth = ET.SubElement(size,"depth").text = "3"
        segmented = ET.SubElement(annotation,"segmented").text = "0"
        for box in objects:
            object = ET.SubElement(annotation,"object")
            name = ET.SubElement(object,"name").text = "coconut"
            pose = ET.SubElement(object,"pose").text = "Unspecified"
            truncated = ET.SubElement(object,"truncated").text = str(box[4])
            difficulty = ET.SubElement(object,"difficulty").text = "0"
            bndbox = ET.SubElement(object,"bndbox")
            xmin = ET.SubElement(bndbox,"xmin").text = str(box[1])
            ymin = ET.SubElement(bndbox,"ymin").text = str(box[0])
            xmax = ET.SubElement(bndbox,"xmax").text = str(box[3])
            ymax = ET.SubElement(bndbox,"ymax").text = str(box[2])
        
        tree = ET.ElementTree(annotation)
        
        xml_object = tostring(annotation,
                                    pretty_print=True,
                                    xml_declaration=False,
                                    encoding='UTF-8')

        with open(filename[:-4]+".xml", "wb") as writer:
            writer.write(xml_object)
        cv2.imwrite(cfilename,result)
    else:
        pass
    
def splitDraw(file_xml,op):
    tags_to_search = ['folder','filename','path','width','height']
    # createAnnoatationXML()


    tree = ET.parse(file_xml)
    file_map = {}
    count = 1
    for x in range(0,3):
        for y in range(0,3):
            file_map[str(x)+"_"+str(y)] = str(count)
            count+=1


    base_data = getAnnotatedData(tree,tags_to_search)
    width = int(base_data["width"])
    height = int(base_data["height"])
    root = tree.getroot()
    xmins = root.findall('object/bndbox/xmin')
    xmaxs = root.findall('object/bndbox/xmax')
    ymins = root.findall('object/bndbox/ymin')
    ymaxs = root.findall('object/bndbox/ymax')
    main_bb = []
    for a,b,c,d in zip(ymins,xmins,ymaxs,xmaxs):
        main_bb.append([int(a.text),int(b.text),int(c.text),int(d.text)])
    testDraw(main_bb,op)

def split(file_xml):
    tags_to_search = ['folder','filename','path','width','height']
    # createAnnoatationXML()


    tree = ET.parse(file_xml)
    file_map = {}
    count = 1
    for x in range(0,3):
        for y in range(0,3):
            file_map[str(x)+"_"+str(y)] = str(count)
            count+=1


    base_data = getAnnotatedData(tree,tags_to_search)
    width = int(base_data["width"])
    height = int(base_data["height"])
    root = tree.getroot()
    xmins = root.findall('object/bndbox/xmin')
    xmaxs = root.findall('object/bndbox/xmax')
    ymins = root.findall('object/bndbox/ymin')
    ymaxs = root.findall('object/bndbox/ymax')
    main_bb = []
    for a,b,c,d in zip(ymins,xmins,ymaxs,xmaxs):
        main_bb.append([int(a.text),int(b.text),int(c.text),int(d.text)])
    main_bb_keys = getBBKeys(main_bb,width,height)
    normalizeBBImgs(main_bb_keys,width,height)
    for x,y in main_bb_keys.items():
        cfolder = base_data["folder"]
        cfilename = base_data["filename"][:-4]+"_"+file_map[x]+".JPG"
        cpath = base_data["path"][:-4]+"_"+file_map[x]+".JPG"
        cwidth = int(width/2)
        cheight = int(height/2)
        temp = x.split("_")
        yfactor,xfactor = int(temp[0]),int(temp[1])
        corner_y = int((yfactor/2)*height/2)
        corner_x = int((xfactor/2)*width/2)
        region = (corner_y,corner_x,corner_y+cheight,corner_x+cwidth)
        createAnnotationXML(cfolder,cfilename,cpath,cwidth,cheight,y,region)

split("DJI_0859.xml")

splitDraw("DJI_0859_1.xml","1")
# split("DJI_0859_7.xml","7")
# split("DJI_0859_9.xml","9")