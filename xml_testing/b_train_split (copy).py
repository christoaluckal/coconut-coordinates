import numpy as np
# import xml.etree.cElementTree as ET
from lxml.etree import tostring
import lxml.etree as ET
box_list = [[20,20,40,40],[20,50,40,70],[20,80,40,100],[50,20,70,40],[50,50,70,70],[50,80,70,100],[80,20,100,40],[80,50,100,70],[80,80,100,100]]
# final_height = 120
# final_width = 120
# color = (255,255,255)
# channels = 3
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

height,width = 120,120

def getBBKeys(box_list,height,width):
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
            bb[0] = bb[0] - int((yfactor/2)*width/2)
            bb[1] = bb[1] - int((xfactor/2)*height/2)
            bb[2] = bb[2] - int((yfactor/2)*width/2)
            bb[3] = bb[3] - int((xfactor/2)*height/2)


def getAnnotatedData(xml_path,tag):
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
    tree = ET.parse(xml_path)
    statlist = tree.xpath(xpath_dict[tag])
    print(statlist[0].text)


def createAnnoatationXML():
    annotation = ET.Element("annotation")
    folder = ET.SubElement(annotation,"folder").text="test"
    filename = ET.SubElement(annotation,"filename")
    path = ET.SubElement(annotation,"path")
    source = ET.SubElement(annotation,"source")
    database = ET.SubElement(source,"database").text = "Unknown"
    size = ET.SubElement(annotation,"size")
    width = ET.SubElement(size,"width")
    height = ET.SubElement(size,"height")
    depth = ET.SubElement(size,"depth")
    segmented = ET.SubElement(annotation,"segmented").text = "0"
    object = ET.SubElement(annotation,"object")
    name = ET.SubElement(object,"name")
    pose = ET.SubElement(object,"pose")
    truncated = ET.SubElement(object,"truncated")
    difficulty = ET.SubElement(object,"difficulty")
    bndbox = ET.SubElement(object,"name")
    xmin = ET.SubElement(bndbox,"xmin")
    ymin = ET.SubElement(bndbox,"ymin")
    xmax = ET.SubElement(bndbox,"xmax")
    ymax = ET.SubElement(bndbox,"ymax")
    
    tree = ET.ElementTree(annotation)
    
    xml_object = tostring(annotation,
                                pretty_print=True,
                                xml_declaration=False,
                                encoding='UTF-8')

    with open("xmlfile.xml", "wb") as writter:
        writter.write(xml_object)


# createAnnoatationXML()
getAnnotatedData("DJI_0859.xml","width")

tree = ET.parse("DJI_0859.xml")
root = tree.getroot()
objects = root.findall('object/bndbox/xmin')
# objects = tree.xpath('object/bndbox/xmin')
for x in objects:
    tree = ET.tostring(x)
    # statlist = tree.xpath('object/bndbox/xmin')
    # print(statlist[0].text)
    print(tree)