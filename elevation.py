import gdal
import numpy as np
import affine
import sys
import time
demdata = gdal.Open(str(sys.argv[1]))
demarray = np.array(demdata.GetRasterBand(1).ReadAsArray())
dem_width,dem_height = demarray.shape
# print(dem_width,dem_height)
# band = demdata.GetRasterBand(1)
# block_sizes = band.GetBlockSize()
# x_block_size = block_sizes[0]
# y_block_size = 5
# xsize = band.XSize
# ysize = band.YSize
# store = []
# def read_raster():
#     raster = str(sys.argv[1])
#     ds = gdal.Open(raster)
#     band = ds.GetRasterBand(1)
#     xsize = band.XSize
#     ysize = band.YSize
#     blocks = 0
#     for y in range(0, ysize, y_block_size):
#         if y + y_block_size < ysize:
#             rows = y_block_size
#         else:
#             rows = ysize - y
#         for x in range(0, xsize, x_block_size):
#             if x + x_block_size < xsize:
#                 cols = x_block_size
#             else:
#                 cols = xsize - x
#             print(x,y,cols,rows)
#             # time.sleep(1)
#             array = band.ReadAsArray(x, y, cols, rows)
#             store.append(array)
#             blocks += 1
#             print(blocks)
#     band = None
#     ds = None
#     # print "{0} blocks size {1} x {2}:".format(blocks, x_block_size, y_block_size)

# read_raster()
# # 10661	10705	10891	10940
# # print(store)
"""
Temporary Onclicking Function. Only works with DEM but cannot see anything.
Ortho too big and coordinates dont match. TODO Fix this
"""

# import matplotlib.pyplot as plt
# import cv2
# def onclick(event): 
#     print("button=%d, x=%d, y=%d, xdata=%i, ydata=%i" % ( 
#          event.button, event.x, event.y, event.xdata, event.ydata)) 

# img = cv2.imread(str(sys.argv[1]))
# ax = plt.imshow(img)
# fig = ax.get_figure()
# cid = fig.canvas.mpl_connect('button_press_event', onclick) 

# plt.show()

"""
End of Onclickling
"""


"""
Sample for getting highest location
"""
# indices = np.where(demarray == demarray.max())
# ymax, xmax = indices[0][0], indices[1][0]
# print("The highest point is", demarray[ymax][xmax])
# print("  at pixel location", xmax, ymax)

# print(demarray.min(), demarray.max())
"""
End of Sample
"""

# Affine Transforms to Convert X,Y to LatLon and vice-versa
affine_transform = affine.Affine.from_gdal(*demdata.GetGeoTransform())

# Function to get the LatLon of an X,Y coords. 
# CAUTION: Affine reverses values, maintain order
def getLonLat(x_coord,y_coord):
    lon, lat = affine_transform * (x_coord, y_coord)
    return (lon,lat)


# Function to get the X,Y from LatLon. 
# CAUTION: Affine reverses values, maintain order
def getXY(lat,lon):
    inverse_transform = ~affine_transform
    x_coord, y_coord = [ round(f) for f in inverse_transform * (lon, lat) ]
    return (x_coord,y_coord)

# Processing Function
# TODO Use more complicated functions instead of max
def processRegion(x_min,y_min,x_max,y_max):
    # print(x_min,y_min,x_max,y_max,dem_width,dem_height)
    try:
        assert isinstance(x_min,int) and isinstance(y_min,int) and isinstance(x_max,int) and isinstance(y_max,int)
    except:
        print("ERROR: Region Coords must be integers")
        return
    try:
        assert x_min > 0 and y_min > 0 and  x_max <= dem_width  and y_max <= dem_height
    except:
        print("ERROR: Out of Bounds")
        return

    # Assuming getting max
    region = []
    for row in range(x_min,x_max):
        for col in range(y_min,y_max):
            region.append([row,col,demarray[col][row]])

    # 
    region_heights = np.asarray(region)
    heights = region_heights[:, [2]]
    index_of_highest = np.where(heights == heights.max())[0][0]
    x_highest,y_highest = region[index_of_highest][0],region[index_of_highest][1]
    height = open('coords_height.txt','a')
    height.write(str(getLonLat(x_highest,y_highest))+"  "+str(heights[index_of_highest][0])+"\n")



# Temporarily Passing Coordinates directly.
# TODO minn,maxx will be replaced programatically
# min_coords = getXY(19.487784,73.140601)
# max_coords = getXY(19.486528,73.141831)
# print(min_coords,max_coords)
coords = open('raw.txt','r')
for x in coords.readlines():
    li = x.split('\t')
    y_min,x_min,y_max,x_max = int(li[0]),int(li[1]),int(li[2]),int(li[3])
    processRegion(x_min,y_min,x_max,y_max)

# 10661	10705	10891	10940

