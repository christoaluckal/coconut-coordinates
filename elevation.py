import gdal
import numpy as np
import affine
import sys
import time
demdata = gdal.Open(str(sys.argv[1]))
# demarray = np.array(demdata.GetRasterBand(1).ReadAsArray())
# dem_width,dem_height = demarray.shape
# print(dem_width,dem_height)
band = demdata.GetRasterBand(1)
# block_sizes = band.GetBlockSize()
# x_block_size = block_sizes[0]
# y_block_size = block_sizes[1]
# xsize = band.XSize
# ysize = band.YSize
# # store = []
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
#             print(x, y, cols, rows)
#             array = band.ReadAsArray(x, y, cols, rows)
#             # print(array)
#             blocks += 1
#             # print(blocks)
#     band = None
#     ds = None
#     # print "{0} blocks size {1} x {2}:".format(blocks, x_block_size, y_block_size)
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
def processRegion(x_min,y_min,array):
    # print(x_min,y_min,x_max,y_max,dem_width,dem_height)
    index_of_highest = list(np.unravel_index(np.argmax(array, axis=None), array.shape))
    x_highest,y_highest = index_of_highest[0],index_of_highest[1]
    print(array.shape,x_highest,y_highest)
    height = open('coords_height.txt','a')
    height.write(str(getLonLat(x_min+x_highest,y_min+y_highest))+"  "+str(array[x_highest][y_highest])+"\n")
    # print(str(getLonLat(x_min+x_highest,y_min+y_highest))+"  "+str(array[x_highest][y_highest]))

# 22022	5611	22398	6053	

# ReadAsArray(x_min,y_min,x_len,y_len)

# indices = np.where(area == area.max())

# Temporarily Passing Coordinates directly.
# TODO minn,maxx will be replaced programatically
# min_coords = getXY(15.1079367,74.1492764)
# max_coords = getXY(15.1077252,74.1495008)
# print(min_coords,max_coords)
# print(getLonLat(min_coords[0],min_coords[1]))
# area = band.ReadAsArray(22159,24117,23249-22159,24997-24117)
# processRegion(22159,24117,area)


coords = open('raw.txt','r')
for x in coords.readlines():
    li = x.split('\t')
    y_min,x_min,y_max,x_max = int(li[0]),int(li[1]),int(li[2]),int(li[3])
    area = band.ReadAsArray(x_min,y_min,x_max-x_min,y_max-y_min)
    processRegion(x_min,y_min,area)

# 10661	10705	10891	10940

