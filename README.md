# coconut-coordinates

a.  The function of this code is to be able to extract the coordinates of coconut trees from a pre-acquired Orthomosaic and DEM image pair of a location<br>
b.  The code first takes the Orthomosaic image and determines the coordinates of the coconut trees using Tensorflow and the COCO API <br>
c.  Then the elevation script reads the coordinates outputted by the Orthomosaic processing and finds the elevation of these coordinates using the DEM image <br>

Download the files [here](https://1drv.ms/u/s!AiSJsfSikINmg40IF2dxOpKxb9LB3Q?e=g4r5Qd)<br>
Unzip as is
1.  Run the code <br> `python3 py_infer.py my_model/pipeline.config my_model/ exported_models/ my_model/label_map.pbtxt <image> outputs/predictions/ outputs/cropped_images/ outputs/text_files/bounding_box.txt ` <br>
    a. Input `y` if cropping is necessary, input the height and width (Standard DJI height x width is 3648x5472) <br>
2.  The outputted files will be saved as: <br>
    a.  Cropped Images at `outputs/cropped_images` \***Make this folder beforehand** <br>
    b.  Inferenced Images at `outputs/predictions/` \***Make this folder beforehand** <br>
    c.  Coordinates of coconut trees wrt the Ortho will be saved at `outputs/text_files/bounding_box.txt` <br>
3.  Run `python3 elevation.py <DEM path> outputs/text_files/bounding_box.txt outputs/text_files/coord_height.txt ` <br>
    a.  The coordinates of the highest elevation of each coordinate box from bounding_box.txt will be saved into coord_height.txt

### TODO

a.  The cropping and splitting takes a very long time due to OpenCV having to literally load the entire image onto memory
