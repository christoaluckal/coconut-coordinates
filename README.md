# image-manipulation
Download the files at: <br> 
1.  Run the code <br> `python3 py_infer.py my_model/pipeline.config my_model/ exported_models/ my_model/label_map.pbtxt <image> outputs/predictions/ outputs/cropped_images/ outputs/text_files/bounding_box.txt ` <br>
2.  The outputted files will be saved as: <br>
    a.  Cropped Images at `outputs/cropped_images` <br>
    b.  Inferenced Images at `outputs/predictions/` <br>
    c.  Coordinate of <image> at `outputs/text_files/bounding_box.txt` <br>
3.  Run `python3 elevation.py DBCA_Export.tif outputs/text_files/bounding_box.txt outputs/text_files/coord_height.txt ` <br>
    a.  The coordinates of the highest elevation of each coordinate box from bounding_box.txt will be saved into coord_height.txt
