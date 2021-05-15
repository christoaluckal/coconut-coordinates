# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Use this Jupyter Notebook as a guide to run your trained model in inference mode
# 
# created by Anton Morgunov
# 
# inspired by [tensorflow object detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#exporting-a-trained-model)
# %% [markdown]
# Your first step is going to specify which unit you are going to work with for inference. Select between GPU or CPU and follow the below instructions for implementation.


# %%
import os # importing OS in order to make GPU visible
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # do not change anything in here

# specify which device you want to work on.
# Use "-1" to work on a CPU. Default value "0" stands for the 1st GPU that will be used
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # TODO: specify your computational device


# %%
import tensorflow as tf # import tensorflow

# checking that GPU is found
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


# %%
# other import
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

# %% [markdown]
# Next you will import import scripts that were already provided by Tensorflow API. **Make sure that Tensorflow is your current working directory.**


# %%
import sys # importyng sys in order to access scripts located in a different folder

path2scripts = 'models/research/' # TODO: provide pass to the research folder
sys.path.insert(0, path2scripts) # making scripts in models/research available for import


# %%
# importing all scripts that will be needed to export your model and use it for inference
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


# %%
# Now you can import and build your trained model:


# %%
# NOTE: your current working directory should be Tensorflow.

# TODO: specify two pathes: to the pipeline.config file and to the folder with trained model.
path2config ='/home/christo/Desktop/sshrepo/coconut-detection/workspace_neptune/models/my_ssd/pipeline.config' #CHANGE
path2model = '/home/christo/Desktop/sshrepo/coconut-detection/workspace_neptune/exported_models/' #CHANGE


# %%
# do not change anything in this cell
configs = config_util.get_configs_from_pipeline_file(path2config) # importing config
model_config = configs['model'] # recreating model config
detection_model = model_builder.build(model_config=model_config, is_training=False) # importing model


# %%
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore('/home/christo/Desktop/sshrepo/coconut-detection/workspace_neptune/exported_models/checkpoint/ckpt-0').expect_partial() #CHANGE

# %% [markdown]
# Next, path to label map should be provided. Category index will be created based on labal map file


# %%
path2label_map = '/home/christo/Desktop/sshrepo/coconut-detection/workspace_neptune/data/label_map.pbtxt' #CHANGE
category_index = label_map_util.create_category_index_from_labelmap(path2label_map,use_display_name=True)

# %% [markdown]
# Now, a few supporting functions will be defined


# %%
def detect_fn(image):
    """
    Detect objects in image.
    
    Args:
      image: (tf.tensor): 4D input image
      
    Returs:
      detections (dict): predictions that model made
    """

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


# %%
def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      numpy array with shape (img_height, img_width, 3)
    """
    
    return np.array(Image.open(path))

# %% [markdown]
# **Next function is the one that you can use to run inference and plot results an an input image:**
# %% [markdown]
# Next, we will define a few other supporting functions:


# %%
def nms(rects, thd=0.5):
    """
    Filter rectangles
    rects is array of oblects ([x1,y1,x2,y2], confidence, class)
    thd - intersection threshold (intersection divides min square of rectange)
    """
    out = []

    remove = [False] * len(rects)

    for i in range(0, len(rects) - 1):
        if remove[i]:
            continue
        inter = [0.0] * len(rects)
        for j in range(i, len(rects)):
            if remove[j]:
                continue
            inter[j] = intersection(rects[i][0], rects[j][0]) / min(square(rects[i][0]), square(rects[j][0]))

        max_prob = 0.0
        max_idx = 0
        for k in range(i, len(rects)):
            if inter[k] >= thd:
                if rects[k][1] > max_prob:
                    max_prob = rects[k][1]
                    max_idx = k

        for k in range(i, len(rects)):
            if (inter[k] >= thd) & (k != max_idx):
                remove[k] = True

    for k in range(0, len(rects)):
        if not remove[k]:
            out.append(rects[k])

    boxes = [box[0] for box in out]
    scores = [score[1] for score in out]
    classes = [cls[2] for cls in out]
    return boxes, scores, classes


def intersection(rect1, rect2):
    """
    Calculates square of intersection of two rectangles
    rect: list with coords of top-right and left-boom corners [x1,y1,x2,y2]
    return: square of intersection
    """
    x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]));
    y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
    overlapArea = x_overlap * y_overlap;
    return overlapArea


def square(rect):
    """
    Calculates square of rectangle
    """
    return abs(rect[2] - rect[0]) * abs(rect[3] - rect[1])

# %% [markdown]
# **Next function is the one that you can use to run inference and save results into a file:**


# %%
def inference_as_raw_output(path2images,box_th = 0.25,nms_th = 0.5,to_file = False,data = None,path2dir = False):
    """
    Function that performs inference and return filtered predictions
    
    Args:
      path2images: an array with pathes to images
      box_th: (float) value that defines threshold for model prediction. Consider 0.25 as a value.
      nms_th: (float) value that defines threshold for non-maximum suppression. Consider 0.5 as a value.
      to_file: (boolean). When passed as True => results are saved into a file. Writing format is
      path2image + (x1abs, y1abs, x2abs, y2abs, score, conf) for box in boxes
      data: (str) name of the dataset you passed in (e.g. test/validation)
      path2dir: (str). Should be passed if path2images has only basenames. If full pathes provided => set False.
      
    Returs:
      detections (dict): filtered predictions that model made
    """
    
    print (f'Current data set is {data}')
    print (f'Ready to start inference on {len(path2images)} images!')
    
    for image_path in tqdm(path2images):
        if path2dir: # if a path to a directory where images are stored was passed in
            image_path = os.path.join(path2dir, image_path.strip())
            
        image_np = load_image_into_numpy_array(image_path)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        
        # checking how many detections we got
        num_detections = int(detections.pop('num_detections'))
        
        # filtering out detection in order to get only the one that are indeed detections
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        # defining what we need from the resulting detection dict that we got from model output
        key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']
        
        # filtering out detection dict in order to get only boxes, classes and scores
        detections = {key: value for key, value in detections.items() if key in key_of_interest}
        
        if box_th: # filtering detection if a confidence threshold for boxes was given as a parameter
            for key in key_of_interest:
                scores = detections['detection_scores']
                current_array = detections[key]
                filtered_current_array = current_array[scores > box_th]
                detections[key] = filtered_current_array
        
        if nms_th: # filtering rectangles if nms threshold was passed in as a parameter
            # creating a zip object that will contain model output info as
            output_info = list(zip(detections['detection_boxes'],
                                   detections['detection_scores'],
                                   detections['detection_classes']
                                  )
                              )
            boxes, scores, classes = nms(output_info)
            
            detections['detection_boxes'] = boxes # format: [y1, x1, y2, x2]
            detections['detection_scores'] = scores
            detections['detection_classes'] = classes
            
        if to_file and data: # if saving to txt file was requested

            image_h, image_w, _ = image_np.shape
            file_name = f'pred_result_{data}.txt'
            
            line2write = list()
            line2write.append(os.path.basename(image_path))
            
            with open(file_name, 'a+') as text_file:
                # iterating over boxes
                for b, s, c in zip(boxes, scores, classes):
                    
                    y1abs, x1abs = b[0] * image_h, b[1] * image_w
                    y2abs, x2abs = b[2] * image_h, b[3] * image_w
                    
                    list2append = [x1abs, y1abs, x2abs, y2abs, s, c]
                    line2append = ','.join([str(item) for item in list2append])
                    
                    line2write.append(line2append)
                
                line2write = ' '.join(line2write)
                text_file.write(line2write + os.linesep)
        
        return detections


# %%
def inference_with_plot(path2images,outputs,count=0, box_th=0.33,iou_threshold=0.1):
    """
    Function that performs inference and plots resulting b-boxes
    
    Args:
      path2images: an array with pathes to images
      box_th: (float) value that defines threshold for model prediction.
      
    Returns:
      None
    """
    print(path2images)
    for image_path in path2images:

        print('Running inference for {}... '.format(image_path), end='')

        image_np = load_image_into_numpy_array(image_path)
        
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        selected_indices = tf.image.non_max_suppression(boxes=detections['detection_boxes'], max_output_size=100, iou_threshold=0.01,scores=detections['detection_scores'])
        boxes = tf.gather(detections['detection_boxes'],selected_indices).numpy()
        scores = tf.gather(detections['detection_scores'],selected_indices).numpy()
        fin_boxes = []
        for i in range(0,scores.shape[0]):
          if scores[i] > box_th:
            fin_boxes.append(boxes[i].tolist())
        if(len(fin_boxes)!=0):
          label_id_offset = 1
          image_np_with_detections = image_np.copy()
          viz_utils.visualize_boxes_and_labels_on_image_array(
                  image_np_with_detections,
                  boxes,
                  detections['detection_classes']+label_id_offset,
                  scores,
                  category_index,
                  use_normalized_coordinates=True,
                  max_boxes_to_draw=200,
                  min_score_thresh=box_th,
                  agnostic_mode=False,
                  line_thickness=5)

          # plt.figure(figsize=(15,10))
          # plt.imshow(image_np_with_detections)
          plt.imsave(outputs+'/op{}.png'.format(count),image_np_with_detections) #CHANGE
          print('Done')
          return fin_boxes
        else:
          break


# %%
import time
import image
import sys
args = sys.argv[1:]
Image.MAX_IMAGE_PIXELS = None
img_name = args[0]
op_path = args[1]
sp_path = args[2]
count = 0
t_img = [img_name] #CHANGE
choice = input(print("Do you want to break up the image? y/n"))
if choice == 'n':
    start = time.time()
    box_list = inference_with_plot(t_img,op_path,box_th=0.20)
    end = time.time()
    print(box_list)
    print(end-start)
else:
    start = time.time()
    img_name_list = image.breakImage(t_img[0],sp_path)
    # print(img_list)
    for x in img_name_list:
        box_list = inference_with_plot([x],op_path,count,box_th=0.20)
        print(box_list)
        count+=1
    end = time.time()
    print(end-start)
# inference_as_raw_output(new_img,to_file=True,data="test")


# %%




