import os
import time
import tensorflow as tf
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import cv2
from path import Path
import b64

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

PATH_TO_LABELS='object_detection\\data\\mscoco_label_map.pbtxt'
PATH_TO_SAVED_MODEL='object_detection\\models\\ssd_mobilenet_v2_coco_2018_03_29\\saved_model'

detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
detect_fn = detect_fn.signatures['serving_default']

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

image_path='input_b64.jpg'
image_np=load_image_into_numpy_array(image_path)
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}

detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

def search_item(item_name):
    counts=0
    item_name=item_name.lower()
    for(label,score) in zip(detections['detection_classes'],detections['detection_scores']):
        if category_index[label]['name']==item_name:
            counts+=1
    if counts==0:
        print('Not found!')
    print(counts,item_name)

search_item('table')