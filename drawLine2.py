import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


pt1 = (0,0)
pt2 = (0,0)
clicked1 = False
clicked2 = False
points = []

d = set()
d1 = {'car':set(), 'truck':set(), 'bus':set(), 'motorbike':set(), 'bicycle':set()}



def find_if_ojbect_crossed_line(line_coordinates,object_coordinates,track,class_name, count):
    
    global d,d1
    x1,y1,x2,y2 = line_coordinates
    x,y,w,h = object_coordinates
    
    if x>x1 and x<x2:
        if y1>y2:
            if y<y1 and y+h>y1:
                if track.track_id not in d:
                    d[class_name] = count
                    d1[class_name][track.track_id] += 1
                return True
        else:
            if y<y2 and y+h>y2:
                if track.track_id not in d:
                    d[class_name] = count
                    d1[class_name][track.track_id] += 1
                return True
    return False

def find_if_ojbect_crossed_line(line_coordinates,object_coordinates,track):
    
    global d,d1
    x1,y1,x2,y2 = line_coordinates
    x,y,w,h = object_coordinates
    
    if x>x1 and x<x2:
        if y1>y2:
            if y<y1 and y+h>y1:
                if track.track_id not in d:
                    d.add(track.track_id)
                    d1[track.get_class()].add(track.track_id)
                return True
        else:
            if y<y2 and y+h>y2:
                if track.track_id not in d:
                    d.add(track.track_id)
                    d1[track.get_class()].add(track.track_id)
                return True
    return False


def draw_line(event, x, y, flags, param):

    global pt1, pt2, clicked1, clicked2,points

    
    if event == cv2.EVENT_LBUTTONDOWN:
        
        if clicked1 and clicked2:
            clicked1 = False
            clicked2 = False
            pt1 = (0,0)
            pt2 = (0,0)
        
        if not clicked1:

            pt1 = (x,y)
            points.append(pt1)
            clicked1 = True
        
        elif not clicked2:
            pt2 = (x,y)
            points.append(pt2)
            clicked2 = True

cap = cv2.VideoCapture('video3.mp4')
cv2.namedWindow(winname='myName')
cv2.setMouseCallback('myName', draw_line)


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 1.0


model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

tracker = Tracker(metric)


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

input_size = 416


saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']


first_frame = True
count = 0
frame_num = 0


while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    while first_frame :

        cv2.imshow('myName', frame)
        if clicked1 and clicked2:
            cv2.line(frame, pt1, pt2, (255,0,0), 2)

            count+=1
        
        if cv2.waitKey(1) &0xFF == ord('c') :
            first_frame = False
            break
    
    
    
    cv2.line(frame, points[0],points[1], (255,0,0), 2)
    
    image = Image.fromarray(frame)
    frame_num +=1
    # print('Frame 
    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    start_time = time.time()
    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.80
    )
    num_objects = valid_detections.numpy()[0]
    bboxes = boxes.numpy()[0]
    bboxes = bboxes[0:int(num_objects)]
    scores = scores.numpy()[0]
    scores = scores[0:int(num_objects)]
    classes = classes.numpy()[0]
    classes = classes[0:int(num_objects)]

    original_h, original_w, _ = frame.shape
    bboxes = utils.format_boxes(bboxes, original_h, original_w)

    
    pred_bbox = [bboxes, scores, classes, num_objects]

    
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    
    allowed_classes = list(class_names.values())
    
    
    
    allowed_classes = ['car', 'truck', 'bus', 'motorbike', 'bicycle']


    

    names = []
    deleted_indx = []
    for i in range(num_objects):
        class_indx = int(classes[i])
        class_name = class_names[class_indx]
        if class_name not in allowed_classes:
            deleted_indx.append(i)
        else:
            names.append(class_name)
            
            

    names = np.array(names)
    
    

    bboxes = np.delete(bboxes, deleted_indx, axis=0)
    scores = np.delete(scores, deleted_indx, axis=0)

    
    features = encoder(frame, bboxes)
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

    
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]       

    
    
    tracker.predict()
    tracker.update(detections)

    
    obj_count = {}
    
        
    
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue 
        bbox = track.to_tlbr()
        class_name = track.get_class()
        
        # for every new object assigned to track update the dict

        # if track.track_id not in d1[class_name]:
        #     d1[class_name][track.track_id] = 0
        
        x1,y1 = int(bbox[0]), int(bbox[1])
        x2,y2 = int(bbox[2]), int(bbox[3])
        x,y = points[0]
        p,q = points[1]
        # d[track.track_id] = 0

        if (find_if_ojbect_crossed_line((x,y,p,q),(x1,y1,x2,y2), track)):
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(len(d1[class_name])),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if (find_if_ojbect_crossed_line((x,y,p,q),(x1,y1,x2,y2),track,class_name,d[class_name])) or  d1[class_name][track.track_id]:
        #     color = colors[int(track.track_id) % len(colors)]
        #     color = [i * 255 for i in color]
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        #     cv2.putText(frame, class_name + "-" + str(len(d1[class_name])),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            
    cv2.putText(frame, "Car: {}".format( str(len(d1['car']))), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0),2)
    cv2.putText(frame, "Truck: {}".format( str(len(d1['truck']))), (5, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0),2)
    cv2.putText(frame, "Bike : {}".format( str(len(d1['motorbike']))), (5, 165), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0),2)
    cv2.putText(frame, "Bus : {}".format( str(len(d1['bus']))), (5, 230), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0),2)
    cv2.putText(frame, "Bicycle : {}".format( str(len(d1['bicycle']))), (5, 295), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0),2)
    fps = 1.0 / (time.time() - start_time)
    print("FPS: %.2f" % fps)
    frame = np.asarray(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)
    
    cv2.imshow('myName', frame)
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(points)
        

