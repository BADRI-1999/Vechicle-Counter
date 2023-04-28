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
import pandas as pd


df = pd.DataFrame(columns=['frame_num','class_name','class_count','total_count','timestamp'])

pt1 = [0,0]
pt2 = [0,0]
clicked1 = False
clicked2 = False
points = []

d = set()
d1 = {'car':set(), 'truck':set(), 'bus':set(), 'motorbike':set(), 'bicycle':set()}
font_size = {'car':0, 'truck':0, 'bus':0, 'motorbike':0, 'bicycle':0,'Unknown':0}
count_unknown = 0


# get optimal font Scale for text on an Image
def get_font_size(text, width):
    font_size = 1
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
    text_width = text_size[0][0]
    while text_width > width:
        font_size -= 0.1
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
        text_width = text_size[0][0]
    return font_size

def create_line(pt1,pt2):
        
    x1,y1 = pt1
    x2,y2 = pt2
    
    if x1==x2:
        A = 1
        B = 0
        C = -x1
    else:
        A = (y2-y1)/(x2-x1)
        B = -1
        C = y1 - A*x1
    
    return A,B,C

def crosssed(line_coordinates,object_coordinates,track):
    global d,d1
    x1,y1,x2,y2 = line_coordinates
    x,y,w,h = object_coordinates

    A,B,C = create_line((x1,y1),(x2,y2))

    cross = A * x + B * y + C
    if cross >= 0:
        if track.track_id not in d:
            d.add(track.track_id)
            d1[track.get_class()].add(track.track_id)

        return True
    else:
        return False

def intersect(x1,y1, k , x , min_y, max_y):

    y = y1 + (x - x1) * k
    
    # y *= 2//
    if y > min_y and y < max_y:
        print(y,y1,x,x1, k, min_y, max_y)
        return True
    # return (y >= min_y and y <= max_y)

def collision_detection_rect(line, rect,track):
    x, y, w, h = rect
    x1, y1, x2, y2 = line
    if x1 > x and x1 < x + w and y1 > y and y1 < y + h:
        if track.track_id not in d:
            d.add(track.track_id)
            d1[track.get_class()].add(track.track_id)
        # push_into_dataframe(frame_num,track.get_class(),1,1,timestamp)
        return True
    if x2 > x and x2 < x + w and y2 > y and y2 < y + h:
        if track.track_id not in d:
            d.add(track.track_id)
            d1[track.get_class()].add(track.track_id)
        return True
    return False

# Collision for Retangle and line opencv
def collision_detection(line_coordinates,object_coordinates, track):
    x1,y1,x2,y2 = line_coordinates
    x,y,w,h = object_coordinates

    if x1 == x2:
        if x1 >= x and x1 <= w:
            print("Collision x 1")
            if track.track_id not in d:
                d.add(track.track_id)
                d1[track.get_class()].add(track.track_id)
            return True
        
    
    elif y1 == y2:
        if y1 >= y and y1 <= h:
            print("Collision y 1")
            if track.track_id not in d:
                d.add(track.track_id)
                d1[track.get_class()].add(track.track_id)
            return True
        # return y1 >= y and y1 <= y+h
    
    k = (y2-y1)//(x2-x1)

    # chcek if a point is in between the line


    if intersect(x1,y1, k , x,y,h -1) and ((x>x1 and x<x2 ) or (h < x1 and h > x2)) :
        print("Collision x 1 with slope")
        if track.track_id not in d:
            d.add(track.track_id)
            d1[track.get_class()].add(track.track_id)
        return True
    
    if intersect(x1,y1, k , w-1,y,h -1) and ((x>x1 and x<x2 ) or (h < x1 and h > x2)):
        print("Collision x 2 with slope")
        if track.track_id not in d:
            d.add(track.track_id)
            d1[track.get_class()].add(track.track_id)
        return True
    try:
        k_inv = (x2 - x1 ) // (y2 - y1)
    except:
        k_inv = 1

    if intersect(y1, x1, k_inv, y, x, w -1) and ((x>x1 and x<x2 ) or (h < x1 and h > x2)) :
        print("Collision y 1 with inv slope")
        if track.track_id not in d:
            d.add(track.track_id)
            d1[track.get_class()].add(track.track_id)
        return True

    if intersect(y1,x1,k_inv, h-1, x, w -1) and ((x>x1 and x<x2 ) or (h < x1 and h > x2)) :
        print("Collision y 2 with inv slope")
        if track.track_id not in d:
            d.add(track.track_id)
            d1[track.get_class()].add(track.track_id)
        return True
    
    return False

def find_if_object_crossed_line_opp_lane(line_coordinates,object_coordinates,track):

    global d,d1
    x1,y1,x2,y2 = line_coordinates
    x,y,w,h = object_coordinates
        
    # check if object crossed the line  with points x1,y1 and x2,y2 and object coordinates x,y,w,h
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


def find_if_object_crossed_line1(line_coordinates,object_coordinates,track,frame_num, timestamp):
    
    global d,d1
    # x1,y1,x2,y2 = line_coordinates
    # x,y,w,h = object_coordinates

    x1, y1, x2, y2 = line_coordinates
    x3, y3, x4, y4 = object_coordinates

    # Calculate the distance to intersection point
    den = ((y4-y3) * (x2-x1) - (x4-x3) * (y2-y1))
    if den == 0:
        return False

    ua = ((x4-x3) * (y1-y3) - (y4-y3) * (x1-x3)) / den
    ub = ((x2-x1) * (y1-y3) - (y2-y1) * (x1-x3)) / den

    if not (0 <= ua <= 1 or 0 <= ub <= 1):
        return False

    x = x1 + ua*(x2-x1)
    y = y1 + ua*(y2-y1)

    # Find the difference between the actual intersection point and the line segment
    dl = ((x-x1)**2 + (y-y1)**2)**0.5
    dr = ((x-x2)**2 + (y-y2)**2)**0.5

    if dl <=10 or dr <=10:
        if track.track_id not in d:
            d.add(track.track_id)
            d1[track.get_class()].add(track.track_id)
        push_into_dataframe(frame_num,track.get_class(),len(d1[track.get_class()]),len(d),timestamp)
        return True

    # return dl <= 10 or dr <= 10

def find_if_object_crossed_line(line_coordinates,object_coordinates,track,frame_num, timestamp):
    
    global d,d1
    x1,y1,x2,y2 = line_coordinates
    x,y,w,h = object_coordinates
    
    if x>x1 and x<x2:
        if y1>y2:
            if y<y1 and y+h>y1:
                if track.track_id not in d:
                    d.add(track.track_id)
                    d1[track.get_class()].add(track.track_id)
                
                push_into_dataframe(frame_num,track.get_class(),len(d1[track.get_class()]),len(d),timestamp)
                return True
        else:
            if y<y2 and y+h>y2:
                if track.track_id not in d:
                    d.add(track.track_id)
                    d1[track.get_class()].add(track.track_id)
                push_into_dataframe(frame_num,track.get_class(),len(d1[track.get_class()]),len(d),timestamp)
                return True
    return False

def push_into_dataframe(frame_num,class_name,class_count,total_count,timestamp):

    global df
    df = df.append({'frame_num':frame_num,'class_name':class_name,'class_count':class_count,'total_count':total_count,'timestamp':timestamp},ignore_index=True)

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

cap = cv2.VideoCapture('video6.mp4')
cv2.namedWindow(winname='myName')
cv2.setMouseCallback('myName', draw_line)


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

out = cv2.VideoWriter('results9/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

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

for i in d1.keys():
    font_size[i] = get_font_size(i, frame_width)

a,b = 5,50
off_set = 30

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    while first_frame :

        cv2.imshow('myName', frame)
        if clicked1 and clicked2:
            cv2.line(frame, pt1, pt2, (255,0,0), 2)

            # count+=1
        
        if cv2.waitKey(1) &0xFF == ord('c') :
            first_frame = False
            break
    
    
    
    cv2.line(frame, points[0],points[1], (255,0,0), 2)
    
    cv2.putText(frame, '{}'.format(points[0]), (points[0][0],points[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(frame, '{}'.format(points[1]), (points[1][0],points[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

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
            score_threshold=0.60
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

    # update count_unknown if the detected object is not from allowed classes
    for det in detections:
        if det.class_name not in allowed_classes:
            count_unknown += 1
        else:
            count_unknown = 0
    
    tracker.predict()
    tracker.update(detections)

    
    obj_count = {}
    
        
    
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1 or track.track_id in d:
            continue 
        bbox = track.to_tlbr()
        class_name = track.get_class()
        
        
        x1,y1 = int(bbox[0]), int(bbox[1])
        x2,y2 = int(bbox[2]), int(bbox[3])
        x,y = points[0]
        p,q = points[1]
        
        
        


        # , track,frame_num, time.time())
        if (collision_detection((x,y,p,q),(x1,y1,x2,y2),track)):
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.circle(frame, (int(bbox[0]), int(bbox[1])), 5, color, -1)
            cv2.circle(frame, (int(bbox[2]), int(int(bbox[3]))), 5, color, -1)
            cv2.putText(frame, ('{} , {}'.format(str(np.round(bbox[0])), str(np.round(bbox[1]-30)))), (int(bbox[0]), int(bbox[1]-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255),2)
            print(x1,y1,x2,y2)
            # cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            cv2.putText(frame, class_name + "-" + str(len(d1[class_name])),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
    

    for i,j in enumerate(d1.keys()):
        cv2.putText(frame, "{}: {}".format(j ,str(len(d1[j]))), (a, b+off_set*i), font_size[j], 2, (0, 255, 0),2)

    fps = 1.0 / (time.time() - start_time)
    # print("FPS: %.2f" % fps)
    # cv2.putText(frame, "FPS : {}".format(fps), (frame_width - 250, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0),2)
    frame = np.asarray(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)
    
    cv2.imshow('myName', frame)
    
    
    
    if cv2.waitKey(100) & 0xFF == ord('q'): 
        break

cap.release()
out.release()
cv2.destroyAllWindows()


df.to_csv('results9/data.csv')

        

