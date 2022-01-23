from __future__ import division, print_function, absolute_import

from timeit import time
import warnings

from deepsort import nn_matching
from deepsort.detection import Detection
from deepsort.tracker import Tracker
from training.data_preprocessing import generate_angles, batch
from yolov5 import YOLOv5

from config.config_reader import config_reader

from tools import processing
from tools import generate_detections as gdet
from tools.processing import extract_parts
from tools.coord_in_box import coordinates_in_box,bbox_to_fig_ratio

warnings.filterwarnings('ignore')

import cv2
import numpy as np
import shelve
from PIL import Image


# from training.data_preprocessing import batch, generate_angles
from keras.models import load_model

import tensorflow as tf
from helpmodels.openpose_model import pose_detection_model

import yolov5

print("Num GPUs Available: ", tf.test.gpu_device_name())

# from yolov5 import YOLOv5

# set model params
model_path = "yolov5/weights/yolov5s.pt" # it automatically downloads yolov5s model to given path
device = "cpu" # or "cpu"

# init yolov5 model
yolo = YOLOv5(model_path, device)

# Intializing YOLO model
# yolo = yolov5.load('yolov5s')

# Intializing OpenPose Model
model = pose_detection_model()

# Defining parameters for openpose model
param, model_params = config_reader()

# Definition of the parameters
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

# Deep SORT
model_filename = 'helpmodels/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

# Initializing the tracker with given metrics.
tracker = Tracker(metric)

model_ts = load_model('helpmodels/Time Series.h5')
writeVideo_flag = True
path = './fight_train.mp4'
video_capture = cv2.VideoCapture(path)  # changing paths

if writeVideo_flag:
    # Define the codec and create VideoWriter object
    # w = int(video_capture.get(3))
    # h = int(video_capture.get(4))
    w = 640
    h = 480
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(path + '_out.avi', fourcc, 6, (w, h))

frame_index = 0
person_TS = {}
count = 0
fps = 0.0
labels = {}
while True:
    ret, frame = video_capture.read()  # frame shape 640*480*3
    # print(ret)

    if ret != True:
        break
    if count % 50 != 0:
        print('SKIPPED {} FRAME'.format(count))
        count += 1
    else:
        t1 = time.time()
        frame = cv2.resize(frame, (640, 480))
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        findings = yolo.predict(image)
        # print("BOXES", boxs)
        findings = findings.pandas().xyxy[0]
        findings = findings[findings["class"] == 0]
        boxes = []
        for index, box in findings.iterrows():
            x=int(box["xmin"])
            y=int(box["ymin"])
            w=int(box["xmax"]-x)
            h=int(box["ymax"]-y)
            boxes.append([x,y,w,h])

        print(boxes)

        features = encoder(frame, boxes)
        # score to 1.0 here.
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = processing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        person_dict = extract_parts(frame, param, model, model_params)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

            flag = 0

            # Association of tracking with body keypoints
            for i in person_dict.keys():
                # If given body keypoints lie in the bounding box or not.
                if coordinates_in_box(bbox, list(person_dict[i].values())) and bbox_to_fig_ratio(bbox, list(
                        person_dict[i].values())):
                    if 'person_' + str(track.track_id) not in person_TS.keys():
                        person_TS['person_' + str(track.track_id)] = []

                    person_TS['person_' + str(track.track_id)].append(person_dict[i])
                    flag = 1
                    break
            if flag == 1:
                del (person_dict[i])

            if track.track_id not in labels.keys():
                labels[track.track_id] = 0
            # skeleton(frame, person_dict)
            print(person_TS)
            # print(person_dict)
            if not labels[track.track_id] and 'person_' + str(
                    track.track_id) in person_TS.keys():  # If not violent previously
                if len(person_TS['person_' + str(track.track_id)]) >= 6:
                    temp = []
                    for j in person_TS['person_' + str(track.track_id)][-6:]:
                        temp.append(generate_angles(j))
                        print(j)
                        print(temp[-1])
                    angles = batch(temp)
                    target = int(np.round(model_ts.predict(angles)))
                    labels[track.track_id] = target

            if labels[track.track_id]:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)

        frame_index += 1

        if writeVideo_flag:
            # Saving frame
            out.write(frame)

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %f" % (fps))
        print('PROCESSED {} FRAME'.format(count))
        count += 1
        t2 = time.time()
        print("PASSED TIME ", t2 - t1)

video_capture.release()
if writeVideo_flag:
    out.release()
