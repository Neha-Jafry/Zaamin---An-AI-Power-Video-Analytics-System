#!/usr/bin/env python3
"""
 Copyright (C) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import colorsys
from concurrent.futures import thread
import logging as log
from os import stat
import random
from re import L
import sys
from tkinter import W
import matplotlib.pyplot as plt
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter
from flask import Flask, render_template, Response
from flask import request

from influxdb import InfluxDBClient
import threading
import datetime
import time
from matplotlib import image
import numpy as np
import cv2
import copy
# from make_video import make_video
# from progress.bar import Bar

import cv2
from itsdangerous import json
import numpy as np
# deep sort imports
# from deep_sort import preprocessing, nn_matching
# from deep_sort import tracker
# from deep_sort.detection import Detection
# from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import json as json_p
# YOLO detectors
from detectors import *

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/openvino/model_zoo'))

from model_api.models import DetectionModel, DetectionWithLandmarks, RESIZE_TYPES, OutputTransform
from model_api.performance_metrics import PerformanceMetrics
from model_api.pipelines import get_user_config, AsyncPipeline
from model_api.adapters import create_core, OpenvinoAdapter, RemoteAdapter

import monitors
from images_capture import open_images_capture
from helpers import resolution, log_latency_per_stage

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)
THRESHOLD = 0.8
ppl_in_frame = {"current":0,"max":0}
db_client = None
zones = []
PORT = 8086
IPADDRESS = "localhost"
DATABASE_NAME = "CarDB"
dbLock = threading.Lock()
imgbase = []
def updateDB(data):
    global db_client
    global dbLock
    # print(data)
    with dbLock:
        db_client.write_points(data)
        print("written")
def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                      required=True)
    available_model_wrappers = [name.lower() for name in DetectionModel.available_wrappers()]
    args.add_argument('-at', '--architecture_type', help='Required. Specify model\' architecture type.',
                      type=str, required=True, choices=available_model_wrappers)
    args.add_argument('--adapter', help='Optional. Specify the model adapter. Default is openvino.',
                      default='openvino', type=str, choices=('openvino', 'remote'))
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is '
                           'acceptable. The demo will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('--labels', help='Optional. Labels mapping file.', default=None, type=str)
    common_model_args.add_argument('-t', '--prob_threshold', default=0.5, type=float,
                                   help='Optional. Probability threshold for detections filtering.')
    common_model_args.add_argument('--resize_type', default=None, choices=RESIZE_TYPES.keys(),
                                   help='Optional. A resize type for model preprocess. By defauld used model predefined type.')
    common_model_args.add_argument('--input_size', default=(600, 600), type=int, nargs=2,
                                   help='Optional. The first image size used for CTPN model reshaping. '
                                        'Default: 600 600. Note that submitted images should have the same resolution, '
                                        'otherwise predictions might be incorrect.')
    common_model_args.add_argument('--anchors', default=None, type=float, nargs='+',
                                   help='Optional. A space separated list of anchors. '
                                        'By default used default anchors for model. Only for YOLOV4 architecture type.')
    common_model_args.add_argument('--masks', default=None, type=int, nargs='+',
                                   help='Optional. A space separated list of mask for anchors. '
                                        'By default used default masks for model. Only for YOLOV4 architecture type.')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                            default=0, type=int)
    infer_args.add_argument('-nstreams', '--num_streams',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                            default='', type=str)
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    io_args.add_argument('-o', '--output', required=False,
                         help='Optional. Name of the output file(s) to save.')
    io_args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    io_args.add_argument('--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    input_transform_args = parser.add_argument_group('Input transform options')
    input_transform_args.add_argument('--reverse_input_channels', default=False, action='store_true',
                                      help='Optional. Switch the input channels order from '
                                           'BGR to RGB.')
    input_transform_args.add_argument('--mean_values', default=None, type=float, nargs=3,
                                      help='Optional. Normalize input by subtracting the mean '
                                           'values per channel. Example: 255.0 255.0 255.0')
    input_transform_args.add_argument('--scale_values', default=None, type=float, nargs=3,
                                      help='Optional. Divide input by scale values per channel. '
                                           'Division is applied after mean values subtraction. '
                                           'Example: 255.0 255.0 255.0')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser


class ColorPalette:
    def __init__(self, n, rng=None):
        assert n > 0

        if rng is None:
            rng = random.Random(0xACE) # nosec - disable B311:random check

        candidates_num = 100
        hsv_colors = [(1.0, 1.0, 1.0)]
        for _ in range(1, n):
            colors_candidates = [(rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0))
                                 for _ in range(candidates_num)]
            min_distances = [self.min_distance(hsv_colors, c) for c in colors_candidates]
            arg_max = np.argmax(min_distances)
            hsv_colors.append(colors_candidates[arg_max])

        self.palette = [self.hsv2rgb(*hsv) for hsv in hsv_colors]

    @staticmethod
    def dist(c1, c2):
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

    @classmethod
    def min_distance(cls, colors_set, color_candidate):
        distances = [cls.dist(o, color_candidate) for o in colors_set]
        return np.min(distances)

    @staticmethod
    def hsv2rgb(h, s, v):
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

    def __getitem__(self, n):
        return self.palette[n % len(self.palette)]

    def __len__(self):
        return len(self.palette)


FRAME_COUNT = 0
FRAME_HEATMAP_DATA = [] 

def push_heatmap_info(num_rows,num_cols,bboxes):
    # print(num_rows,num_cols)
    # 72,127
    # 127.72
    new_heatmap = {(i,j):0 for i in range(num_cols) for j in range(num_rows)}
    for xmin,ymin,xmax,ymax in bboxes:
        x_start = xmin // 10
        y_start = ymin // 10
        x_end = min(xmax // 10,num_cols-1)
        y_end = min(ymax // 10,num_rows -1)
        for x in range(x_start,x_end+1):
            for y in range(y_start,y_end+1):
                if x == x_start or x == x_end or y == y_start or y == y_end:
                    x_left = xmin - (x*10)
                    y_left = ymin - (y*10)
                    x_left = min(-x_left,10)
                    y_left = min(-y_left,10)
                    new_heatmap[(x,y)] += (x_left*y_left) *0.01
                    # 95,62
                    # y = 25 x = 83
                else:
                    new_heatmap[(x,y)] += 1
    detection_body = []
    for row,col in new_heatmap.keys():
        detection_body.append(
        {
        "measurement": "people_heatmap",
        "tags": {"detect" : num_cols*row+col},
        "fields": {
            "x": row,
            "time": 1644345608119000,
            "y": col,
            "value": float(new_heatmap[(row,col)]),
            }
        })
    y = threading.Thread(target=updateDB, args=(detection_body,))
    y.start()
    # print("DB TIME")
    return


def draw_detections(frame, detections, palette, labels, output_transform):
    global FRAME_COUNT,FRAME_HEATMAP_DATA
    
    frame = output_transform.resize(frame)
    # if FRAME_COUNT == 20:
    #     h,w,c = frame.shape
    #     num_rows = h // 10 + (h % 10 > 0)
    #     num_cols = w //10 + (w % 10 >0)
    #     push_heatmap_info(num_rows,num_cols,FRAME_HEATMAP_DATA.copy())
    #     FRAME_HEATMAP_DATA = []
    #     FRAME_COUNT = 0
    # FRAME_COUNT+=1
        
    count = 0
    bboxes = []
    scores = []
    classes = []
    names = []
    car_info = {
        "num_cars": 0,
        "num_bike": 0,
        "num_bus": 0,
    }
    global zones
    for bbox in zones:
        cv2.rectangle(frame, (bbox["x"], bbox["y"]), (bbox["x"]+bbox["w"],bbox["y"]+ bbox["h"]), (53,0,255),3)
    for detection in detections:
        if detection.score > THRESHOLD and int(detection.id) in [2,3,5]:
            class_id = int(detection.id)
            if class_id ==2 :
                car_info['num_cars'] +=1
            elif class_id==3:
                car_info['num_bike']+=1
            elif class_id==5:
                car_info['num_bus']+=1
            color = palette[class_id]
            det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
            names.append(det_label)
            xmin, ymin, xmax, ymax = detection.get_coords()
            xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
            count +=1
            bboxes.append(np.array([xmin, ymin, xmax-xmin, ymax-ymin]))
            scores.append(detection.score)
            cv2.rectangle(frame, (xmin,ymax), (xmax,ymax), palette[class_id],2)
            # classes.append(class_id)

            # FRAME_HEATMAP_DATA.append((xmin,ymin,xmax,ymax))
            # cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
            #             (xmin, ymin s- 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            if isinstance(detection, DetectionWithLandmarks):
                for landmark in detection.landmarks:
                    landmark = output_transform.scale(landmark)
                    cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 255), 2)
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), palette[0], 2)
            flag = False
            perspective_mode =True
            for zone in zones:
                z_x1,z_y1,z_x2,z_y2=zone["x"], zone["y"], zone["x"]+zone["w"], zone["y"]+zone["h"]
                a_x1 = max(xmin,z_x1)
                a_y1 = max(ymin,z_y1)
                a_x2 = min(xmax,z_x2)
                a_y2 = min(ymax,z_y2)
                # print(a_x1,a_y1,a_x2,a_y2)
                # print(b_x1,b_y1,b_x2,b_y2)
                # print(z_x1,z_y1,z_x2,z_y2)
                if a_x1 < a_x2 and a_y1 < a_y2:
                    if not perspective_mode or (z_x1<=xmin and xmin<=z_x2 and z_y1<=ymin and ymin <= z_y2):
                        flag = True
                        break
            print(f"Object of Class {str(detection.id)} in Zone: {flag}") 
            json_body = [{
                    "measurement": "bbox_data",
                    "fields": {
                        "x":xmin,
                        "y": ymin,
                        "w": xmax-xmin,
                        "h": ymax-ymin,
                        "class_name": str(detection.id),
                        "in_zone": int(flag)
                    }
                }]
            # updateDB(json_body)
            x = threading.Thread(target=updateDB, args=(json_body,))
            # db_client.write_points(json_body)
            x.start()
        # draw bbox on screen
        # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])),  palette[0], -1)
        # cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

    # if enable info flag then print details about each track

    # calculate frames per second of running detections
    json_body = [{
        "measurement": "car_count",
        "fields": {
            "cars":int(car_info['num_cars']),
            "bikes": int(car_info['num_bike']),
            "busses": int(car_info['num_bus']),
        }
    }]
    # updateDB(json_body)
    x = threading.Thread(target=updateDB, args=(json_body,))
    # db_client.write_points(json_body)
    x.start()

    if count:
        object_count = len(names)
        cv2.putText(frame, "Objects being tracked: {}".format(object_count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        print("Objects being tracked: {}".format(object_count))








    # cv2.putText(frame,f'Number of people: {count}',(frame.shape[1]//2,30),cv2.FONT_HERSHEY_DUPLEX,1.3,(255,255,0),1)
    # json_body = [{
    #             "measurement": "people_count",
    #             "fields": {
    #                 "count":len(names)
    #             }
    #         }]
    # # updateDB(json_body)
    # x = threading.Thread(target=updateDB, args=(json_body,))
    # # db_client.write_points(json_body)
    # x.start()
        
    return frame


def print_raw_results(detections, labels, frame_id):
    log.debug(' ------------------- Frame # {} ------------------ '.format(frame_id))
    log.debug(' Class ID | Confidence | XMIN | YMIN | XMAX | YMAX ')
    for detection in detections:
        xmin, ymin, xmax, ymax = detection.get_coords()
        class_id = int(detection.id)
        det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
        log.debug('{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} '
                  .format(det_label, detection.score, xmin, ymin, xmax, ymax))

def vid_stream_grafana():
    global imgbase
    while True:
        if len(imgbase) >0:
            yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + imgbase[0] + b'\r\n\r\n')
        if len(imgbase) > 1:
            imgbase = imgbase[1:]



def main():
    global imgbase
    # calculate cosine distance metric

    args = build_argparser().parse_args()
    if args.architecture_type != 'yolov4' and args.anchors:
        log.warning('The "--anchors" options works only for "-at==yolov4". Option will be omitted')
    if args.architecture_type != 'yolov4' and args.masks:
        log.warning('The "--masks" options works only for "-at==yolov4". Option will be omitted')

    cap = open_images_capture(args.input, args.loop)

    if args.adapter == 'openvino':
        plugin_config = get_user_config(args.device, args.num_streams, args.num_threads)
        model_adapter = OpenvinoAdapter(create_core(), args.model, device=args.device, plugin_config=plugin_config,
                                        max_num_requests=args.num_infer_requests)
    elif args.adapter == 'remote':
        log.info('Reading model {}'.format(args.model))
        serving_config = {"address": "localhost", "port": 9000}
        model_adapter = RemoteAdapter(args.model, serving_config)

    configuration = {
        'resize_type': args.resize_type,
        'mean_values': args.mean_values,
        'scale_values': args.scale_values,
        'reverse_input_channels': args.reverse_input_channels,
        'path_to_labels': args.labels,
        'confidence_threshold': args.prob_threshold,
        'input_size': args.input_size, # The CTPN specific
    }
    model = DetectionModel.create_model(args.architecture_type, model_adapter, configuration)
    model.log_layers_info()
    detector_pipeline = AsyncPipeline(model)

    next_frame_id = 0
    next_frame_id_to_show = 0

    palette = ColorPalette(len(model.labels) if model.labels else 100)
    metrics = PerformanceMetrics()

    render_metrics = PerformanceMetrics()
    presenter = None
    output_transform = None
    video_writer = cv2.VideoWriter()

    while True:
        if detector_pipeline.callback_exceptions:
            raise detector_pipeline.callback_exceptions[0]

        # Process all completed requests
        # if next_frame_id_to_show %3 ==0:
        results = detector_pipeline.get_result(next_frame_id_to_show)
        if results:
            objects, frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            if len(objects) and args.raw_output_message:
                print_raw_results(objects, model.labels, next_frame_id_to_show)

            presenter.drawGraphs(frame)
            rendering_start_time = perf_counter()
            frame = draw_detections(frame, objects, palette, model.labels, output_transform)
            render_metrics.update(rendering_start_time)
            
            # metrics.update(start_time, frame)

            if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
                video_writer.write(frame)
            next_frame_id_to_show += 1

            if not args.no_show:
                #cv2.imshow('Detection Results', frame)
                ret, img = cv2.imencode('.jpg',frame)
                img = img.tobytes()
                imgbase.append(img)
                # log_latency_per_stage(cap.reader_metrics.get_latency(),
                #           detector_pipeline.preprocess_metrics.get_latency(),
                #           detector_pipeline.inference_metrics.get_latency(),
                #           detector_pipeline.postprocess_metrics.get_latency(),
                #           render_metrics.get_latency())
                info_decode = cap.reader_metrics.get_latency()
                info_preprocess = detector_pipeline.preprocess_metrics.get_latency()+info_decode
                info_inference = detector_pipeline.inference_metrics.get_latency()+info_preprocess
                info_postprocess = detector_pipeline.postprocess_metrics.get_latency()+info_inference
                info_render = render_metrics.get_latency()+info_postprocess
                # info_fps = info_decode + info_preprocess+info_inference+info_postprocess+info_render
                info_fps = render_metrics.get_total()[1]
                print(info_fps)
                json_body = [{
                "measurement": "model_latency",
                "fields": {
                    "decode":info_decode,
                    "preprocess":info_preprocess,
                    "inference": info_inference,
                    "postprocess": info_postprocess,
                    "render": info_render,
                    "info_fps": info_fps
                    }
                }]
                send_to_db = threading.Thread(target=updateDB, args=(json_body,))
                send_to_db.start()

                #key = cv2.waitKey(1)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n')
            continue

        if detector_pipeline.is_ready():
            # Get new image/frame
            start_time = perf_counter()
            frame = cap.read()
            if frame is None:
                if next_frame_id == 0:
                    raise ValueError("Can't read an image from the input")
                break
            if next_frame_id == 0:
                output_transform = OutputTransform(frame.shape[:2], args.output_resolution)
                if args.output_resolution:
                    output_resolution = output_transform.new_resolution
                else:
                    output_resolution = (frame.shape[1], frame.shape[0])
                presenter = monitors.Presenter(args.utilization_monitors, 55,
                                               (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
                if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                                         cap.fps(), output_resolution):
                    raise RuntimeError("Can't open video writer")
            # Submit for inference
            detector_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1

        else:
            # Wait for empty request
            detector_pipeline.await_any()

    detector_pipeline.await_all()
    # Process completed requests
    for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
        results = detector_pipeline.get_result(next_frame_id_to_show)
        while results is None:
            results = detector_pipeline.get_result(next_frame_id_to_show)
        objects, frame_meta = results
        frame = frame_meta['frame']
        start_time = frame_meta['start_time']

        if len(objects) and args.raw_output_message:
            print_raw_results(objects, model.labels, next_frame_id_to_show)

        presenter.drawGraphs(frame)
        rendering_start_time = perf_counter()
        frame = draw_detections(frame, objects, palette, model.labels, output_transform)
        render_metrics.update(rendering_start_time)
        metrics.update(start_time, frame)

        if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
            video_writer.write(frame)

        if not args.no_show:
            ret, img = cv2.imencode('.jpg',frame)
            img = img.tobytes()

            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n')

            #cv2.imshow('Detection Results', frame)
            #key = cv2.waitKey(1)

            ESC_KEY = 27
            # Quit.
            # if key in {ord('q'), ord('Q'), ESC_KEY}:
            #     break
            presenter.handleKey(key)

    metrics.log_total()
    # print(cap.reader_metrics.get_latency())
    log_latency_per_stage(cap.reader_metrics.get_latency(),
                          detector_pipeline.preprocess_metrics.get_latency(),
                          detector_pipeline.inference_metrics.get_latency(),
                          detector_pipeline.postprocess_metrics.get_latency(),
                          render_metrics.get_latency())
    for rep in presenter.reportMeans():
        log.info(rep)











capture = None
capture_frame_len = 0
capture_frame_num = 0
background_subtractor = None
accum_image = None
first_iteration_indicator =1
def heat_map():
    global capture,capture_frame_len,first_iteration_indicator,background_subtractor,accum_image,capture_frame_num
    if capture == None:
        args = build_argparser().parse_args()
        capture = cv2.VideoCapture(args.input)
        first_iteration_indicator = 1
        capture_frame_num = 0
        background_subtractor = cv2.createBackgroundSubtractorMOG2()
        capture_frame_len = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while capture_frame_num < capture_frame_len:
        ret, frame = capture.read()
        capture_frame_num+=1
        if first_iteration_indicator == 1:
            height, width = frame.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0
        else:

            filter = background_subtractor.apply(frame)  # remove the background
            # cv2.imwrite('./frame.jpg', frame)
            # cv2.imwrite('./diff-bkgnd-frame.jpg', filter)

            threshold = 0.5
            maxValue = 1
            ret, th1 = cv2.threshold(filter, threshold, maxValue, cv2.THRESH_BINARY)

            # add to the accumulated image
            accum_image = cv2.add(accum_image, th1)
        
            color_image_video = cv2.applyColorMap(accum_image, cv2.COLORMAP_MAGMA)

            video_frame = cv2.addWeighted(frame, 0.7, color_image_video, 0.3, 0)

            # name = "./frames/frame%d.jpg" % i
            # cv2.imwrite(name, video_frame)
            ret, img = cv2.imencode('.jpg',video_frame)
            img = img.tobytes()
            yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n')



# Create object for Flask class
app = Flask(__name__, template_folder="../templates/",static_folder="../static/")
#app.logger.disabled = True
log_ = log.getLogger('werkzeug')
#log_.disabled = True




@app.route('/')
def index():
    """
    Trigger the index() function on opening "0.0.0.0:5000/" URL
    :return: html file
    """
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """
    Trigger the video_feed() function on opening "0.0.0.0:5000/video_feed" URL
    :return:
    """
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_video_feed')
def test_video_feed():
    """
    Trigger the video_feed() function on opening "0.0.0.0:5000/video_feed" URL
    :return:
    """
    return Response(vid_stream_grafana(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/zones')
def bbox_module():
    """
    Trigger the video_feed() function on opening "0.0.0.0:5000/video_feed" URL
    :return:
    """
    return render_template("main.html")

@app.route('/heat_map_video_feed')
def heat_map_video_feed():
    """
    Trigger the video_feed() function on opening "0.0.0.0:5000/video_feed" URL
    :return:
    """
    return Response(heat_map(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/submitRectangles",methods=["POST"])
def submit_rectangles():
    output = request.get_json()
    
    global zones
    zones = output
    with open("zones.json","w") as f:
        f.write(json_p.dumps(output))
    print(output) # This is the output that was stored in the JSON within the browser
    print(type(output[0]))
    return "done"
def create_database():
    """
    Connect to InfluxDB and create the database

    :return: None
    """
    global db_client

    proxy = {"http": "http://{}:{}".format(IPADDRESS, PORT)}
    db_client = InfluxDBClient(host=IPADDRESS, port=PORT, proxies=proxy, database=DATABASE_NAME)
    db_client.create_database(DATABASE_NAME)

if __name__ == '__main__':
    import os
    if os.path.exists("zones.json"):
        with open("zones.json","r") as f:
            zones = json_p.load(f)
    create_database()
    argsf = build_argparser().parse_args()
    vidcap = cv2.VideoCapture(argsf.input)
    success, img= vidcap.read()
    if success:
        cv2.imwrite("../static/img_zone.png", img)  # save frame as JPEG file
    print("AAAAAAAAAAAAAAAA")

    app.run(host='0.0.0.0')


if __name__ == '__main__':
    sys.exit(main() or 0)

