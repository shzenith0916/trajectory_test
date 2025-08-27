# -*- coding: utf-8 -*-
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

from utils.torch_utils import select_device, load_classifier, time_sync
from utils.plots import Annotator, colors, save_one_box
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, is_ascii, non_max_suppression, \
    apply_classifier, scale_segments, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.datasets import LoadStreams, LoadImages
from models.experimental import attempt_load
import math
from math import pi

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from pandas import Series
from pandas import DataFrame
import pandas as pd

# 2022-08-22(월), 윤*민, 선형보간 선언
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

# sys.stdout = open('output_detect.txt', 'w')

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path


def distance(x1, y1, x2, y2):
    result = ((((x2 - x1)**2) + ((y2-y1)**2))**0.5)
    return result


def crosspoint(x11, y11, x12, y12, x21, y21):

    m1 = (y12 - y11) / (x12 - x11)
    n = -(1.0/m1)
    b = y21 - (n * x21)
    if x12 > x11:
        x22 = x12
    else:
        x22 = x11
    y22 = (n*x22)+b
    m2 = n

    if x12 == x11 or x22 == x21:
        print('delta x=0')
        if x12 == x11:
            print('a')
            cx = x12
            # m2 = (y22 - y21) / (x22 - x21)
            cy = m2 * (cx - x21) + y21
            return x22, y22, cx, cy
        if x22 == x21:
            print('b')
            cx = x22
            # m1 = (y12 - y11) / (x12 - x11)
            cy = m1 * (cx - x11) + y11
            return x22, y22, cx, cy

    # m1 = (y12 - y11) / (x12 - x11)
    # m2 = (y22 - y21) / (x22 - x21)
    if m1 == m2:
        print('parallel')
        return None
    cx = (x11 * m1 - y11 - x21 * m2 + y21) / (m1 - m2)
    cy = m1 * (cx - x11) + y11

    return x22, y22, cx, cy


@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):

    count_1 = 0
    count_2 = 0
    count_3 = 0

    # 보정된 설골 위치 계산용
    hyoid_bone_x = []
    hyoid_bone_y = []

    # A와 C사이에 B점을 찾기 위한 값
    search_B_x = []
    search_B_y = []
    search_B = 0

    search_C = 0

    # C에서 끝까지 D점을 찾기 위한 값
    search_D_x = []
    search_D_y = []
    search_D = 0

    save_img = not nosave and not source.endswith(
        '.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    cnt = 0
    # Directories
    save_dir = increment_path(Path(project) / name,
                              exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

#     print(f' $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 1')

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

#     print(f' $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 2')

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix = False, Path(w).suffix.lower()
    pt, onnx, tflite, pb, saved_model = (
        suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
#     print(f' $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 3')
    if pt:
        #         print(f' $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ pt')
        model = attempt_load(weights, device=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(
            model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)[
                                   'model']).to(device).eval()
    elif onnx:
        #         print(f' $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ tonnx')
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        #         print(f' $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ TensorFlow models')
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(
                    lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(
                gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(
                model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # is TFLite quantized uint8 model
            int8 = input_details[0]['dtype'] == np.uint8
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # Dataloader
    if webcam:
        #         print(f' $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ webcam')
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        #         print(f' $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ no webcam')
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
#         print(f' $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ LoadImages')
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    print('vid_writer', vid_writer)

#     print(f' vid_writer ++++++++++++++++++++++++++++++++++++++++++++++> {vid_writer}')
    # Run inference
    if pt and device.type != 'cpu':
        #         print(f' $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ cpu')
        # run once
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))
#         print(f' $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ cpu after')
    t0 = time.time()
    cnt = 0

#     print(f' $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ for')
    print(f' \n')
    # count_1 = total frame 갯수
#     video 1/1 (1/361) /mnt/hdd6tb/yunjaemin/work/20220725_rsrehap/2021_hyoid_bone_tracking_ml/data/12183636최진순2LF.avi:
    # 2022-08-22(월), 윤*민,  vid_frames에 동영상 전체 프레임개수 리턴
    for path, img, im0s, vid_cap, vid_frames in dataset:

        print(
            f' \n vid_frames ++++++++++++++++++++++++++++++++++++++++++++++> {vid_frames}')
        print(
            f' \n path ++++++++++++++++++++++++++++++++++++++++++++++> {path}')
#         /mnt/hdd6tb/yunjaemin/work/20220725_rsrehap/2021_hyoid_bone_tracking_ml/data/12183636최진순2LF.avi
#         print(f' \n img ++++++++++++++++++++++++++++++++++++++++++++++> {img}')
#         [[[114 114 114 ... 114 114 114]
#           [114 114 114 ... 114 114 114]
#           [114 114 114 ... 114 114 114]
#           ...

#         print(f' \n im0s ++++++++++++++++++++++++++++++++++++++++++++++> {im0s}')
#         [[[  0   0   0]
#           [  0   0   0]
#           [  0   0   0]
#           ...

#         print(f' \n vid_cap ++++++++++++++++++++++++++++++++++++++++++++++> {vid_cap}')
#         < cv2.VideoCapture 0x7f7df55d3b10>

#         print(f' \n count_1 ++++++++++++++++++++++++++++++++++++++++++++++> {count_1}')

        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if pt:
            visualize = increment_path(
                save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {
                                session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) -
                            zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)

        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions

# det1============ tensor([[367.57428, 226.68011, 423.02277, 294.79071,   0.90382,   1.00000],
#         [377.51666, 394.08276, 426.43988, 443.24683,   0.89979,   1.00000],
#         [384.53210, 439.46432, 438.62903, 493.57785,   0.89802,   1.00000],
#         [373.41669, 343.86963, 424.20569, 395.00885,   0.89660,   1.00000],
#         [370.38721, 292.53577, 423.11176, 344.72473,   0.89432,   1.00000],
#         [262.23856, 338.13074, 290.30423, 376.48718,   0.88344,   0.00000]], device='cuda:0')

#             print(f'\n')
#             print("i============", i)

        for i, det in enumerate(pred):  # detections per image
            count_2 += 1
            print(
                f' count_2 ++++++++++++++++++++++++++++++++++++++++++++++> {count_2}')
            print("len(det)============", len(det))
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(
                ), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + \
                ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0  # for save_crop
            test_img = im0.copy()
            annotator = Annotator(
                im0, line_width=line_thickness, pil=not ascii)

            # 2022-07-28 윤*민, 위치변경(하 -> 상)
            # Write results
            x_neck_bone = []
            y_neck_bone = []
#             zeros_im0 = []

            # if len(det):
            if len(det) == 0:
                continue
            else:
                # Rescale boxes from img_size to im0 size
                #                 print("im0.shape============", im0.shape)
                print(
                    f"img.shape {img.shape} det.shape {det.shape}, {im0.shape}")
                print(det)
                det[:, :4] = scale_segments(
                    img.shape[2:], det[:, :4], im0.shape).round()
                # print(det[:, :4])
                # sys.exit()
#                 print("det2============", det)
# im0.shape============ (1024, 1280, 3)
# det2============ tensor([[639.00000, 378.00000, 736.00000, 496.00000,   0.90382,   1.00000],
#         [657.00000, 669.00000, 742.00000, 754.00000,   0.89979,   1.00000],
#         [669.00000, 748.00000, 763.00000, 842.00000,   0.89802,   1.00000],
#         [649.00000, 581.00000, 738.00000, 670.00000,   0.89660,   1.00000],
#         [644.00000, 492.00000, 736.00000, 583.00000,   0.89432,   1.00000],
#         [456.00000, 571.00000, 505.00000, 638.00000,   0.88344,   0.00000]], device='cuda:0')
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

#                 # Write results
#                 x_neck_bone=[]
#                 y_neck_bone=[]

                if cnt == 0:
                    # 이미지와 동일한 크기의 하얀색 배경 생성
                    zeros_im0 = np.zeros_like(im0) + 255

                cnt += 1
                obj_true = 0
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
                        print("xywh1============", xywh)
                        # label format
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
#                         print("xywh2============", xywh)
#                         xywh2============ [0.3753906190395355, 0.59033203125, 0.03828125074505806, 0.0654296875]

                        w, h = im0.shape[1], im0.shape[0]
#                         print("w, h ============", w, h)
#                         w, h ============ 1280 1024

                        x_center = int(float(xywh[0])*w)
                        y_center = int(float(xywh[1])*h)
                        box_w = int(float(xywh[2])*w)
                        box_h = int(float(xywh[3])*h)
                        c = int(cls)  # integer class
                        if cls == 2:  # food_locate
                            x1 = int(x_center-box_w/2)
                            x2 = int(x_center+box_w/2)
                            y1 = int(y_center-box_h/2)
                            y2 = int(y_center+box_h/2)
                            print("x1, y1 ============", x1, y1)
                            print("x2, y2 ============", x2, y2)

                            cropping_img = im0s[y1:y2, x1:x2, :]
                            cropping_grayimg = cv2.cvtColor(
                                cropping_img, cv2.COLOR_BGR2GRAY)
                            cv2.imwrite('./test.jpg', cropping_grayimg)
                            unique, counts = np.unique(
                                cropping_grayimg, return_counts=True)
                            result = np.column_stack((unique, counts))
                            thres2 = 0
                            m_cnt = 0
                            for a, b in result:
                                thres2 += a*b
                                m_cnt += b
                            thres2 = int(thres2/m_cnt)
                            threshold = (int(max(unique))+int(min(unique)))/2

                            if threshold > 160:
                                threshold = int(threshold*0.80)

                            # print(result)
                            # print(threshold,thres2)
                            cropping_grayimg[cropping_grayimg < threshold] = 0
                            pixel_value = sum(
                                sum(cropping_grayimg < threshold))
                            cv2.imwrite('./test2.jpg', cropping_grayimg)
                            print('food_pixel: ', pixel_value)
                            print('bbox_size: ', box_w*box_h)
                            print('food/food_locate:',
                                  pixel_value/(box_w*box_h))
                        if c == 0:  # obj
                            obj_true = 1
                            x_obj = x_center
                            y_obj = y_center
                            print("x_obj, y_obj ============", x_obj, y_obj)
                            # x_obj, y_obj ============ 480 604
                            cv2.circle(im0, (x_center, y_center), 3,
                                       (0, 255, 0), cv2.FILLED, cv2.LINE_4)
                        if c == 1:  # neck_bone

                            x_neck_bone.append(x_center)  # 목뼈 x축 정보
                            y_neck_bone.append(y_center)  # 목뼈 y축 정보
                            # print("x_neck_bone, y_neck_bone ============", x_neck_bone, y_neck_bone)
                            # x_neck_bone, y_neck_bone ============ [690] [537]
                            # x_neck_bone, y_neck_bone ============ [690, 693] [537, 625]
                            # x_neck_bone, y_neck_bone ============ [690, 693, 715] [537, 625, 795]
                            # x_neck_bone, y_neck_bone ============ [690, 693, 715, 699] [537, 625, 795, 711]
                            # x_neck_bone, y_neck_bone ============ [690, 693, 715, 699, 687] [537, 625, 795, 711, 437]

                        label = None if hide_labels else (
                            names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(
                                xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            if len(y_neck_bone) > 2 and obj_true == 1:  # 목뼈가 두개 이상일 때 제일 위에 목뼈와 아래 목뼈를 찾음)

                min_y_value = min(y_neck_bone)
                max_y_value = max(y_neck_bone)
                y_neck_line = [min_y_value, max_y_value]
                max_y_index = y_neck_bone.index(max_y_value)
                min_y_index = y_neck_bone.index(min_y_value)
                min_x_value = x_neck_bone[min_y_index]
                max_x_value = x_neck_bone[max_y_index]
#                   print("min_x_value, min_y_value ============", min_x_value, min_y_value)
#                   min_x_value, min_y_value ============ 687 437

                if min_x_value - max_x_value != 0:

                    x22, y22, cx, cy = crosspoint(
                        min_x_value, min_y_value, max_x_value, max_y_value, x_obj, y_obj)
                    x22 = int(x22)
                    y22 = int(y22)
                    cx = int(cx)
                    cy = int(cy)

                    x_pit = int((w*0.75)-cx)
                    result = distance(x_obj, y_obj, cx, cy)

                    # 왼쪽에 설골, 목뼈, 직선
                    # org obj center ## 설골에 점 찍기
                    cv2.circle(im0, (x_obj, y_obj), 10,
                               (0, 0, 255), cv2.FILLED, cv2.LINE_4)
                    cv2.line(im0, (min_x_value, min_y_value), (max_x_value,
                             max_y_value), (0, 0, 255), 3, cv2.LINE_AA)  # 원본 목뼈 부분 직선
                    # org obj-neck_bone line ##원본 설골과 목뼈 연결 직선
                    cv2.line(im0, (x_obj, y_obj), (x22, y22),
                             (0, 0, 255), 3, cv2.LINE_AA)

                    # 왼쪽에 보정된 설골, 목뼈, 직선
                    # mod obj center ## 수정된 설골에 점찍기
                    cv2.circle(im0, (int(cx-result), cy), 10,
                               (255, 0, 0), cv2.FILLED, cv2.LINE_4)
                    # mod neck_bone line ##수정된 설골과 목뼈 직선
                    cv2.line(im0, (int(cx-result), cy), (x22, cy),
                             (0, 0, 255), 3, cv2.LINE_AA)
                    # mod neck_bone line  ## 수정된 목뼈직선
                    cv2.line(im0, (cx, min_y_value), (cx, max_y_value),
                             (0, 0, 255), 3, cv2.LINE_AA)

                    # 오른쪽 보정된 설골 그림 (파란색)
                    cv2.circle(zeros_im0, (x_obj, y_obj), 3, (0, 0, 255),
                               cv2.FILLED, cv2.LINE_4)  # org obj center
                    hyoid_bone_x.append(x_obj)
                    hyoid_bone_y.append(y_obj)

                    # 오른쪽 설골 그림 (빨간색)
                    # cv2.circle(zeros_im0, (int(cx-result), cy), 3, (255, 0, 0), cv2.FILLED, cv2.LINE_4)  ## 수정된 설골 위치를 그림

                    # 리스트를 데이터프레임으로 변환(list to DataFrame)
                    df = DataFrame(hyoid_bone_x, hyoid_bone_y)
                    # 열 이름 지정
                    df2 = DataFrame(
                        zip(hyoid_bone_x, hyoid_bone_y), columns=['x', 'y'])
                    # x 정렬
                    df3 = df2.sort_values("x")
                    # y 정렬
                    df4 = df2.sort_values("y")

                    # ast frame check
                    print(
                        f'len(df2)++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ {len(df2)}')
                    print(
                        f'count_1 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ {count_1}')

                    # mp4 => 동영상 전체 프레임개수는 (vid_frames) 개
                    # avi => 동영상 전체 프레임개수는 (vid_frames - 2) 개
                    if ((vid_frames == count_2) & ("mp4" in path)) or (((vid_frames - 2) == count_2) & ("avi" in path)):

                        #                     if (vid_frames - 2) == count_2 :
                        ################## 이동 시작 ########################
                        print(f'Done. ({time.time() - t0:.3f}s)')
                        # Done. (5.026s)
                        print(f'hyoid_bone_x = {hyoid_bone_x}')
                        print(f'hyoid_bone_y = {hyoid_bone_y}')

                        # x,y 각각 최대값, 최소값
                        xy_max = df2.max(axis=0)
                        xy_min = df2.min(axis=0)

                        # pdy.Series(hyoid_bone_y)
                        # print(f'pdx =================== {pdx}')
    #                     print(f'df =================== \n{df}')
    #                     print(f'df.iloc[0] =================== \n{df.iloc[0:1]}') # 첫 1개행만
                    #     df.iloc[0] ===================
                    #            0
                    #     612  460
                        # df = df.iloc[0:1][0]
                        print(f' df2.iloc[0][x] = {df2.iloc[0]["x"]}')
                        print(f' df2.iloc[0][y] = {df2.iloc[0]["y"]}')
                        # print(f'df.iloc[0:1][0] =================== \n{df.iloc[0:1][0]}') # 첫 1개행만
                    # df.iloc[0:1][0] ===================
                    # 612    460

    #                     print(f'df2 =================== {df2}')
    #                     print(f'len(df2) =================== {len(df2)}')
    #
    #                     for b in range(len(df2)):
    #                         print(f' df2.iloc[{b}][{b}] = {df2.iloc[b]["x"], df2.iloc[b]["y"]}')

                        # print(f'len(df2) =================== {len(df2)}')
                        # len(df2) =================== 112
                        # df2 ===================
                        #       x    y
                        # 0    460  612
                        # 1    460  612
                        # 2    459  613
                        # 3    459  614
                        # 4    460  616
                        # ..   ...  ...
                        # 107  480  605
                        # 108  480  604
                        # 109  480  604
                        # 110  480  604
                        # 111  480  604

                        print(f'df2.x =================== {df2.x}')
                        print(f'x 정렬 =================== {df3}')
                        print(f'y 정렬 =================== {df4}')

                        print(f'xy최대값 =================== {xy_max}')
                        print(f'xy최소값 =================== {xy_min}')

                        print(f'x최대값 =================== {xy_max[0]}')
                        print(f'y최대값 =================== {xy_max[1]}')

                        print(f'x최소값 =================== {xy_min[0]}')
                        print(f'y최소값 =================== {xy_min[1]}')

                        x_min = xy_min[0]
                        x_min1 = df2.query('x == @x_min')
                        print(f' x 최소값 =================== \n{x_min1}')
                        print(
                            f' C x 최소값, y 최소값 1쌍만 =================== \n{x_min1.min(axis=0)}')
                        print(
                            f' df2.loc[(df2[y] == y_max)].iloc[0][x] ===================  \n{x_min1.iloc[0]["x"]}')
                        print(
                            f' df2.loc[(df2[y] == y_max)].iloc[1][y] ===================  \n{x_min1.iloc[0]["y"]}')

                        x_max = xy_max[0]
                        x_max1 = df2.query('x == @x_max')
                        print(f' x 최대값 =================== \n{x_max1}')
                        print(
                            f' A x 최대값, y 최대값 1쌍만 ===================  \n{x_max1.max(axis=0)}')
                        print(
                            f' df2.loc[(df2[y] == y_max)].iloc[0][x] ===================  \n{x_max1.iloc[0]["x"]}')
                        print(
                            f' df2.loc[(df2[y] == y_max)].iloc[1][y] ===================  \n{x_max1.iloc[0]["y"]}')

                        y_min = xy_min[1]
                        y_min1 = df2.query('y == @y_min')
                        print(f' y 최소값 =================== \n{y_min1}')
                        print(
                            f' D x 최소값, y 최소값 1쌍만 =================== \n{y_min1.min(axis=0)}')
                        print(
                            f' df2.loc[(df2[y] == y_max)].iloc[0][x] ===================  \n{y_min1.iloc[0]["x"]}')
                        print(
                            f' df2.loc[(df2[y] == y_max)].iloc[1][y] ===================  \n{y_min1.iloc[0]["y"]}')

                        y_max = xy_max[1]
                        y_max1 = df2.query('y == @y_max')
                        print(f' y 최대값 =================== \n{y_max1}')
                        print(
                            f' B x 최대값, y 최대값 1쌍만 ===================  \n{y_max1.max(axis=0)}')
                        print(
                            f' df2.loc[(df2[y] == y_max)].iloc[0][x] ===================  \n{y_max1.iloc[0]["x"]}')
                        print(
                            f' df2.loc[(df2[y] == y_max)].iloc[1][y] ===================  \n{y_max1.iloc[0]["y"]}')

                        # interoperate 함수로 보간법을 적용하여 linear(선형보정) quadratic(부드러운 보정) 두가지 방법으로 만든다
                        # fl = interp1d(x,y,kind = 'linear')
#                         fq = interp1d(hyoid_bone_x,hyoid_bone_y,kind = 'quadratic')
#                         xint = np.linspace(x_min,x_max, 100)
#                         print(f'xint = {xint}')
#                         xintq = fq(xint)
#                         print(f'xintq = {xintq}')

                        # 텍스트
                        # cv2.putText(zeros_im0, str(y_max1.max(axis=0)) , (480, 600), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                        # A start
                        # cv2.putText(zeros_im0, "A" , (df2.iloc[0]["x"]+50, df2.iloc[0]["y"]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                        # cv2.putText(zeros_im0, "A" , (460+50, 612), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(zeros_im0, "A", (x_max+50, int((y_min + y_max)/2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

                        # B, Max Y, y최소값보다 적게(y좌표가 뒤바뀜)
                        # cv2.putText(zeros_im0, "B" , (y_min1.iloc[0]["x"], y_min1.iloc[0]["y"]-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(zeros_im0, "B", (int((x_min + x_max)/2), y_min-50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

                        # C, Max X, x최소값보다 적게(x좌표가 뒤바뀜)
                        # cv2.putText(zeros_im0, "C" , (kk.iloc[0]["x"]-50, kk.iloc[0]["y"]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(zeros_im0, "C", (x_min-50, int((y_min + y_max)/2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

                        # D, Min Y, y최대갸값보다 크게(y좌표가 뒤바뀜)
                        # cv2.putText(zeros_im0, "D" , (y_max1.iloc[0]["x"], y_max1.iloc[0]["y"]+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(zeros_im0, "D", (int((x_min + x_max)/2), y_max+50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

                        ################## 이동 끝 ########################

                        print(f'df2 =================== {df2}')
                        print(f'len(df2) =================== {len(df2)}')

                        B_point_count = 0
                        C_point_count = 0
                        D_point_count = 0

#
#     #A와 C사이에 B점을 찾기 위한 값
#     search_B_x=[]
#     search_B_y=[]
#
#     #C에서 끝까지 D점을 찾기 위한 값
#     search_D_x=[]
#     search_D_y=[]

                        for b in range(len(df2)):
                            #                             print(f' df2.iloc[{b}][{b}] = {df2.iloc[b]["x"], df2.iloc[b]["y"]}')
                            str1 = str(df2.iloc[b]["x"]) + \
                                str(df2.iloc[b]["y"])
                            # print(f' str1 = {str1}')
                            A_point = str(df2.iloc[0]["x"]) + \
                                str(df2.iloc[0]["y"])  # A
#                             B_point = str(y_min1.iloc[0]["x"]) + str(y_min1.iloc[0]["y"]) #B
                            C_point = str(
                                x_min1.iloc[0]["x"]) + str(x_min1.iloc[0]["y"])  # C
#                             D_point = str(y_max1.iloc[0]["x"]) + str(y_max1.iloc[0]["y"]) #D

#                             if B_point == str1:
#                                 print(f' ******************************************* find B point = {B_point}')
    #                             B_point_count = 1
                            if C_point == str1:
                                # "C_point == str1"인 하나의 점을 포함해서 값을 입력한다. 이렇게 하면, A~C에서 최고 y값을 가진 B를 찾을 때, B=C인 점도 찾게 된다.
                                search_B_x.append(df2.iloc[b]["x"])  # 목뼈 x축 정보
                                search_B_y.append(df2.iloc[b]["y"])  # 목뼈 x축 정보
                                search_B = 1
                                search_C = 1
                                # print(f' ******************************************* search_B_x = {search_B_x}')
    #                         if (B_point_count == 1) & (C_point == str1):
#                                 print(f' ******************************************* find C point = {C_point}')
    #                             C_point_count = 1
    #                         if (C_point_count == 1 ) & (D_point == str1):
                            # B점 구하기
                            elif ((C_point != str1) & (search_B == 0)):
                                search_B_x.append(df2.iloc[b]["x"])  # 목뼈 x축 정보
                                search_B_y.append(df2.iloc[b]["y"])  # 목뼈 x축 정보
                                # print(f' ******************************************* search_B_x = {search_B_x}')
                            # D점은 C와 끝점 사이에 y값이 max point
                            # if D_point == str1:
                            if search_C == 1:
                                # print(f' ******************************************* find D point = {D_point}')
                                search_D_x.append(df2.iloc[b]["x"])  # 목뼈 x축 정보
                                search_D_y.append(df2.iloc[b]["y"])  # 목뼈 x축 정보

                        # 리스트를 데이터프레임으로 변환(list to DataFrame)
                        df_search_B = DataFrame(
                            zip(search_B_x, search_B_y), columns=['x', 'y'])
                        df_search_D = DataFrame(
                            zip(search_D_x, search_D_y), columns=['x', 'y'])

                        # x,y 각각 최대값, 최소값
                        search_B_xy_max = df_search_B.max(axis=0)  # 중복 가능
                        search_B_xy_min = df_search_B.min(axis=0)  # 중복 가능

                        print(
                            f' search_B_xy_max ===================  \n{search_B_xy_max}')
                        print(
                            f' search_B_xy_min ===================  \n{search_B_xy_min}')
#  search_B_xy_max ===================
# x    473
# y    619
# dtype: int64
#  search_B_xy_min ===================
# x    399
# y    572
# dtype: int64
                        # x,y 각각 최대값, 최소값
                        search_D_xy_max = df_search_D.max(axis=0)  # 중복 가능
                        search_D_xy_min = df_search_D.min(axis=0)  # 중복 가능

                        print(
                            f' search_D_xy_max ===================  \n{search_D_xy_max}')
                        print(
                            f' search_D_xy_min ===================  \n{search_D_xy_min}')

                        search_B_x_min = search_B_xy_min[0]  # x 최소값
                        search_B_y_min = search_B_xy_min[1]  # y 최소값

                        print(
                            f' search_B_x_min ===================  \n{search_B_x_min}')
                        print(
                            f' search_B_y_min ===================  \n{search_B_y_min}')

#  search_B_x_min ===================
# 399
#  search_B_y_min ===================
# 572

                        search_D_x_max = search_D_xy_max[0]  # x max
                        search_D_y_max = search_D_xy_max[1]  # y max

                        print(
                            f' search_D_x_min ===================  \n{search_D_x_max}')
                        print(
                            f' search_D_y_min ===================  \n{search_D_y_max}')

                        # A~C구간에서 y값이 최소인 쌍 찾기
                        B_y_min = df_search_B.query('y == @search_B_y_min')
                        print(
                            f' B_y_min x ===================  \n{B_y_min.iloc[0]["x"]}')
                        print(
                            f' B_y_min y ===================  \n{B_y_min.iloc[0]["y"]}')
#
#  B_y_min x ===================
# 399
#  B_y_min y ===================
# 572

                        # C~End 구간에서 y값이 max인 쌍 찾기
                        D_y_max = df_search_D.query('y == @search_D_y_max')
                        print(
                            f' D_y_min x ===================  \n{D_y_max.iloc[0]["x"]}')
                        print(
                            f' D_y_min y ===================  \n{D_y_max.iloc[0]["y"]}')

                        # A point
                        print(
                            f' A point : {df2.iloc[0]["x"]}, {df2.iloc[0]["y"]}')
                        # B point
#                         print(f' B point : {y_min1.iloc[0]["x"]}, {y_min1.iloc[0]["y"]}')
                        print(
                            f' B point : {B_y_min.iloc[0]["x"]}, {B_y_min.iloc[0]["y"]}')

                        B_point = str(
                            B_y_min.iloc[0]["x"]) + str(B_y_min.iloc[0]["y"])  # B
                        # C point
                        print(
                            f' C point : {x_min1.iloc[0]["x"]}, {x_min1.iloc[0]["y"]}')
                        # D point
#                         print(f' D point : {y_max1.iloc[0]["x"]}, {y_max1.iloc[0]["y"]}')
                        print(
                            f' D point : {D_y_max.iloc[0]["x"]}, {D_y_max.iloc[0]["y"]}')
                        D_point = str(
                            D_y_max.iloc[0]["x"]) + str(D_y_max.iloc[0]["y"])  # B

                        # time calculation value initialization
                        dist_AB_y = 0
                        dist_AC_y = 0
                        dist_AD_y = 0
                        dist_BC_y = 0
                        dist_BD_y = 0
                        dist_AB_x = 0
                        dist_AC_x = 0
                        dist_AD_x = 0
                        dist_AC_straight = 0

                        dist_AB_y = abs(
                            df2.iloc[0]["y"] - B_y_min.iloc[0]["y"])
                        dist_AC_y = abs(df2.iloc[0]["y"] - x_min1.iloc[0]["y"])
                        dist_AD_y = abs(
                            df2.iloc[0]["y"] - D_y_max.iloc[0]["x"])
                        dist_BC_y = abs(
                            B_y_min.iloc[0]["y"] - x_min1.iloc[0]["y"])
                        dist_BD_y = abs(
                            B_y_min.iloc[0]["y"] - D_y_max.iloc[0]["y"])
                        dist_AB_x = abs(
                            df2.iloc[0]["x"] - B_y_min.iloc[0]["x"])
                        dist_AC_x = abs(df2.iloc[0]["x"] - x_min1.iloc[0]["x"])
                        dist_AD_x = abs(
                            df2.iloc[0]["x"] - D_y_max.iloc[0]["x"])
                        dist_AC_straight = round(math.sqrt(
                            (abs(df2.iloc[0]["x"] - x_min1.iloc[0]["x"]))**2 + (abs(df2.iloc[0]["y"] - x_min1.iloc[0]["y"]))**2), 2)

                        # A~B Y축 거리
                        print(f' A~B Y축 거리 : {dist_AB_y}')
                        # A~C Y축 거리
                        print(f' A~C Y축 거리 : {dist_AC_y}')
                        # A~D Y축 거리
                        print(f' A~D Y축 거리 : {dist_AD_y}')
                        # B~C Y축 거리
                        print(f' B~C Y축 거리 : {dist_BC_y}')
                        # B~D Y축 거리
                        print(f' B~D Y축 거리 : {dist_BD_y}')
                        # A~B X축 거리
                        print(f' A~B X축 거리 : {dist_AB_x}')
                        # A~C X축 거리
                        print(f' A~C X축 거리 : {dist_AC_x}')
                        # A~D X축 거리
                        print(f' A~D X축 거리 : {dist_AD_x}')
                        # A~C 의 직선거리
                        print(f' A~C의 직선거리 : {dist_AC_straight}')

                        # time calculation value initialization
                        count_AB = 0
                        count_AB_check = 0
                        count_AC = 0
                        count_AC_check = 0
                        count_AD = 0
                        count_BC = 0
                        count_BD = 0
                        count_CD = 0
                        count_A_check = 0

                        # speed calculation value initialization
                        speed_AB = []
                        speed_AC = []
                        speed_AD = []
                        speed_BC = []
                        speed_BD = []
                        speed_CD = []

                        # acceleration calculation value initialization
                        accel_AB = []
                        accel_AC = []
                        accel_AD = []
                        accel_BC = []
                        accel_BD = []
                        accel_CD = []

                        for c in range(len(df2)):
                            str2 = str(df2.iloc[c]["x"]) + \
                                str(df2.iloc[c]["y"])

                            # time calculation start
                            count_A_check += 1
                            if count_A_check > 1:  # exclude point A
                                if (B_point != str2) & (count_AB_check == 0):  # AB
                                    count_AB += 1  # point count
#                                     if count_A_check > 2 : # 출발점에서 1회 이동 후 계산해야 한다.
                                    speed_AB.append(math.sqrt((abs(df2.iloc[c]["x"] - df2.iloc[c-1]["x"]))**2 + (
                                        abs(df2.iloc[c]["y"] - df2.iloc[c-1]["y"]))**2))  # speed AB
#                                     print(f' {c}~{c-1}의 직선거리 : {speed_AB}')
#                                     print(f' count_AB : {count_AB}')
                                if (B_point == str2):
                                    count_AB_check = 1
                                if (C_point != str2) & (count_AC_check == 0):  # AC
                                    count_AC += 1
                                    speed_AC.append(math.sqrt((abs(df2.iloc[c]["x"] - df2.iloc[c-1]["x"]))**2 + (
                                        abs(df2.iloc[c]["y"] - df2.iloc[c-1]["y"]))**2))  # speed AC
#                                     print(f' {c}~{c-1}의 직선거리 : {speed_AC}')
#                                     print(f' count_AC : {count_AC}')
                                if (C_point == str2):
                                    count_AC_check = 1
                                if D_point != str2:  # AD
                                    count_AD += 1
                                    speed_AD.append(math.sqrt((abs(df2.iloc[c]["x"] - df2.iloc[c-1]["x"]))**2 + (
                                        abs(df2.iloc[c]["y"] - df2.iloc[c-1]["y"]))**2))  # speed AD
#                                     print(f' {c}~{c-1}의 직선거리 : {speed_AD}')
#                                     print(f' count_AD : {count_AD}')
                                if (count_AB_check == 1) & (count_AC_check == 0):  # BC
                                    count_BC += 1
                                    speed_BC.append(math.sqrt((abs(df2.iloc[c]["x"] - df2.iloc[c-1]["x"]))**2 + (
                                        abs(df2.iloc[c]["y"] - df2.iloc[c-1]["y"]))**2))  # speed BC
#                                     print(f' {c}~{c-1}의 직선거리 : {speed_BC}')
#                                     print(f' count_BC : {count_BC}')
                                if (count_AB_check == 1) & (D_point != str2):  # BD
                                    count_BD += 1
                                    speed_BD.append(math.sqrt((abs(df2.iloc[c]["x"] - df2.iloc[c-1]["x"]))**2 + (
                                        abs(df2.iloc[c]["y"] - df2.iloc[c-1]["y"]))**2))  # speed BD
#                                     print(f' {c}~{c-1}의 직선거리 : {speed_BD}')
#                                     print(f' count_BD : {count_BD}')
                                if (count_AC_check == 1) & (D_point != str2):  # CD
                                    count_CD += 1
                                    speed_CD.append(math.sqrt((abs(df2.iloc[c]["x"] - df2.iloc[c-1]["x"]))**2 + (
                                        abs(df2.iloc[c]["y"] - df2.iloc[c-1]["y"]))**2))  # speed CD
#                                     print(f' {c}~{c-1}의 직선거리 : {speed_CD}')
#                                     print(f' count_CD : {count_CD}')

                            # time calculation end

                        # A~B 점까지의 모든 선분의 합 길이 = A~B speed sum
                        sum_dist_AB = round(sum(speed_AB), 2)
                        # A~C 점까지의 모든 선분의 합 길이 = A~C speed sum
                        sum_dist_AC = round(sum(speed_AC), 2)
                        # A~D 점까지의 모든 선분의 합 길이 = A~D speed sum
                        sum_dist_AD = round(sum(speed_AD), 2)
                        print(f' A~B 점까지의 모든 선분의 합 길이 : {sum_dist_AB}')
                        print(f' A~C 점까지의 모든 선분의 합 길이 : {sum_dist_AC}')
                        print(f' A~D 점까지의 모든 선분의 합 길이 : {sum_dist_AD}')

                        # A~B point sum
                        print(f' A~B point sum : {count_AB}')
                        # A~C point sum
                        print(f' A~C point sum : {count_AC}')
                        # A~D point sum
                        print(f' A~D point sum : {count_AD}')
                        # B~C point sum
                        print(f' B~C point sum : {count_BC}')
                        # B~D point sum
                        print(f' B~D point sum : {count_BD}')
                        # C~D point sum
                        print(f' C~D point sum : {count_CD}')

                        # SPEED
#                         print(f' A~B speed : {speed_AB}')
#                         print(f' average A~B speed : {sum(speed_AB)/len(speed_AB)}')
#                         print(f' A~C speed : {speed_AC}')
#                         print(f' average A~C speed : {sum(speed_AC)/len(speed_AC)}')
#                         print(f' A~D speed : {speed_AD}')
#                         print(f' average A~D speed : {sum(speed_AD)/len(speed_AD)}')
#                         print(f' B~C speed : {speed_BC}')
#                         print(f' average B~C speed : {sum(speed_BC)/len(speed_BC)}')
#                         print(f' B~D speed : {speed_BD}')
#                         print(f' average B~D speed : {sum(speed_BD)/len(speed_BD)}')
#                         print(f' C~D speed : {speed_CD}')
#                         print(f' average C~D speed : {sum(speed_CD)/len(speed_CD)}')
#                         print(f' total average speed : {sum(speed_AD)/len(speed_AD)}')
#                         print(f' total max speed : {max(speed_AD)}')

                        if len(speed_AB) == 0:
                            average_AB_speed = 0
                        else:
                            average_AB_speed = round(
                                sum(speed_AB)/len(speed_AB), 2)

                        if len(speed_AC) == 0:
                            average_AC_speed = 0
                        else:
                            average_AC_speed = round(
                                sum(speed_AC)/len(speed_AC), 2)

                        if len(speed_AD) == 0:
                            average_AD_speed = 0
                        else:
                            average_AD_speed = round(
                                sum(speed_AD)/len(speed_AD), 2)

                        if len(speed_BC) == 0:
                            average_BC_speed = 0
                        else:
                            average_BC_speed = round(
                                sum(speed_BC)/len(speed_BC), 2)

                        if len(speed_BD) == 0:
                            average_BD_speed = 0
                        else:
                            average_BD_speed = round(
                                sum(speed_BD)/len(speed_BD), 2)

                        if len(speed_CD) == 0:
                            average_CD_speed = 0
                        else:
                            average_CD_speed = round(
                                sum(speed_CD)/len(speed_CD), 2)

                        total_average_speed = average_AD_speed
                        total_max_speed = round(max(speed_AD), 2)

                        print(f' average A~B speed : {average_AB_speed}')
                        print(f' average A~C speed : {average_AC_speed}')
                        print(f' average A~D speed : {average_AD_speed}')
                        print(f' average B~C speed : {average_BC_speed}')
                        print(f' average B~D speed : {average_BD_speed}')
                        print(f' average C~D speed : {average_CD_speed}')
                        print(f' total average speed : {total_average_speed}')
                        print(f' total max speed : {total_max_speed}')

                        # Acceleration
                        for d in range(len(speed_AB)):
                            if d > 0:
                                # print(f' speed_AB[{d}] : {speed_AB[d]}') #speed_AB[0] : 0.0 => skip, speed_AB[1] : 1.4142135623730951 => start
                                accel_AB.append(
                                    abs(speed_AB[d] - speed_AB[d-1]))
                        print(f' accel_AB : {accel_AB}')
                        for d in range(len(speed_AC)):
                            if d > 0:
                                # print(f' speed_AB[{d}] : {speed_AB[d]}') #speed_AB[0] : 0.0 => skip, speed_AB[1] : 1.4142135623730951 => start
                                accel_AC.append(
                                    abs(speed_AC[d] - speed_AC[d-1]))
                        print(f' accel_AC : {accel_AC}')
                        for d in range(len(speed_AD)):
                            if d > 0:
                                accel_AD.append(
                                    abs(speed_AD[d] - speed_AD[d-1]))
                        print(f' accel_AD : {accel_AD}')
                        for d in range(len(speed_BC)):
                            if d > 0:
                                accel_BC.append(
                                    abs(speed_BC[d] - speed_BC[d-1]))
                        print(f' accel_BC : {accel_BC}')
                        for d in range(len(speed_BD)):
                            if d > 0:
                                accel_BD.append(
                                    abs(speed_BD[d] - speed_BD[d-1]))
                        print(f' accel_BD : {accel_BD}')
                        for d in range(len(speed_CD)):
                            if d > 0:
                                accel_CD.append(
                                    abs(speed_CD[d] - speed_CD[d-1]))
                        print(f' accel_CD : {accel_CD}')

                        if len(accel_AB) == 0:
                            average_AB_accel = 0
                        else:
                            average_AB_accel = round(
                                sum(accel_AB)/len(accel_AB), 2)
                        if len(accel_AC) == 0:
                            average_AC_accel = 0
                        else:
                            average_AC_accel = round(
                                sum(accel_AC)/len(accel_AC), 2)
                        if len(accel_AD) == 0:
                            average_AD_accel = 0
                        else:
                            average_AD_accel = round(
                                sum(accel_AD)/len(accel_AD), 2)
                        if len(accel_BC) == 0:
                            average_BC_accel = 0
                        else:
                            average_BC_accel = round(
                                sum(accel_BC)/len(accel_BC), 2)
                        if len(accel_BD) == 0:
                            average_BD_accel = 0
                        else:
                            average_BD_accel = round(
                                sum(accel_BD)/len(accel_BD), 2)
                        if len(accel_CD) == 0:
                            average_CD_accel = 0
                        else:
                            average_CD_accel = round(
                                sum(accel_CD)/len(accel_CD), 2)

                        total_average_accel = average_AD_accel
                        total_max_accel = round(max(accel_AD), 2)

                        print(f' average A~B accel : {average_AB_accel}')
                        print(f' average A~C accel : {average_AC_accel}')
                        print(f' average A~D accel : {average_AD_accel}')
                        print(f' average B~C accel : {average_BC_accel}')
                        print(f' average B~D accel : {average_BD_accel}')
                        print(f' average C~D accel : {average_CD_accel}')
                        print(f' total average accel : {total_average_accel}')
                        print(f' total max accel : {total_max_accel}')

            # 텍스트
            # cv2.putText(zeros_im0, "drawing test", (x_obj, y_obj), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

            add_img = cv2.addWeighted(zeros_im0, 0.2, test_img, 0.8, 0)

            # 텍스트 : cv2.putText(img, text, org, fontFace, fontScale, color [, thickness [, lineType [, bottomLeftOrigin]]])
            # text : 출력할 텍스트 문자열
            # org : 영상에 출력할 텍스트 문자열의 왼쪽 아래 모서리 좌표
            # fontFace : 글꼴 유형
            # FontScale : 글꼴 크기(배율)
            # bottomLeftOrigin : True일 시 데이터 원점이 왼쪽 하단 모서리, 아니면 왼쪽 상단 모서리

            dst = np.hstack((im0, add_img))

            print(f'{s}Done. ({t2 - t1:.3f}s)')
            # 608x736 1 obj, 5 neck_bones, Done. (0.009s)

            # Stream results
            # im0 = annotator.result()
            im0 = np.asarray(dst)
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                print(f' +++++++++++++++++++++++++++++++++++ ')
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        cnt = 0
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            # release previous video writer
                            vid_writer[i].release()
                        if vid_cap:  # video
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            # fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    print(fps, w, h)
                    vid_writer[i].write(im0)
#     if save_txt or save_img:
#         s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
#         print(f"Results saved to {colorstr('bold', save_dir)}{s}")
        # 프레임 갯수 카운팅
        count_1 += 1
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")


#  x 최소값 ===================
#      x    y
# 56  399  572
# 57  399  571
#  x 최소값, y 최소값 1쌍만 ===================
# x    399
# y    571
# dtype: int64
#  x 최대값 ===================
#       x    y
# 96   480  604
# 97   480  604
# 98   480  604
# 99   480  604
# 100  480  604
# 101  480  604
# 102  480  604
# 103  480  604
# 104  480  604
# 105  480  604
# 106  480  604
# 107  480  605
# 108  480  604
# 109  480  604
# 110  480  604
# 111  480  604
#  x 최대값, y 최대값 1쌍만 ===================
# x    480
# y    605


# x최대값 =================== 480
# y최대값 =================== 619
# x최소값 =================== 399
# y최소값 =================== 571


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='yolov5s.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str, default='data/의료영상전체데이터셋_2023_28개_split/test/images',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+',
                        type=int, default=[720], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000,
                        help='maximum detections per image')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true',
                        help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize features')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3,
                        type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False,
                        action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False,
                        action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    print(colorstr('detect: ') +
          ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
