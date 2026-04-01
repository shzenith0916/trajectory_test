# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

# 추가 라이브러리 import
from utils.torch_utils import select_device, smart_inference_mode
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from models.common import DetectMultiBackend
from ultralytics.utils.plotting import Annotator, colors, save_one_box
import cv2
import numpy as np
import math
from pyparsing import C
import scipy.interpolate
import matplotlib.pyplot as plt
import pandas as pd

# 기존 라이브러리 import
import argparse
import csv
import os
import platform
import re
import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith(
        ".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(
        ".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name,
                              exist_ok=exist_ok)  # increment run
    # (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / "labels").mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(
        weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # 추가 부분

    trajectory_path = ""
    # Hyoid Bone List
    hyoid_x_list = []
    hyoid_y_list = []
    bluedot_x_list = []
    bluedot_y_list = []
    # 추가 끝

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz,
                              stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(
            source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz,
                             stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(
        device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)
        # Inference
        with dt[1]:
            visualize = increment_path(
                save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment,
                                     visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat(
                            (pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name,
                    "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # 이미지별 객체탐지 detections per image file
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # Converting the file path to Path object
            save_path = str(save_dir / p.name)  # im.jpg

            # 추가
            original_stem = p.stem
            masked_stem = mask_korean_name(original_stem)
            suffix = p.suffix
            if suffix == ".jpg":
                trajectory_path = "{}_trajectory{}".format(masked_stem, suffix)
                # print("trajectory_path is:", trajectory_path)
            elif suffix == ".avi":
                trajectory_path = f"{masked_stem}_trajectory.jpg"
                # print("trajectory_path is:", trajectory_path)
            # 추가 끝!!

            txt_path = str(save_dir / "labels" / p.stem) + \
                ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(
                im0, line_width=line_thickness, example=str(names))

            # 추가부분!!!
            neck_centers = []
            hyoid_bone_center = None
            # 추가 끝

            # detect의 배열구조 => 0:4까지는 x1,y1,x2,y2이고 4는 각 객체의 신뢰도confidence score(모델이 객체를 감지한 확률), 5는 각 객체의 클래스 식별자를 나타냄.
            if len(det):
                # Rescale boxes from img_size to im0 size 바운딩박스 리스케일
                det[:, :4] = scale_boxes(
                    im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    # 추가!!
                    # reference1:https://csm-kr.tistory.com/13
                    # reference2:https://github.com/ultralytics/yolov5/blob/master/utils/general.py
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                            gn).view(-1).tolist()  # normalized xywh
                   # print("normalized xywh:",xywh)
                    normalized_center = (xywh[0], xywh[1])
                   # print("normalized center:",normalized_center)
                    bbox_width = xywh[2]
                    bbox_height = xywh[3]
                   # print("normalized bbox width and heigth:", bbox_width, bbox_height)
                    width, height = im0.shape[1], im0.shape[0]

                    # Above "xywh" values are in tensor format. Needs to be in pixel value to get the actual point.
                    # Convert normalized center coordinates to pixel coordinates
                    center_x = int(normalized_center[0] * width)
                    center_y = int(normalized_center[1] * height)
                    pixel_center = (center_x, center_y)
                    # print("unnormalized bbox center values:", pixel_center)

                    # Classify dections
                    if c == 0:  # hyoid bone
                        hyoid_bone_center = pixel_center
                        # print("hyoid_bone_bbox_center:", hyoid_bone_center)

                        # 객체탐지로 찾아진 설골 센터 정보를 리스트에 저장
                        hyoid_x_list.append(hyoid_bone_center[0])
                        hyoid_y_list.append(hyoid_bone_center[1])

                    elif c == 1:  # neck bone
                        neck_centers.append(pixel_center)
                        # print("neck_bone_center:", neck_centers)

                    # 추가 끝!!!

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
                        print("xywh", xywh)
                        # label format
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (
                            names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" /
                                     names[c] / f"{p.stem}.jpg", BGR=True)

                # 추가!!
                # Check if neck bone BBox is at least 2 and hyoid bone bbox exsits, then draw a line
                if len(neck_centers) >= 2 and hyoid_bone_center:

                    # Convert neck center points to numpy array for cv2.fitLine
                    data_points = np.array(neck_centers, dtype=np.float32)
                    # print("neck bone center points:", data_points)

                    # Use the method to find sublists with min and max y-values of neck bone bboxes
                    sublist_with_min, sublist_with_max = find_min_max_points(
                        data_points)

                    # Fit line to the points, DIST_L2 -> euclidean distance
                    [vx, vy, x0, y0] = cv2.fitLine(
                        data_points, cv2.DIST_L2, 0, 0.01, 0.01)

                    # Slope and intercept of the main line. (y = mx + c)
                    m1 = vy / vx
                    c1 = y0 - m1 * x0

                    if m1 != 0:
                        x_min = (sublist_with_min[1] - c1) / m1
                        x_max = (sublist_with_max[1] - c1) / m1

                        # Ensure x_min and x_max are scalars
                        x_min = x_min.item() if isinstance(x_min, np.ndarray) else x_min
                        x_max = x_max.item() if isinstance(x_max, np.ndarray) else x_max

                        start_point = (int(x_min), int(
                            sublist_with_min[1]))  # 목뼈 직선의 처음 포인트
                        end_point = (int(x_max), int(
                            sublist_with_max[1]))  # 목뼈 직선의 끝 포인트

                    else:
                        # If the line is vertical, use the x-coordinate directly from the points
                        start_point = (
                            int(sublist_with_min[0]), int(sublist_with_min[1]))
                        end_point = (int(sublist_with_max[0]), int(
                            sublist_with_max[1]))

                    # 목뼈들을 이은 초록색 근사선 Draw the main line in green
                    cv2.line(im0, start_point, end_point,
                             color=(0, 255, 0), thickness=2)

                    # Calculate slope and intercpet of the perpendicular line from the Hyoid bone center
                    m2 = -1 / m1
                    c2 = hyoid_bone_center[1] - m2 * hyoid_bone_center[0]

                    # Calculate intersection using crosspoint function
                    try:
                        # intercept_x and intercept_y are numpy array.
                        intercept_x, intercept_y = crosspoint(m1, c1, m2, c2)

                        # Change to the "scalar" format to calcaulate math.dist later
                        intercept_x_scalar = intercept_x.item()  # Correctly converts to scalar(float)
                        # Assuming intercept_y is numpy arrays with a single element
                        intercept_y_scalar = intercept_y.item()

                        # 설골에서 목뼈선의 교차포인트까지의 초록 선. Draw the perpendicular line
                        cv2.line(im0, (int(hyoid_bone_center[0]), int(hyoid_bone_center[1])), (int(
                            intercept_x_scalar), int(intercept_y_scalar)), (0, 255, 0), 2)
                        # 설골 경계상자 센터점 (빨간색) hyoid bbox center point (red dot)
                        cv2.circle(im0, (int(hyoid_bone_center[0]), int(
                            hyoid_bone_center[1])), 5, (0, 0, 255), -1)
                        # 목뼈들을 이은 근사선에서 설골까지 거리 distance from the fitted line to cetner point of hyoid bbox
                        distance = math.dist(
                            (hyoid_bone_center[0], hyoid_bone_center[1]), (intercept_x_scalar, intercept_y_scalar))
                        # math.dist expects scarlar but the given variables are in an array.
                        # print("Distance between points ({}, {}) and ({}, {}) is: {}".format(hyoid_bone_center[0], hyoid_bone_center[1], intercept_x, intercept_y, distance))

                        # Horizontal line at the y-intersect, 수평을 이루는 기준 하얀색 선, 목뼈에서의 시작 포인트
                        horizontal_start_point = (
                            int(intercept_x_scalar - distance), int(intercept_y_scalar))
                        # 수평을 이루는 기준 하얀색 선, 목뼈에서의 끝끝 포인트
                        horizontal_end_point = (
                            int(intercept_x_scalar), int(intercept_y_scalar))

                        # 보정된 설골 점 x,y 값을 위의 리스트에 저장
                        bluedot_x_list.append(horizontal_start_point[0])
                        bluedot_y_list.append(horizontal_start_point[1])

                        # 보정된 설골점에서 수평을 이루는 하얀색 선 Draw the horizontal line in blue
                        cv2.line(im0, horizontal_start_point, horizontal_end_point, color=(
                            255, 255, 255), thickness=2)
                        # 보정된 파란색 설골 점. Draw the "Blue DOT" for the modified point
                        cv2.circle(im0, horizontal_start_point,
                                   5, (255, 0, 0), -1)

                        # 보정된 설골점에서 직각인 보정된 목뼈선. Vertical line at the x-intersect
                        vertical_start_point = (
                            int(intercept_x_scalar), start_point[1])
                        vertical_end_point = (
                            int(intercept_x_scalar), end_point[1])
                        # 보정된 하얀색 목뼈선 Draw the vertical line in white
                        cv2.line(im0, vertical_start_point, vertical_end_point, color=(
                            255, 255, 255), thickness=2)

                    except ValueError as e:
                        print(e)

                # 추가 끝!!

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    # allow window resize (Linux)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL |
                                    cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            # release previous video writer
                            vid_writer[i].release()
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        # force *.avi suffix on results videos
                        save_path = str(Path(save_path).with_suffix(".avi"))
                        vid_writer[i] = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(
            f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # 추가

    # Create dictionary to save the hyoid trajectory coordinates
    hyoid_data = {"hyoid_x_coor": [hyoid_x_coor],
                  "hyoid_y_coor": [hyoid_y_coor]}
    print("test printing hyoid data", hyoid_data)

    # 데이터 점들로 이동/방향을 보여주기 위해서 np.diff를 계산하려면 리스트를 배열로 바꾸어 주어야함.
    # Convert lists to NumPy arrays to calculate the descrete difference between consecutive elements of arrays.
    hyoid_x_coor = np.array(hyoid_x_list)
    hyoid_y_coor = np.array(hyoid_y_list)
    # print("test printing hyoid_x_coor",hyoid_x_coor)
    # print("test printing hyoid_y_coor",hyoid_y_coor)

    # Diff calculation for directional change
    dx = np.diff(hyoid_x_coor)
    dy = np.diff(hyoid_y_coor)

    # Min, Max points of x_coor and y_coor
    min_x_idx = np.argmin(hyoid_x_coor)
    max_x_idx = np.argmax(hyoid_x_coor)
    min_y_idx = np.argmin(hyoid_y_coor)
    max_y_idx = np.argmax(hyoid_y_coor)

    # 맨처음 명시한 배열을 이용하여, 궤적용 빈 이미지에 파란 설골점 그리기.
    # Ploting the points using the hyoid_x_coor and hyoid_y_coor arrays.
    plt.figure(figsize=(8, 6))
    plt.plot(hyoid_x_coor, hyoid_y_coor, 'b-', marker='o',
             color='cadetblue', linestyle='-', markersize=3, linewidth=1)

    # 방향표시 Add arrows to show direction (quiver plot)
    for i in range(len(dx)):
        plt.quiver(hyoid_x_coor[i], hyoid_y_coor[i], dx[i], dy[i], angles='xy',
                   scale_units='xy', scale=1.5, color='darksalmon', width=0.003)
    # YOLO 객체 탐지 결과 데이터를 이미지 좌표 시스템에서 수학 좌표 시스템으로 바꾸기
    # Invert the y axis to match image coordinate system
    plt.gca().invert_yaxis()

    # Highlighting the START point (A point)
    plt.scatter(hyoid_x_coor[0], hyoid_y_coor[0],
                color='orange', s=100, label='Start')
    plt.annotate('Start(A)', (hyoid_x_coor[0], hyoid_y_coor[0]), textcoords='offset points', xytext=(
        20, 10), ha='center', fontsize=10)
    # Highlighting the END point (D point)
    plt.scatter(hyoid_x_coor[-1], hyoid_y_coor[-1],
                color='seagreen', s=100, label='End')
    plt.annotate('End', (hyoid_x_coor[-1], hyoid_y_coor[-1]),
                 textcoords='offset points', xytext=(0, 10), ha='center', fontsize=10)

    # C point
    plt.scatter(hyoid_x_coor[min_x_idx], hyoid_y_coor[min_x_idx],
                color='peru', s=80, label='C_point')
    plt.annotate('X_Min(C)', (hyoid_x_coor[min_x_idx], hyoid_y_coor[min_x_idx]), textcoords='offset points', xytext=(
        60, -30), ha='center', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.8"))
    # Max x point
    plt.scatter(hyoid_x_coor[max_x_idx], hyoid_y_coor[max_x_idx],
                color='gold', s=80, label='A_point')
    plt.annotate('X_Max', (hyoid_x_coor[max_x_idx], hyoid_y_coor[max_x_idx]), textcoords='offset points', xytext=(
        -10, 20), ha='center', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.5"))
    # B point
    plt.scatter(hyoid_x_coor[min_y_idx], hyoid_y_coor[min_y_idx],
                color='orchid', s=80, label='B_point')
    plt.annotate('Y_Max(B)', (hyoid_x_coor[min_y_idx], hyoid_y_coor[min_y_idx]), textcoords='offset points', xytext=(
        40, -10), ha='center', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.5"))
    # D Point
    plt.scatter(hyoid_x_coor[max_y_idx], hyoid_y_coor[max_y_idx],
                color='royalblue', s=80, label='D_point')
    plt.annotate('Y_Min(D)', (hyoid_x_coor[max_y_idx], hyoid_y_coor[max_y_idx]), textcoords='offset points', xytext=(
        30, 30), ha='center', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.8"))

    # Set up the plot
    # margin=15
    # x_min, x_max = min(hyoid_x_coor) - margin, max(hyoid_x_coor) + margin
    # y_min, y_max = min(hyoid_y_coor) - margin, max(hyoid_y_coor) + margin
    # Set inverted limits directly
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)

    # Adjust y-axis labels to increase upwards
    plt.yticks(range(min(hyoid_y_coor), max(hyoid_y_coor) + 1, 10),  # 10,-10은 grid, tick의 간격
               range(max(hyoid_y_coor), min(hyoid_y_coor) - 1, -10))

    title = stem[0:8] + '_' + stem[-10:-8]
    plt.title("Trajectory of {}".format(title))
    plt.xlabel("X Coordinate of Hyoid", fontsize=10)
    plt.ylabel("Y Coordinate of Hyoid", fontsize=10)
    plt.grid(True)
    # Save the plot
    plt.savefig(trajectory_path)
    plt.close()

    # 데이터 프레임 생성 후 궤적 데이터 CSV로 저장
    save_trajectories_to_csv(hyoid_x_list, hyoid_y_list,
                             bluedot_x_list, bluedot_y_list, save_dir, clean_stem)
    # 추가 끝!!!

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        # update model (to fix SourceChangeWarning)
        strip_optimizer(weights[0])

# 추가!!


def crosspoint(m1, c1, m2, c2):
    '''두 라인의 서로 만나는 점 계산
      calculating the intersection point of two lines given their slopes m1,m2 and intercepts c1,c2
    '''
    # check if lines are parallel
    if m1 == m2:
        raise ValueError("The lines are parallel and do not intersect")

    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1

    return x, y


def find_min_max_points(data_points):
    '''
    목뼈 경계상자들의 중심 포인트 좌표를 가지는 data_point에서 
    y값이 최고, 최저 값을 가지는 좌표값 찾기
    Find the max and min values from the list having Neck_Bone class
    '''
    # Ensure data_points is a NumPy array
    data_points = np.array(data_points)

    # Find the second element from each sublist
    second_elements = data_points[:, 1]
    # Find the minimum and maximum values in these extracted elements
    min_value = np.min(second_elements)
    max_value = np.max(second_elements)
    # Find the index of the sublist containing the minimum and maximum
    min_index = np.argmin(second_elements)
    max_index = np.argmax(second_elements)
    # Retrieve the sublist that contains the minimum and maximum values
    sublist_with_min = data_points[min_index]
    sublist_with_max = data_points[max_index]

    return sublist_with_min, sublist_with_max


def save_trajectories_to_csv(red_x, red_y, blue_x, blue_y, save_dir, base_filename):
    """
    주어진 궤적 좌표 리스트들을 pandas DataFrame으로 만들어 CSV 파일로 저장합니다.
    """
    # pandas Series를 사용하면 길이가 다른 리스트도 NaN으로 채워져 안전하게 DataFrame으로 만들 수 있습니다.
    df = pd.DataFrame({
        'red_dot_x': pd.Series(red_x),
        'red_dot_y': pd.Series(red_y),
        'blue_dot_x': pd.Series(blue_x),
        'blue_dot_y': pd.Series(blue_y),
    })

    csv_path = save_dir / f"{base_filename}_trajectories.csv"
    # 한글 경로/파일명 문제를 피하기 위해 encoding을 'utf-8-sig'로 지정
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Trajectories data saved to {csv_path}")


def mask_korean_name(name_string: str) -> str:
    """한국어 이름으로 추정되는 문자열을 찾아 중간 글자를 '*'로 바꾸기.
    Args:
        name_string (str): 파일명 등 마스킹할 문자열.
    Returns:
        str: 마스킹 처리된 문자열. 예: '1234홍길동_test' -> '1234홍*동_test'
    """
    # 2~4글자의 한글 문자열을 이름으로 간주
    match = re.search(r'[\uac00-\ud7a3]{2,4}', name_string)
    if match:
        name = match.group(0)
        # 이름이 3글자 이상일 경우에만 마스킹
        if len(name) >= 3:
            # 첫 글자와 마지막 글자를 제외한 중간 글자들을 '*' 하나로 바꾸기
            masked_name = f"{name[0]}*{name[-1]}"
            return name_string.replace(name, masked_name)
    return name_string

# 추가 끝!!


def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str,
                        default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT /
                        "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT /
                        "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+",
                        type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float,
                        default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float,
                        default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000,
                        help="maximum detections per image")
    parser.add_argument("--device", default="",
                        help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true",
                        help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true",
                        help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true",
                        help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true",
                        help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true",
                        help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int,
                        help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true",
                        help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true",
                        help="augmented inference")
    parser.add_argument("--visualize", action="store_true",
                        help="visualize features")
    parser.add_argument("--update", action="store_true",
                        help="update all models")
    parser.add_argument("--project", default=ROOT /
                        "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp",
                        help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true",
                        help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3,
                        type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False,
                        action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False,
                        action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true",
                        help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true",
                        help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1,
                        help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
    check_requirements(ROOT / "requirements.txt",
                       exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
