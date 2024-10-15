# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
import os
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
from boxmot import BoTSORT
from ultralytics import YOLO
from pathlib import Path
import torch
from utils import camera_utils

class yolo_det_tracker():
    def __init__(self, img_path,camera_type):
        self.img_path = img_path
        self.camera_type = camera_type
        self.max_num_objects = 4

        self.model = self._load_yolo_model()

    def _load_yolo_model(self):

        return YOLO("C:\Users\trevo\Downloads\yolov8l.pt")

    def _initialize_tracker(self):
        self.frame_width, self.frame_height = camera_utils.CAMERA_MAP_WH[self.camera_type]
        self.tracker = BoTSORT(
            model_weights = Path('osnet_x0_25_msmt17.pt'),  # which ReID model to use
            device = 'cuda:0',
            fp16 = False,
            # det_thresh = DETECTION_THRESHOLD,
            # per_class=per_class,
            # track_high_thresh=0.80,
            # track_low_thresh=0.01,
            # new_track_thresh=cfg.new_track_thresh,
            # track_buffer=cfg.track_buffer,
            # match_thresh=cfg.match_thresh,
            # proximity_thresh=cfg.proximity_thresh,
            # appearance_thresh=DETECTION_THRESHOLD,
            # cmc_method=cfg.cmc_method,
            # frame_rate=cfg.frame_rate,`   
        )
        self.tracker.det_threshold = camera_utils.DETECTION_THRESHOLD
        self.tracker.track_high_thresh = 0.3
        self.tracker.track_low_thresh = camera_utils.DETECTION_THRESHOLD

if __name__ == "__main__":
    pass
