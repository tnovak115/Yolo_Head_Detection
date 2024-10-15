# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import cv2
import os
import sys

from ultralytics import YOLO

# get parent dir of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# add to system path 
sys.path.insert(0, parent_dir)
from yolo_detect.utils import camera_utils

class yolo_detector:
    def __init__(self, model_path, class_names=camera_utils.CLASS_NAME):
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.important_labels = camera_utils.IMPORTANT_LABEL
        self.activate_imp_labels = True
        
    def single_img_inferernce(self, image_path,vis_show=False):
        im2 = cv2.imread(image_path)
        results = self.model.predict(source=im2)
        
        height, width, _ = im2.shape

        yolo_det_results = []
        
        for result in results:
            for bbox in result.boxes:
                class_index = bbox.cls.tolist()
                label = self.class_names.get(class_index[0], "unknown")
                confidence = bbox.conf.tolist()[0]
                xtl, ytl, xbr, ybr = bbox.xyxy[0].tolist()
                
                cx = (xtl + xbr) / 2
                cy = (ytl + ybr) / 2
                
                if self.activate_imp_labels:
                    if label in self.important_labels:
                        detection = {
                            "label": label,
                            "confidence": confidence,
                            "bbox": {
                                "xtl": xtl,
                                "ytl": ytl,
                                "xbr": xbr,
                                "ybr": ybr,
                                "center": (cx,cy)
                            }
                        }
                        # visualization part 
                        if vis_show:
                            color = (0, 255, 0)  # green
                            cv2.rectangle(im2, (int(xtl), int(ytl)), (int(xbr), int(ybr)), color, 2)
                            cv2.putText(im2, f"{label} {confidence:.2f}", (int(xtl), int(ytl) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            # center part
                            cv2.circle(im2, (int(cx), int(cy)), 3, color, -1)
                else:
                    detection = {
                        "label": label,
                        "confidence": confidence,
                        "bbox": {
                            "xtl": xtl,
                            "ytl": ytl,
                            "xbr": xbr,
                            "ybr": ybr,
                            "center": (cx,cy)
                        }
                    }

                yolo_det_results.append(detection)

        # show visualization
        if vis_show:
            cv2.imshow("Detection Results", im2)
            cv2.waitKey(0)  # press any key to close window
            cv2.destroyAllWindows()

        return height, width, yolo_det_results

    def single_img_inferernce_mul_results(self, image_path):
        im2 = cv2.imread(image_path)
        results = self.model.predict(source=im2, conf=0.1)  # Lower the confidence threshold
        
        height, width, _ = im2.shape

        yolo_det_results = []
        
        for result in results:
            for bbox in result.boxes:
                class_scores = bbox.conf.tolist()
                class_indices = bbox.cls.tolist()
                xtl, ytl, xbr, ybr = bbox.xyxy[0].tolist()
                
                # Collect all classes with their confidence scores
                detections = []
                for idx, score in zip(class_indices, class_scores):
                    label = self.class_names.get(idx, "unknown")
                    detections.append({
                        "label": label,
                        "confidence": score
                    })

                # Sort detections by confidence in descending order
                detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

                yolo_det_results.append({
                    "bbox": {
                        "xtl": xtl,
                        "ytl": ytl,
                        "xbr": xbr,
                        "ybr": ybr
                    },
                    "detections": detections
                })

        return height, width, yolo_det_results
    
    def multi_img_inference(self, imgs_path):

        img_names = os.listdir(imgs_path)

        results = {}

        for idx, img in enumerate(img_names):
            img_path = os.path.join(imgs_path,img)

            results[img_path] = {}
            results[img_path]['id'] = idx
            height, width, yolo_det_results = self.single_img_inferernce(img_path)
            results[img_path]['height'] = height
            results[img_path]['width'] = width
            results[img_path]['bbx'] = yolo_det_results

        return results

    
if __name__ == "__main__":
    model_path = "C:\Users\trevo\Downloads\yolov8l.pt"
    imgs_path = "C:\Users\trevo\Downloads\face_blur\Yolo_Head_Detection\test_data"
    img_path = "C:\Users\trevo\Downloads\face_blur\Yolo_Head_Detection\test_data\1\000000_cam1_left.jpeg"
    my_detector = yolo_detector(model_path)
    height, width, yolo_det_results = my_detector.single_img_inferernce(img_path,vis_show=True)
    print(len(yolo_det_results))
    print(yolo_det_results)

