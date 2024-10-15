# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import cv2
import numpy as np
import matplotlib.pyplot as plt
# from yolo_det_agent import yolo_detector

class visualizer:
    def __init__(self,yolo_detector=None):

        if yolo_detector is not None:
            self.my_detector = yolo_detector
        else:
            self.my_detector = yolo_detector

    def visulize_multi_centers(self,imgs_path,bg_img,color=(0, 255, 0),\
                               vis_show=False,save_img=False,output_path=None):
        points_2d = []
        if self.my_detector is not None:
            #color = (0, 255, 0)  # green
            results = self.my_detector.multi_img_inference(imgs_path)

            for key,result in results.items():

                single_detection = result['bbx']

                for detection in single_detection:
                    center = detection['bbox']['center']
                    cx = center[0]
                    cy = center[1]
                    # center part
                    cv2.circle(bg_img, (int(cx), int(cy)), 3, color, -1)

                    points_2d.append([cx,cy])
            
            if vis_show:
                cv2.imshow("Detection Results", bg_img)
                cv2.waitKey(0)  # press any key to close window
                cv2.destroyAllWindows()

            if save_img:
                if output_path is not None:
                    cv2.imwrite(output_path, bg_img)
                else:
                    print("WARNING: No output path provided. Image not saved.")

        else:
            print("WARNING: no detector has been assigned!!!")

        points_2d_array = np.array(points_2d)
        return bg_img, points_2d_array
    
    def create_img_overlay_points(self, image, points_2d, points_cam, point_radius = 2, intensity=None,colors = [1,1,0]):
    
        if intensity is not None:
            # Define a colormap for intensity values
            colormap = plt.cm.jet

            # Normalize intensity values to range [0, 1] for colormap
            intensity_normalized = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))

            # Map normalized intensity values to colors using colormap
            colors = colormap(intensity_normalized)
        else:
            colors = colors
            
        overlay = np.zeros_like(image)
        
        # point_radius = 2  # Change this value to adjust point size
        
        # Plot projected points on the overlay image
        for i in range(points_2d.shape[0]):
            if points_cam[i, 2] > 0:  # Ensure points are in front of the camera
                
                if intensity is not None:
                    # Get color based on intensity value from colormap
                    color = (int(colors[i][0] * 255), int(colors[i][1] * 255), int(colors[i][2] * 255))  # Convert RGB components to 0-255 range
                else:
                    color = (int(colors[0] * 255), int(colors[1] * 255), int(colors[2] * 255))
                # Get coordinates of the point
                x, y = int(points_2d[i, 0]), int(points_2d[i, 1])
                
                # Check if the coordinates are within the image range before drawing
                if x >= 0 and y >= 0 and x < overlay.shape[1] and y < overlay.shape[0]:
                    # Draw circle on the overlay image with modified radius
                    cv2.circle(overlay, (x, y), point_radius, color, -1, lineType=cv2.LINE_AA)  # Change -1 to adjust circle's border thickness
        
        return overlay
    
    def add_overlay_img(self,image,overlay,alpha = 0.4):
        image_with_overlay = cv2.addWeighted(image, 1.0, overlay, alpha, 0)

        return image_with_overlay
    
    def img_resize_n_vis(self, mod_img, width, height, \
                         vis_show=False, save_img=False, output_file=None):
        # Resize the image
        resized_img = cv2.resize(mod_img, (width, height))

        # Show the resized image if vis_show is True
        if vis_show:
            cv2.imshow("Resized Image", resized_img)
            cv2.waitKey(0)  # Press any key to close the window
            cv2.destroyAllWindows()

        # Save the resized image if save_img is True and output_file is provided
        if save_img:
            if output_file is not None:
                cv2.imwrite(output_file, resized_img)
            else:
                print("WARNING: No output file provided. Image not saved.")

        return resized_img


if __name__ == "__main__":
    # model_path = "./yolo_weights/yolov8x.pt"
    # imgs_path = "/media/zzl/Extreme SSD/validation/images/example/214_vc4"
    # img_path = "/media/zzl/Extreme SSD/validation/images/example/214_vc4/2024-02-20-17-58-51_164318.jpg"
    # my_detector = yolo_detector(model_path)
    # my_vis = visualizer(my_detector)
    # im2 = cv2.imread(img_path)
    # my_vis.visulize_multi_centers(imgs_path,im2,vis_show=True)

    pass
