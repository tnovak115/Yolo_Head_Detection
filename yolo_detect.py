# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
import cv2
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import os 

def single_image_detection():
    YOLO("yolov8x.pt")  # 下载 yolov8x.pt 模型
    model = YOLO("./yolov8x.pt")
    index = 1
    image_path = "../../../../../data/v2x-real_old/train/2023-04-04-15-42-18_14_0/2/000105_cam1_left.jpeg"
    source_dir = "../../yolo_results_ver3"
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        print(f"Folder {source_dir} created.")
    else:
        print(f"Folder {source_dir} already exists.")   

    detect_one_image(model, index, image_path,source_dir,is_save=True)

def all_class_name():
    # Define a dictionary to map class indices to class names if necessary
    class_names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 
                   6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant", 
                   11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat", 
                   16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 
                   22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag", 
                   27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard", 
                   32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove", 
                   36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass", 
                   41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 
                   48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 
                   54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 
                   60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 
                   66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 
                   72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 
                   78: "hair drier", 79: "toothbrush"}
    return class_names

def batch_process():
    """
    对 images 目录中的所有图像进行批量处理，并将检测结果保存到 yolo_results 目录中。
    """

    # 输入图像目录路径
    all_file_path = "../../../../../data/v2x-real_old/train/2023-04-05-16-25-26_22_1"
    # 保存检测结果的根目录
    source_dir = "../../yolo_results2"

    # 如果目标目录不存在，则创建
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        print(f"Folder {source_dir} created.")
    else:
        print(f"Folder {source_dir} already exists.")

    # 加载 YOLO 模型
    model = YOLO("./yolov8l.pt")

    # 遍历 images 目录中的每个子文件夹（如 1, 2）
    directories = [d for d in os.listdir(all_file_path) if os.path.isdir(os.path.join(all_file_path, d))]
    
    # 遍历每个子目录
    for folder_name in directories:
        # 获取当前子文件夹路径（如 ./images/1 或 ./images/2）
        folder_path = os.path.join(all_file_path, folder_name)
        print(f"Processing folder: {folder_path}")

        # 目标保存目录（保持原目录结构）
        save_dir = os.path.join(source_dir, folder_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 遍历当前子文件夹中的所有图像文件
        for img_name in os.listdir(folder_path):
            if img_name.endswith(".jpeg") or img_name.endswith(".jpg") or img_name.endswith(".png"):
                img_path = os.path.join(folder_path, img_name)  # 图像的完整路径
                print(f"Processing image: {img_path}")
                
                # 检测并保存结果
                detect_one_image(model, img_name, img_path, save_dir, is_save=True)

def detect_one_image(model, index, image_path, source_dir, is_save=False, min_width=50, min_height=50, margin=10, min_vertical_dist=20, min_horizontal_dist=20):
    """
    对单张图像进行目标检测，并在所有目标位置基础上进行裁剪。
    增加判断裁剪出的图像是否过小，如果分辨率低于 min_width x min_height 或上下或左右的距离过小，则不保存。
    增加 margin 参数，裁剪区域比最大检测框稍微大一点。
    """
    # 读取图像文件
    im2 = cv2.imread(image_path)
    if im2 is None:
        print(f"Error: Could not read the image file {image_path}. Skipping this file.")
        return

    # 使用 YOLO 模型进行目标检测
    results = model.predict(source=im2, classes=[0])  # 检测 "person" 类别（类别 0）

    # 提取原始图像文件名（如 000000_cam1_left.jpeg）
    image_name = os.path.basename(image_path)
    save_image_path = os.path.join(source_dir, image_name)

    # 保存检测后的图像
    if is_save:
        # 用于存储所有人的边界框坐标
        all_boxes = []

        # 遍历所有检测到的结果
        for result in results:
            for bbox in result.boxes:
                xtl, ytl, xbr, ybr = bbox.xyxy[0].tolist()  # 获取边界框的坐标
                # 转换为整数，并确保边界框在图像范围内
                xtl, ytl, xbr, ybr = int(xtl), int(ytl), int(xbr), int(ybr)
                xtl, ytl = max(0, xtl), max(0, ytl)  # 保证坐标不小于 0
                xbr = min(im2.shape[1], xbr)  # 保证 xbr 不超过图像宽度
                ybr = min(im2.shape[0], ybr)  # 保证 ybr 不超过图像高度

                # 将所有边界框添加到列表中
                all_boxes.append([xtl, ytl, xbr, ybr])

        # 如果没有检测到任何目标，则返回
        if not all_boxes:
            print(f"No person detected in {image_name}. Skipping cropping.")
            return

        # 计算所有目标的最小外接矩形（得到包含所有人的最小裁剪框）
        xtl_all = min([box[0] for box in all_boxes])  # 最左边界
        ytl_all = min([box[1] for box in all_boxes])  # 最上边界
        xbr_all = max([box[2] for box in all_boxes])  # 最右边界
        ybr_all = max([box[3] for box in all_boxes])  # 最下边界

        # 添加 margin 进行裁剪边界扩展
        xtl_all = max(0, xtl_all - margin)  # 保证左边界不小于 0
        ytl_all = max(0, ytl_all - margin)  # 保证上边界不小于 0
        xbr_all = min(im2.shape[1], xbr_all + margin)  # 保证右边界不超过图像宽度
        ybr_all = min(im2.shape[0], ybr_all + margin)  # 保证下边界不超过图像高度

        # 计算裁剪后的宽度和高度
        cropped_width = xbr_all - xtl_all
        cropped_height = ybr_all - ytl_all

        # 检查裁剪框的左右和上下距离是否太小
        if cropped_width < min_horizontal_dist or cropped_height < min_vertical_dist:
            print(f"Skipped saving: {image_name}_cropped.jpg (Width: {cropped_width}, Height: {cropped_height}) - Distance too small!")
            return  # 跳过保存距离过小的图像

        # 裁剪包含所有人的最小外接矩形区域
        cropped_image = im2[ytl_all:ybr_all, xtl_all:xbr_all]
        height, width = cropped_image.shape[:2]

        # 检查裁剪图像是否过小，如果是则跳过保存
        if width < min_width or height < min_height:
            print(f"Skipped saving: {image_name}_cropped.jpg (resolution: {width}x{height}) - Too small!")
            return  # 跳过保存小于指定分辨率的图像

        # 保存裁剪后的图像
        cropped_image_path = os.path.join(source_dir, f"{image_name.split('.')[0]}_cropped.jpg")
        cv2.imwrite(cropped_image_path, cropped_image)
        print(f"Overall cropped image saved as: {cropped_image_path}")

        # 保存带有检测框的完整图像
        cv2.imwrite(save_image_path, im2)
        print(f"Result image saved as: {save_image_path}")




def batch_process_tqdm():
    ## source directory 
    source_dir = "/media/zzl/WD_BLACK/annotation_data/yolo_results"
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        print(f"Folder {source_dir} created.")
    else:
        print(f"Folder {source_dir} already exists.")   

    # Load the model
    model = YOLO("./yolov8l.pt")

    ################################################################
    all_file_path = "/media/zzl/WD_BLACK/annotation_data/training"
    entries = os.listdir(all_file_path)
    # Filter out directories
    directories = [entry for entry in entries if os.path.isdir(os.path.join(all_file_path, entry))]
    # Sort the directories numerically
    sorted_directories = sorted(directories, key=lambda x: int(x))

    # print(sorted_directories)

    output_file = "output.xml"
    run_subset_name = [698,740,761,805,849,850,870,952,1276]
    for run_name in tqdm(sorted_directories, desc="Processing directories"):

        if int(run_name) in run_subset_name:
        
            run_path_cam = os.path.join(all_file_path, run_name, "camera") # run_name = 91,1276,...
            if not os.listdir(run_path_cam):
                continue
            else:
                entries = os.listdir(run_path_cam)

            sorted_entries = sorted(entries, key=lambda x: int(x[2:]))

            for item in tqdm(sorted_entries, desc="Processing cameras", leave=False):
                vc_path = os.path.join(run_path_cam, item) # item = vc1,2,4,5,6
                if not os.listdir(vc_path):
                    print("skipping {}".format(vc_path))
                    continue 
                else:
                    img_entries = os.listdir(vc_path)

                # check if the vc_path is empty, if it's empty, then skip this folder

                sorted_img_filenames = sorted(img_entries, key=lambda x: x.split('.')[0])
                
                xml_content_all = ""
                xml_path = os.path.join(vc_path, output_file)
                
                for img_name in tqdm(sorted_img_filenames, desc="Processing images", leave=False):
                    img_name_path = os.path.join(vc_path, img_name)
                    # print(img_name_path)
                    xml_content = detect_one_image(model, img_name, img_name_path, source_dir, is_save=True)
                    xml_content_all += xml_content

                relative_path = xml_path.split(all_file_path)[-1]
                xml_path_final = source_dir + relative_path
                
                # print("xml_path:", xml_path_final)

                with open(xml_path_final, "w") as f:
                    f.write(xml_content_all)

                print(f"XML file saved as {xml_path_final}")

def test():

    # Load the model
    model = YOLO("./yolov8l.pt")
    image_name = "2024-02-05-14-57-03_281712.jpg"

    # Perform prediction using the model (replace with your image path)
    im2 = cv2.imread("./2024-02-05-14-57-03_281712.jpg")
    results = model.predict(source=im2)

    class_names = all_class_name()

    important_label = ["person", "car", "truck", "bus"]

    # Get image information
    image_id = 1
    height, width, _ = im2.shape

    # Create XML content
    xml_content = f'<image id="{image_id}" name="{image_name}" width="{width}" height="{height}">\n'

    # Draw bounding boxes and class names on the image
    for result in results:
        for bbox in result.boxes:
            # class_index = bbox.cls
            class_index = bbox.cls.tolist()
            # print(class_index)
            label = class_names.get(class_index[0], "unknown")
            xtl, ytl, xbr, ybr = bbox.xyxy[0].tolist()

            # Filter out the traffic sign and draw bounding boxes for other classes
            if label in important_label:
                # Append box element to XML content
                xml_content += f'  <box label="{label}" source="manual" occluded="0" '
                xml_content += f'xtl="{xtl:.2f}" ytl="{ytl:.2f}" xbr="{xbr:.2f}" ybr="{ybr:.2f}" z_order="0">\n'
                xml_content += f'  </box>\n'

                # Draw a green rectangle
                cv2.rectangle(im2, (int(xtl), int(ytl)), (int(xbr), int(ybr)), (0, 255, 0), 2)
                
                # Put the class name on the image
                cv2.putText(im2, str(label), (int(xtl), int(ytl) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    xml_content += '</image>'

    # Save the result image to another image file
    result_image_path = "./result_image2.jpg"
    cv2.imwrite(result_image_path, im2)
    print(f"Result image saved as {result_image_path}")
    
    # Save the XML content to a TXT file
    output_file = "output.xml"
    with open(output_file, "w") as f:
        f.write(xml_content)

    print(f"XML file saved as {output_file}")


if __name__ == "__main__":
    batch_process()
    # batch_process_tqdm()
    # test()
    # single_image_detection()