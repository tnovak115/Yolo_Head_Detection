import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from torchvision import transforms
from detecto import core
import xml.etree.ElementTree as ET
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# YOLO Detection and Crop Functionality
def detect_one_image(model, image_path, margin=10):
    im2 = cv2.imread(image_path)
    if im2 is None:
        print(f"Error: Could not read the image file {image_path}. Skipping this file.")
        return None, None, None, None

    results = model.predict(source=im2, classes=[0])  # Detect "person" category (class 0)

    xtl_all, ytl_all = im2.shape[1], im2.shape[0]
    xbr_all, ybr_all = 0, 0

    print("YOLO Detection Results Structure:")
    yolo_boxes = []
    for i, result in enumerate(results):
        print(f"Result {i}:")
        print(f"  Number of boxes: {len(result.boxes)}")
        for j, bbox in enumerate(result.boxes):
            xtl, ytl, xbr, ybr = bbox.xyxy[0].tolist()
            confidence = bbox.conf.tolist()
            class_id = bbox.cls.tolist()
            print(f"    Box {j}: Coordinates: ({xtl}, {ytl}, {xbr}, {ybr}), Confidence: {confidence}, Class ID: {class_id}")

            xtl, ytl, xbr, ybr = int(xtl), int(ytl), int(xbr), int(ybr)
            xtl, ytl = max(0, xtl), max(0, ytl)
            xbr, ybr = min(im2.shape[1], xbr), min(im2.shape[0], ybr)

            xtl_all = min(xtl_all, xtl)
            ytl_all = min(ytl_all, ytl)
            xbr_all = max(xbr_all, xbr)
            ybr_all = max(ybr_all, ybr)

            yolo_boxes.append([xtl, ytl, xbr, ybr])

    if xtl_all >= xbr_all or ytl_all >= ybr_all:
        print(f"Invalid bounding box for image {image_path}. Skipping...")
        return None, None, None, None

    xtl_all = max(0, xtl_all - margin)
    ytl_all = max(0, ytl_all - margin)
    xbr_all = min(im2.shape[1], xbr_all + margin)
    ybr_all = min(im2.shape[0], ybr_all + margin)

    cropped_image = im2[ytl_all:ybr_all, xtl_all:xbr_all]
    return im2, cropped_image, (xtl_all, ytl_all, xbr_all, ybr_all), yolo_boxes

# Detect heads in cropped images
def detect_head_positions(model, cropped_image):
    transf = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage()])
    img_inp = transf(cropped_image)
    pred = model.predict(img_inp)
    return pred

# Filter head detection results
def filter_boxes(pred, b_prob=0.65):
    prob = pred[2]
    pred_boxes = pred[1]
    boxes = []
    for box, prob in zip(pred_boxes, prob):
        if prob.numpy() > b_prob:
            boxes.append(box.numpy())
    return np.array(boxes)


# 绘制并生成 CVAT 格式输出的函数
def draw_boxes_on_original_image(original_image, result_boxes, crop_box, yolo_boxes, save_dir, img_name, save_intermediate=True):
    xtl_all, ytl_all = crop_box[:2]
    filtered_image = original_image.copy()
    person_image = original_image.copy()  # 创建单独的行人检测图像

    valid_heads = []  # 保存所有有效头部的框
    head_to_person_mapping = []  # 记录 head 到 person 的映射

    # 遍历所有 `head` 框，检查和匹配的 `person` 框
    for idx, box in enumerate(result_boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        x_min_orig = x_min + crop_box[0]
        y_min_orig = y_min + crop_box[1]
        x_max_orig = x_max + crop_box[0]
        y_max_orig = y_max + crop_box[1]

        head_center_x = (x_min_orig + x_max_orig) / 2
        head_center_y = (y_min_orig + y_max_orig) / 2
        valid_head = False
        linked_person_idx = None  # 用于记录当前头部框对应的行人编号

        print(f"\nChecking Head Box {idx+1}: ({x_min_orig}, {y_min_orig}, {x_max_orig}, {y_max_orig}), Head Center Y: {head_center_y}")

        # 遍历所有 `person` 框，检查 `head` 是否符合要求
        for person_idx, person_box in enumerate(yolo_boxes):
            person_xmin, person_ymin, person_xmax, person_ymax = person_box
            print(f"  Against Person Box {person_idx+1}: ({person_xmin}, {person_ymin}, {person_xmax}, {person_ymax})")

            # 检查 head 的 x_min 和 x_max 是否在 person 的 x 范围内
            if person_xmin <= x_min_orig <= person_xmax and person_xmin <= x_max_orig <= person_xmax:
                print(f"    Match found! Head Box {idx+1} is within Person {person_idx+1}'s horizontal range [{person_xmin}, {person_xmax}].")
                valid_head = True
                linked_person_idx = person_idx
                break

        if valid_head:
            print(f"Head {idx+1} is valid and will be drawn.")
            valid_heads.append(box)  # 记录有效的 head
            head_to_person_mapping.append((len(valid_heads) - 1, linked_person_idx, y_min_orig))  # 记录 head 索引，person 索引及 y_min
        else:
            print(f"Head {idx+1} is not linked to any person and will be removed.")

    # 使用字典来记录每个 `person` 的头部框，并去重选择最高的头部框
    person_head_dict = {}
    for head_idx, person_idx, head_y_min in head_to_person_mapping:
        if person_idx not in person_head_dict or head_y_min < person_head_dict[person_idx][1]:
            # 如果此 person 还没有分配头部框，或当前头部框位置更高（y_min 更小）
            person_head_dict[person_idx] = (head_idx, head_y_min)

    # 使用过滤后的最高头部框进行绘制
    selected_heads = [valid_heads[person_head_dict[person_idx][0]] for person_idx in person_head_dict]

    # 绘制去重后的头部框
    selected_boxes = []  # 存储被选中的 `head` 框，用于后续生成 CVAT 格式
    for idx, box in enumerate(selected_heads):
        x_min, y_min, x_max, y_max = map(int, box)
        x_min_orig = x_min + crop_box[0]
        y_min_orig = y_min + crop_box[1]
        x_max_orig = x_max + crop_box[0]
        y_max_orig = y_max + crop_box[1]

        center_x = int((x_min_orig + x_max_orig) / 2)
        center_y = int((y_min_orig + y_max_orig) / 2)
        radius = max((x_max_orig-x_min_orig),(y_max_orig-y_min_orig))
        print(f"Drawing Head {idx+1} on original image: ({x_min_orig}, {y_min_orig}, {x_max_orig}, {y_max_orig})")

        cv2.circle(filtered_image, (center_x, center_y), radius, (0,255,0), 2)
        #cv2.rectangle(filtered_image, (x_min_orig, y_min_orig), (x_max_orig, y_max_orig), (0, 255, 0), 2)
        cv2.putText(filtered_image, f"Head {idx+1} (Selected)", (x_min_orig, y_min_orig - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        selected_boxes.append([x_min_orig, y_min_orig, x_max_orig, y_max_orig])  # 存储最终选择的 head 框

    # 保存过滤后的结果图像
    filtered_image_path = os.path.join(save_dir, f"{img_name.split('.')[0]}_head_selected.jpg")
    cv2.imwrite(filtered_image_path, filtered_image)
    print(f"Saved final image with selected heads as: {filtered_image_path}")

    # 生成 CVAT 格式的 XML 文件
    cvat_export(selected_boxes, img_name, original_image.shape[1], original_image.shape[0], save_dir)


# 将结果导出为 CVAT 格式的函数
def cvat_export(boxes, image_name, image_width, image_height, output_dir):
    # 创建 XML 根节点
    annotation = ET.Element("annotations")
    meta = ET.SubElement(annotation, "meta")

    # 构建 `meta` 信息
    job = ET.SubElement(meta, "job")
    ET.SubElement(job, "id").text = "1"
    ET.SubElement(job, "size").text = "1"
    ET.SubElement(job, "mode").text = "annotation"
    ET.SubElement(job, "overlap").text = "0"
    ET.SubElement(job, "start_frame").text = "0"
    ET.SubElement(job, "stop_frame").text = "0"

    labels = ET.SubElement(meta, "labels")
    label = ET.SubElement(labels, "label")
    ET.SubElement(label, "name").text = "head"
    ET.SubElement(label, "color").text = "#FF0000"
    ET.SubElement(label, "type").text = "rectangle"

    # 构建 `image` 节点
    image_elem = ET.SubElement(annotation, "image", {
        "id": "0",
        "name": image_name,
        "width": str(image_width),
        "height": str(image_height)
    })

    # 将每个 `box` 添加为 `image` 的子节点
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        ET.SubElement(image_elem, "box", {
            "label": "head",
            "source": "manual",
            "occluded": "0",
            "xtl": str(x_min),
            "ytl": str(y_min),
            "xbr": str(x_max),
            "ybr": str(y_max),
            "z_order": "0"
        })

    # 保存 XML 到指定路径
    tree = ET.ElementTree(annotation)
    output_path = os.path.join(output_dir, f"{image_name.split('.')[0]}_cvat.xml")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Saved CVAT annotation as: {output_path}")

# Main function
def main():
    print('Starting detection and annotation...')
    all_file_path = "./test_data"
    cropped_output_dir = "./cropped_images5"
    annotated_output_dir = "./annotated_results5"

    yolo_model = YOLO("./yolov8l.pt")
    head_model = core.Model.load('./head_detection.model', ['person'])

    os.makedirs(cropped_output_dir, exist_ok=True)
    os.makedirs(annotated_output_dir, exist_ok=True)

    for folder_name in os.listdir(all_file_path):
        folder_path = os.path.join(all_file_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for img_name in os.listdir(folder_path):
            if img_name.endswith(".jpeg") or img_name.endswith(".jpg") or img_name.endswith(".png"):
                img_path = os.path.join(folder_path, img_name)
                print(f"Processing image: {img_path}")

                original_image, cropped_image, crop_box, yolo_boxes = detect_one_image(yolo_model, img_path)
                if cropped_image is None or original_image is None:
                    continue

                cropped_image_path = os.path.join(cropped_output_dir, f"{img_name.split('.')[0]}_cropped.jpg")
                cv2.imwrite(cropped_image_path, cropped_image)

                head_positions = detect_head_positions(head_model, cropped_image)
                filtered_boxes = filter_boxes(head_positions, b_prob=0.5)

                # Draw boxes on the original image with enhanced debugging
                draw_boxes_on_original_image(original_image, filtered_boxes, crop_box, yolo_boxes, annotated_output_dir, img_name)

if __name__ == "__main__":
    main()
