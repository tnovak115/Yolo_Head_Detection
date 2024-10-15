import os
import cv2
import matplotlib.pyplot as plt

def draw_boxes_on_original_image(original_image_path, result_file_path, output_dir, crop_box):
    """
    在原图上绘制检测框，并保存结果图像。
    :param original_image_path: 原始图像的路径
    :param result_file_path: 保存检测结果的 txt 文件路径
    :param output_dir: 绘制结果保存的目录
    :param crop_box: 裁剪区域在原图中的坐标 (xtl_all, ytl_all, xbr_all, ybr_all)
    """
    # 从裁剪框中获取偏移量
    xtl_all, ytl_all, xbr_all, ybr_all = crop_box

    # 读取原图
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        print(f"Error: Could not read the original image file {original_image_path}. Skipping this file.")
        return

    # 检查保存目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取检测结果
    circles = []
    with open(result_file_path, "r") as f:
        for line in f:
            # 假设每行格式为：x_min, y_min, x_max, y_max
            parts = line.strip().split(",")
            if len(parts) == 4:
                x_min, y_min, x_max, y_max = map(int, parts)
                # 将检测框映射回原图坐标
                x_min_orig = x_min + xtl_all
                y_min_orig = y_min + ytl_all
                x_max_orig = x_max + xtl_all
                y_max_orig = y_max + ytl_all
                #calculations for circle
                center_x = (x_min_orig + x_max_orig)/2
                center_y = (y_min_orig + y_max_orig)/2
                radius = max((x_max_orig-x_min_orig), (y_max_orig-y_min_orig))
                circles.append([center_x, center_y, radius])

    #make code for circle face detection here
    # 在原图上绘制所有检测框
    for (center_x, center_y, radius) in circles:
        cv2.circle(original_image, (center_x, center_y), radius, (0,255,0), 2)
        #cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # 绿色框

    # 保存标注后的图像
    output_image_path = os.path.join(output_dir, f"{os.path.basename(original_image_path).split('.')[0]}_annotated.jpg")
    cv2.imwrite(output_image_path, original_image)
    print(f"Annotated image saved as: {output_image_path}")

    # 使用 matplotlib 显示结果
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Annotated Image: {os.path.basename(original_image_path)}")
    plt.show()


# 示例使用
original_image_folder = "../../yolo_results2/1"  # 原图文件夹
result_folder = "../../yolo_results2/1"  # 存放裁剪结果的文件夹
output_folder = "./annotated_results"  # 保存标注后的图像文件夹

# 遍历所有结果文件
for result_file in os.listdir(result_folder):
    if result_file.endswith("_cropped_result.txt"):
        # 解析原图文件名
        original_image_name = result_file.replace("_cropped_result.txt", ".jpeg")  # 假设原图是 .jpeg 格式
        original_image_path = os.path.join(original_image_folder, original_image_name)

        # 确定当前裁剪框的起始位置（这里可以根据实际情况填入）
        # 例如，假设 crop_box = (xtl_all, ytl_all, xbr_all, ybr_all)
        crop_box = (50, 50, 300, 300)  # 需要根据实际的裁剪框填入

        # 读取结果文件路径
        result_file_path = os.path.join(result_folder, result_file)

        # 在原图上绘制框并保存
        draw_boxes_on_original_image(original_image_path, result_file_path, output_folder, crop_box)
