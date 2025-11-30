import os
import json
import cv2
from tqdm import tqdm

# ================= 你的配置区域 =================
# 1. 数据集根目录 (AutoDL 路径)
ROOT_DIR = '/root/autodl-tmp/RT-DETR-main/image_yolo'

# 2. 需要处理的子集 (现在包含了 'test')
PHASES = ['train', 'val', 'test']

# 3. 类别名称 (0->body, 1->head, 2->leg)
CLASSES = ['body', 'head', 'leg']

# 4. 图片扩展名支持
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}


# ===============================================

def yolo_to_coco(root_path, phases, classes):
    print(f"开始转换，数据集路径: {root_path}")
    print(f"处理子集: {phases}")
    print(f"类别定义: {classes}")

    for phase in phases:
        print(f"\n----------------------------------------")
        print(f"正在处理: {phase} 集...")

        # 定义输入路径
        img_dir = os.path.join(root_path, 'images', phase)
        label_dir = os.path.join(root_path, 'labels', phase)

        # 定义输出 JSON 路径
        output_json = os.path.join(root_path, f'instances_{phase}.json')

        # 检查目录是否存在
        if not os.path.exists(img_dir):
            print(f"[警告] 跳过 {phase}, 图片目录不存在: {img_dir}")
            continue

        # 检查标签目录 (Test集有时可能没有标签，这很正常)
        if not os.path.exists(label_dir):
            print(f"[提示] 标签目录不存在: {label_dir} (如果是纯推理预测，这是正常的)")

        # 初始化 COCO JSON 结构
        dataset = {
            "info": {"description": "Converted from YOLO format"},
            "licenses": [],
            "categories": [],
            "images": [],
            "annotations": []
        }

        # 1. 构建 Categories
        for i, cls_name in enumerate(classes):
            dataset['categories'].append({
                "id": i,
                "name": cls_name,
                "supercategory": "cattle_parts"
            })

        # 获取所有图片文件
        img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]
        print(f"找到 {len(img_files)} 张图片")

        annotation_id = 0

        # 使用 tqdm 显示进度条
        for img_id, img_file in enumerate(tqdm(img_files)):
            # ---------------- 图片处理 ----------------
            image_path = os.path.join(img_dir, img_file)

            # 读取图片以获取宽高
            img = cv2.imread(image_path)
            if img is None:
                print(f"[错误] 无法读取图片: {image_path}")
                continue

            height, width, _ = img.shape

            dataset['images'].append({
                "id": img_id,
                "file_name": img_file,
                "width": width,
                "height": height
            })

            # ---------------- 标签处理 ----------------
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(label_dir, label_file)

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    # 坐标转换: YOLO -> COCO
                    abs_w = w * width
                    abs_h = h * height
                    abs_cx = x_center * width
                    abs_cy = y_center * height

                    abs_x = abs_cx - (abs_w / 2)
                    abs_y = abs_cy - (abs_h / 2)

                    dataset['annotations'].append({
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": class_id,
                        "bbox": [abs_x, abs_y, abs_w, abs_h],
                        "area": abs_w * abs_h,
                        "iscrowd": 0
                    })
                    annotation_id += 1

        # 保存 JSON 文件
        with open(output_json, 'w') as f:
            json.dump(dataset, f)
        print(f"✅ 已保存: {output_json}")


if __name__ == "__main__":
    yolo_to_coco(ROOT_DIR, PHASES, CLASSES)
    print("\n所有转换完成！")