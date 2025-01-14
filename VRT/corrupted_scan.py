import os
import cv2

def find_corrupted_images(root_dir):
    corrupted_files = []

    # 遍历根目录下的所有文件和子文件夹
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            
            # 只处理图像文件（这里假设图像文件的扩展名为常见格式）
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                try:
                    # 尝试读取图像
                    img = cv2.imread(file_path)
                    if img is None:
                        raise ValueError(f"Failed to load image: {file_path}")
                except Exception as e:
                    # 记录损坏的文件
                    print(f"Corrupted image found: {file_path}, Error: {e}")
                    corrupted_files.append(file_path)

    return corrupted_files

# 使用示例
root_directory = '/home/whl/workspace/Open-Sora/VRT/REDS_VRT/001_VRT_videosr_bi_REDS_6frames'
corrupted_images = find_corrupted_images(root_directory)

if corrupted_images:
    print("\nList of corrupted images:")
    for corrupted_image in corrupted_images:
        print(corrupted_image)
else:
    print("No corrupted images found.")