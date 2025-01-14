import os
import cv2

def images_to_video(img_folder, video_path, fps=30, frame_size=None):
    # 获取图片列表，按顺序排序
    images = sorted([img for img in os.listdir(img_folder) if img.endswith('.png')])
    
    # 读取第一张图片以获取帧大小
    first_image = cv2.imread(os.path.join(img_folder, images[0]))
    height, width, _ = first_image.shape
    
    # 如果指定了frame_size，进行大小调整
    if frame_size:
        width, height = frame_size
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4格式
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for image in images:
        img_path = os.path.join(img_folder, image)
        frame = cv2.imread(img_path)
        if frame_size:  # 如果指定了frame_size，调整图片大小
            frame = cv2.resize(frame, (width, height))
        video_writer.write(frame)  # 写入视频帧
    
    video_writer.release()  # 释放视频写入器
    print(f"Video saved to {video_path}")

# 定义根目录路径
root_folder = '/mnt/nfs/YouHQ40-Test'
output_folder = '/mnt/nfs/YouHQ40-Test-Video'

# 创建输出目录
os.makedirs(output_folder, exist_ok=True)

# 遍历每个子文件夹，生成对应视频并保存在对应的子文件夹中
for i in range(40):
    sub_folder = os.path.join(root_folder, f'{i:03d}')  # 例如 '000', '001', ..., '039'
    
    # 创建对应的输出子文件夹
    output_sub_folder = os.path.join(output_folder, f'{i:03d}')
    os.makedirs(output_sub_folder, exist_ok=True)
    
    # 定义视频输出路径
    video_output_path = os.path.join(output_sub_folder, f'{i:03d}.mp4')
    
    # 将图片生成视频
    images_to_video(sub_folder, video_output_path, fps=30)
