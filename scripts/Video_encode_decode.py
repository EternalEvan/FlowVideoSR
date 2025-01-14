import os
import imageio

# 设置输入和输出路径
# input_folder = 'path/to/frames'  # 图像帧所在的文件夹
input_folder = '/mnt/nfs/train_REDS4/train_sharp/000'  # 图像帧所在的文件夹
output_video = 'video_encoded.mp4'  # 输出视频文件名

# 获取所有帧文件
frame_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.png')])

# 确定帧率
fps = 30  # 可以根据需求调整

# 使用 imageio 写入视频
with imageio.get_writer(output_video, fps=fps, codec='libx264', quality=10, ffmpeg_params=['-crf', '0']) as writer:
# with imageio.get_writer(output_video, fps=fps, codec='ffv1') as writer:
    for frame_file in frame_files:
        image = imageio.imread(frame_file)
        writer.append_data(image)

print(f'视频已生成: {output_video}')

# import imageio

# 从视频中提取帧
input_video = 'video_encoded.mp4'
output_folder = 'decoded_frames'  # 提取帧的输出文件夹

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 使用 imageio 读取视频并提取每一帧
video_reader = imageio.get_reader(input_video)

for i, frame in enumerate(video_reader):
    frame_filename = os.path.join(output_folder, f'frame_{i:03d}.png')
    imageio.imwrite(frame_filename, frame)

print(f'帧已提取到: {output_folder}')


# from PIL import Image
# import os

# def images_are_equal(img1_path, img2_path):
#     img1 = Image.open(img1_path)
#     img2 = Image.open(img2_path)
#     return list(img1.getdata()) == list(img2.getdata())

# def calculate_similarity(img1_path, img2_path):
#     img1 = Image.open(img1_path)
#     img2 = Image.open(img2_path)

#     # 确保图片尺寸相同
#     if img1.size != img2.size:
#         raise ValueError("Images must be of the same size.")

#     # 将图片转换为 RGB 模式
#     img1 = img1.convert('RGB')
#     img2 = img2.convert('RGB')

#     # 获取像素数据
#     pixels1 = list(img1.getdata())
#     pixels2 = list(img2.getdata())

#     # 计算相同像素的数量
#     same_pixel_count = sum(p1 == p2 for p1, p2 in zip(pixels1, pixels2))
#     total_pixel_count = len(pixels1)

#     # 计算相同比例
#     similarity_ratio = same_pixel_count / total_pixel_count
#     return similarity_ratio


# folder1 = '/home/whl/workspace/Open-Sora/decoded_frames'
# folder2 = '/mnt/nfs/train_REDS4/train_sharp/000'

# # print(f'os.listdir(folder1): {os.listdir(folder1)}')
# # print(f'os.listdir(folder2): {os.listdir(folder2)}')

# for filename1 in sorted(os.listdir(folder1)):
#     for filename2 in sorted(os.listdir(folder2)):
#         img1_path = os.path.join(folder1, filename1)
#         img2_path = os.path.join(folder2, filename2)
#         if images_are_equal(img1_path, img2_path):
#             # import pdb; pdb.set_trace()
#             print(f"{filename1, filename2} is the same in both folders.")
#         else:
#             print(f"{filename1, filename2} differs between folders.")
#             print(f'similarity_ratio: {calculate_similarity(img1_path, img2_path)}')
#             # import pdb; pdb.set_trace()