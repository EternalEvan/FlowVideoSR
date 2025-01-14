import numpy as np
import av
import io
import random
import logging
import cv2 
from moviepy.editor import VideoFileClip
 
# 确保 av 库可用
has_av = True
def read_video_and_resize(path, frame_size=(256, 256)):
    """
    读取MP4视频文件并将其帧调整为指定尺寸。

    Args:
        path (str): 视频文件路径
        frame_size (tuple): 输出帧的尺寸，默认为(256, 256)

    Returns:
        list: 包含调整后帧的列表，每帧为一个NumPy数组
    """
    cap = cv2.VideoCapture(path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame
        # resized_frame = cv2.resize(frame, frame_size)
        frames.append(frame)

    cap.release()
    print(f'frames[0].shape = {frames[0].shape}')
    return frames


class RandomVideoCompression:
    """Apply random video compression to the input.

    Modified keys are the attributes specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        assert has_av, 'Please install av to use video compression.'

        self.keys = keys
        self.params = params
        logging.getLogger('libav').setLevel(50)

    def _apply_random_compression(self, imgs):
        """This is the function to apply random compression on images.

        Args:
            imgs: list of ndarray: Training images, each image is a numpy array.

        Returns:
            list of ndarray: Images after randomly compressed.
        """
        codec = random.choices(self.params['codec'],
                               self.params['codec_prob'])[0]
        bitrate = self.params['bitrate']
        bitrate = np.random.randint(bitrate[0], bitrate[1] + 1)

        buf = io.BytesIO()
        with av.open(buf, 'w', 'mp4') as container:
            stream = container.add_stream(codec, rate=1)
            stream.height = imgs[0].shape[0]
            stream.width = imgs[0].shape[1]
            stream.pix_fmt = 'yuv420p'
            stream.bit_rate = bitrate

            for img in imgs:
                img = img.astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                frame.pict_type = 'NONE'
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Flush stream
            for packet in stream.encode():
                container.mux(packet)

        outputs = []
        with av.open(buf, 'r', 'mp4') as container:
            if container.streams.video:
                for frame in container.decode(**{'video': 0}):
                    outputs.append(frame.to_rgb().to_ndarray().astype(
                        np.float32))

        return outputs

    def __call__(self, results):
        """Call this transform."""
        if np.random.uniform() > self.params.get('prob', 1):
            return results

        for key in self.keys:
            results[key] = self._apply_random_compression(results[key])

        return results

    def __repr__(self):
        """Print the basic information of the transform."""
        repr_str = self.__class__.__name__
        repr_str += (f'(params={self.params}, keys={self.keys})')
        return repr_str

# video_data = {
#     'lq': [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(10)]
# }
# print(video_data['lq'][0].shape, len(video_data['lq']))
# 配置随机压缩参数
params = dict(
    codec=['libx264', 'h264', 'mpeg4'],
    codec_prob=[1 / 3., 1 / 3., 1 / 3.],
    bitrate=[1e4, 1e5],
    prob=1  # 这里将概率设置为1，确保每次都应用压缩
)
keys = ['lq']


# 示例视频文件路径，确保你有一个有效的 MP4 文件路径
# video_path = '/home/jingxuan/RealBasicVSR/data/demo_001.mp4'
video_path = '/mnt/nfs/YouHQ40-Test-Video/000/000.mp4'

# 读取视频并调整大小
frames = read_video_and_resize(video_path)

video= [np.array(frame, dtype=np.uint8) for frame in frames]
print(f'video[0].shape = {video[0].shape}')
print(f'video[0] = {video[0]}')

# if len(video) > 0:
#     print(f"Frame shape: {video[0].shape}")

# 打印视频帧的数量
print(f"Total frames: {len(video)}")

video_data = {
    'lq': video
}
# # 如果需要获取一个具有指定范围（0, 256）的索引
# random_frame_index = random.randint(0, len(video_data) - 1)
# random_frame = video_data[random_frame_index]
# print(f"Random frame index: {random_frame_index}")
# print(f"Random frame shape: {random_frame.shape}")

# 创建RandomVideoCompression实例
random_compression = RandomVideoCompression(params, keys)

# 打印变换信息
print(random_compression)

# 应用随机视频压缩
compressed_video_data = random_compression(video_data)

print(f"Original frames: {len(video_data['lq'])}")
print(f"Compressed frames: {len(compressed_video_data['lq'])}")

frames = compressed_video_data['lq']

height, width, channels = frames[0].shape

# 定义输出视频文件的编码器和文件名
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编码器
output_file = '/home/whl/workspace/Open-Sora/video_compression_output.mp4'

out = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))

for frame in frames:
    frame = frame.astype(np.uint8)
    print(f'frame.shape = {frame.shape}')
    out.write(frame)

out.release()

print(f"视频已保存为 {output_file}")