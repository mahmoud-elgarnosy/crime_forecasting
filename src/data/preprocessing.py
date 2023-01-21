# import ffmpeg
from torchvision.io import read_video, write_video
from torchaudio.io import StreamReader
import numpy as np
# from PIL import Image
from PIL import Image, ImageFilter


frames, s, l = read_video('../../data/interim/train/Abuse/Abuse006_x264_10_15.mp4', pts_unit='sec')
result = np.median(frames, axis=0).astype(np.uint8)
Image.fromarray(result).save('result.png')


#
# print(frames.shape, s, l)
# #
# #
# # #
# # # ffmpeg \
# # #     .input('../data/interim/train/Abuse/Abuse002_x264_3_8.mp4') \
# # #     .filter('fps', fps=10, round='down') \
# # #     .output('../data/processed/Abuse3.mp4') \
# # #     .run()
# #
# class Preprocessing:
#     def __init__(self, video_path, fps=15):
#         self.video_tensor = video_path
#         self.fps = fps
#         self.streamer = StreamReader('rtmp://localhost/live/stream')
#         self.lowering_fps()
#
#     def lowering_fps(self):
#         # Video stream with 320x320 (stretched) at 3 FPS, grayscale
#         self.streamer.add_basic_video_stream(
#             frames_per_chunk=15,
#             frame_rate=15,
#             width=128,
#             height=128,
#             format="bgr24"
#         )
#
#
# if __name__ == '__main__':
#     print('k')
#     preprocess = Preprocessing(fps=15, video_path='../../data/interim/train/Abuse/Abuse005_x264_24_29.mp4')
#     streamer = preprocess.streamer
#     n_ite = 10
#     waveforms, vids1, vids2 = [], [], []
#     for i, waveform in enumerate(streamer.stream()):
#         print(waveform[0].shape)
#         waveforms.append(waveform)
#         print(len(waveforms))
#         if i + 1 == n_ite:
#             break
