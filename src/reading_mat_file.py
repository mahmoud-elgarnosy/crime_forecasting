import ffmpeg
from torchvision.io import read_video, write_video
frames, s, l = read_video('../data/raw/UBI_FIGHTS/UBI_FIGHTS/videos/fight/F_2_1_2_0_0.mp4',pts_unit='sec')
print(frames.shape, s, l)
#
# ffmpeg \
#     .input('../data/interim/train/Abuse/Abuse002_x264_3_8.mp4') \
#     .filter('fps', fps=10, round='up') \
#     .output('../data/processed/Abuse3.mp4') \
#     .run()