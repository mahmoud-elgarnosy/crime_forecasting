# import ffmpeg
from torchvision.io import read_video, write_video
from torchaudio.io import StreamReader
import numpy as np
from PIL import Image
import pandas as pd
import os
from UCFCrimeTemporalSplitting import video_name
from tqdm import tqdm
import torch

save_path_images = '../../data/processed/videos_background/images'
os.makedirs(save_path_images, exist_ok=True)

save_path_videos = '../../data/processed/videos_background/vid'
os.makedirs(save_path_videos, exist_ok=True)

df = pd.read_csv('../../data/interim/train/train.csv', names=['path', 'crime_name'])
number_samples = 100
for i in tqdm(range(len(df))):
    video_path = df.iloc[i].path
    _, vid_name = video_name(video_path)
    frames, _, _ = read_video(video_path, pts_unit='sec')
    result = np.median(frames, axis=0).astype(np.uint8)
    Image.fromarray(result).save(f'{save_path_images}/{vid_name}.png')
    if i > number_samples:
        break

# for background removal samples
for i in tqdm(range(len(df))):
    video_path = df.iloc[i].path
    _, vid_name = video_name(video_path)
    frames, _, _ = read_video(video_path, pts_unit='sec')
    result = np.median(frames, axis=0).astype(np.uint8)
    vid_without_background = torch.sub(frames, torch.tensor(result))
    write_video(f'{save_path_videos}/{vid_name}.mp4', vid_without_background, 30)
    if i > number_samples:
        break


