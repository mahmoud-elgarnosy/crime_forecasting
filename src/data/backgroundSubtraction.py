from torchvision.io import read_video
import numpy as np
from PIL import Image
import pandas as pd
import os
from src.data.UCFCrimeTemporalSplitting import video_name
import cv2
import random
from skimage import metrics


class BackgroundSubtraction:
    def __init__(self, background, shadow_threshold):
        """
        @param background: the background image of the video (equal to applying the median of video frames)
        @param shadow_threshold: shadow removal threshold (remove a few small pixel values whose value does not
                                 exceed this threshold)

        """
        self.__background_image = background
        self.__threshold = shadow_threshold
        # self.backSub = cv2.createBackgroundSubtractorKNN()

    def subtract_background(self, current_frame) -> np.array:
        """
        Subtract The current frame from background
        @param current_frame: the current frame that you need to extract the foreground frame from it
        @return: foreground image after subtract the current frame from the background image
        @rtype: np.array

        """
        # convert images to arrays of type unsigned integer8

        previous_frame = np.array(self.__background_image)
        previous_frame = cv2.resize(previous_frame, (224, 224))
        current_frame = cv2.resize(current_frame, (224, 224))

        score = metrics.structural_similarity(current_frame, previous_frame, channel_axis=2)
        if score > .70:
            # current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            # previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

            # absolute subtraction step
            foreground = cv2.absdiff(current_frame, previous_frame)
        else:
            foreground = current_frame
        self.__background_image = current_frame
        return foreground


# for testing, we will choose random video and apply video background subtraction on it
if __name__ == '__main__':
    # saving images(foreground & background) path
    save_path_videos = '../../data/processed/videos_background/vid/'
    # making dir if it doesn't exist
    os.makedirs(save_path_videos, exist_ok=True)
    # reading the dataframe that have the paths of cutting videos 5secs
    df = pd.read_csv('../../data/interim/train/train.csv', names=['path', 'crime_name'])
    # print(df)

    # reading random video__21__
    video_path = df.iloc[3020].path
    _, vid_name = video_name(video_path)
    frames, _, _ = read_video(video_path, pts_unit='sec')
    bac_image = BackgroundSubtraction(frames[0], 5)

    # looping on videos to extract foreground
    for i, current_frame in enumerate(frames[4::5]):
        current_frame = np.array(current_frame)
        foreground_image = bac_image.subtract_background(current_frame)
        # print(foreground_image.shape)
        cv2.imwrite(f'{save_path_videos}{vid_name}_{i}_bg.png', current_frame)
        cv2.imwrite(f'{save_path_videos}{vid_name}_{i}_fg.png', foreground_image)
