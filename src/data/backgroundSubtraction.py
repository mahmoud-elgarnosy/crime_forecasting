from torchvision.io import read_video
import numpy as np
from PIL import Image
import pandas as pd
import os
from UCFCrimeTemporalSplitting import video_name
import cv2


class BackgroundSubtraction:
    def __init__(self, background, shadow_threshold):
        """
        @param background: the background image of the video (equal to applying the median of video frames)
        @param shadow_threshold: shadow removal threshold (remove a few small pixel values whose value does not
                                 exceed this threshold)

        """
        self.__background_image = background
        self.__threshold = shadow_threshold

    def subtract_background(self, frame: Image) -> np.array:
        """
        Subtract The current frame from background
        @param frame: the current frame that you need to extract the foreground frame from it
        @return: foreground image after subtract the current frame from the background image
        @rtype: np.array

        """
        # convert images to arrays of type unsigned integer8
        frame = np.array(frame).astype(np.uint8)
        self.__background_image = np.array(self.__background_image).astype(np.uint8)

        # absolute subtraction step
        subtraction = np.abs(np.subtract(frame, self.__background_image))

        # shadow removal step
        subtraction_mask = (subtraction[:, :, 0] < self.__threshold) | (subtraction[:, :, 1] < self.__threshold) | \
                           (subtraction[:, :, 2] < self.__threshold)

        # applying mask to the frame
        frame[subtraction_mask] = 0

        # updating background
        self.update_bg(frame)

        return frame

    def update_bg(self, frame, alpha=.02):
        """
        By applying moving average of all past frames and the median of all frames, we update the background image
        after each subtraction operation

        :param frame: current frame that you need to apply the moving average on it
        :param alpha: The alpha value regulates the quantity of the new frame that will be used as background in the
        update of the background.
        """
        accu_weight = cv2.accumulateWeighted(frame, self.__background_image.astype(float), alpha)
        self.__background_image = cv2.convertScaleAbs(accu_weight)  # Use this to convert to uint8


# for testing, we will choose random video and apply video background subtraction on it
if __name__ == '__main__':
    # saving images(foreground & background) path
    save_path_videos = '../../data/processed/videos_background/vid/'
    # making dir if it doesn't exist
    os.makedirs(save_path_videos, exist_ok=True)
    # reading the dataframe that have the paths of cutting videos 5secs
    df = pd.read_csv('../../data/interim/train/train.csv', names=['path', 'crime_name'])

    # reading random video__21__
    video_path = df.iloc[21].path
    _, vid_name = video_name(video_path)
    frames, _, _ = read_video(video_path, pts_unit='sec')

    # apply the median of the video frames to extract the background of the video
    background_image = np.median(frames, axis=0).astype(np.uint8)
    Image.fromarray(np.array(background_image), "RGB").save(f'{save_path_videos}_bg.png')

    # initiate the BackgroundSubtraction class
    bac_image = BackgroundSubtraction(background_image, 100)

    # looping on videos to extract foreground
    for i, frame in enumerate(frames):
        if i % 3 == 0:
            foreground_image = bac_image.subtract_background(frame)
            Image.fromarray(np.array(frame), "RGB").save(f'{save_path_videos}_{vid_name}_{i}.png')
            Image.fromarray(np.array(foreground_image), "RGB").convert("L").save(f'{save_path_videos}'
                                                                                 f'_{vid_name}_{i}_fg.png')


