import pandas as pd
from UCFCrimeTemporalSplitting import writing_csv_train, video_name
import os
import numpy as np


def extract_temporal_annotation_video(frame_annotation_path):
    video_temporal = {}
    df = pd.read_csv(frame_annotation_path, names=['annotation'])
    index = df[df.annotation == 1].index
    video_temporal['start'], video_temporal['end'] = index[0] / 30, index[-1] / 30
    video_temporal['period'] = video_temporal['end'] - video_temporal['start']

    if video_temporal['period'] >= 5:
        return video_temporal
    else:
        return None


class UBITemporalSplitting:
    def __init__(self):

        # Constants
        self.Frame_LEVEL_ANNOTATION_PATH = '../../data/raw/UBI_FIGHTS/UBI_FIGHTS/annotation'
        self.FIGHT_VIDEOS_PATH = '../../data/raw/UBI_FIGHTS/UBI_FIGHTS/videos/fight'
        self.NORMAL_VIDEOS_PATH = '../../data/raw/UBI_FIGHTS/UBI_FIGHTS/videos/normal'
        self.PATH_TRAIN_VIDEO_LEVEL = '../../data/interim/train/'
        self.PATH_TEST_VIDEO_LEVEL = '../../data/interim/test/'

    def write_videos(self):
        crime_name = 'Fighting'
        dir_list = os.listdir(self.FIGHT_VIDEOS_PATH)
        videos = {}
        for i, video_path in enumerate(dir_list):
            _, vid_name = video_name(video_path)
            full_video_path = self.FIGHT_VIDEOS_PATH + video_path
            full_frame_annotation_path = self.Frame_LEVEL_ANNOTATION_PATH + '/' + vid_name + '.csv'

            videos_temporal = extract_temporal_annotation_video(full_frame_annotation_path)
            videos[i] = np.arange(int(videos_temporal['start']), int(videos_temporal['end']), 5)
            for video in videos:
                # for j in range(len(videos[video])):
                print(videos)
                    # break
                break
            break

u = UBITemporalSplitting()
u.write_videos()
