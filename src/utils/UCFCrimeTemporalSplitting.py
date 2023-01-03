# importing
import json
import csv
import pandas as pd
from torchvision.io import read_video, write_video
import numpy as np
import os
from joblib import Parallel, delayed


def video_name(video_path):
    split_path = video_path.split("/")
    video_name_ext = split_path[-1]
    crime_name = split_path[0]
    vid_name = video_name_ext.split(".")[0]
    # video_ext = video_name_ext.split(".")[-1]
    return crime_name, vid_name


def writing_csv_train(video_write_path, crime_name, path, file_type='train'):
    # Define the actual data
    row_data = [video_write_path, crime_name]

    # 1. Open a new CSV file
    with open(path + f'{file_type}.csv', 'a+') as file:
        # 2. Create a CSV writer
        writer = csv.writer(file)
        # 3. Write data to the file
        writer.writerow(row_data)


class UCFTemporalSplitting:
    def __init__(self, train=True):
        # CONSTANTS

        self.FRAMES_PER_SEC = 30

        # UCF_CRIME_DATASET
        self.TRAIN_ANNOTATIONS_SECS_LEVEL = '../data/raw/ucfcrime_v2/public_ucfcrimev2/annotations/train.json'
        self.TEST_ANNOTATIONS_SECS_LEVEL = '../data/raw/ucfcrime_v2/public_ucfcrimev2/annotations/test.json'

        # VIDEO LEVEL FILE PATHS
        self.PATH_TRAIN_VIDEO_LEVEL = '../data/interim/train/'
        self.PATH_TEST_VIDEO_LEVEL = '../data/interim/test/'

        # VIDEO_PATHS
        self.VIDEO_PATHS = '../data/raw/'

        # CHOSEN CRIME ACTION THAT CAN BE OCCURRED IN CAMPUS
        self.CRIME_ACTION = ['Abuse', 'Fighting', 'Vandalism']

        if train:
            temporal_annotation = self.read_temporal_annotation(self.TRAIN_ANNOTATIONS_SECS_LEVEL)
            self.path_csv = self.PATH_TRAIN_VIDEO_LEVEL
            self.csv_file_type = 'train'
        else:
            temporal_annotation = self.read_temporal_annotation(self.TEST_ANNOTATIONS_SECS_LEVEL)
            self.path_csv = self.PATH_TEST_VIDEO_LEVEL
            self.csv_file_type = 'test'

        self.multiprocessing_temporal_splitting(temporal_annotation)

    def read_temporal_annotation(self, annotation_path):
        with open(annotation_path, 'r') as f:
            temporal_annotation = json.load(f)
        temporal_annotation_copy = temporal_annotation.copy()
        for d in temporal_annotation:
            if all(word not in d for word in self.CRIME_ACTION):
                del temporal_annotation_copy[d]
        return temporal_annotation_copy

    def write_videos(self, temporal_annotation, video_pth):
        videos_temporal = temporal_annotation[video_pth]
        videos_temporal = pd.DataFrame(videos_temporal)
        videos_temporal['period'] = videos_temporal['end'] - videos_temporal['start']
        videos_temporal = videos_temporal[videos_temporal['period'] >= 5]
        videos = {}
        for i in range(len(videos_temporal)):
            videos[i] = np.arange(int(videos_temporal['start'].iloc[i]), int(videos_temporal['end'].iloc[i]), 5)
            for video in videos:
                for j in range(len(videos[video])):
                    start = videos[video][j]
                    end = videos[video][j + 1]
                    full_path = self.VIDEO_PATHS + video_pth
                    crime_name, vid_name = video_name(video_pth)
                    write_path = self.PATH_TRAIN_VIDEO_LEVEL + crime_name
                    video_write_path = write_path + '/' + vid_name + f'_{start}_{end}' + '.mp4'
                    os.makedirs(write_path, exist_ok=True)
                    # print(video_write_path)
                    frames, _, _ = read_video(full_path, start_pts=start, end_pts=end, pts_unit='sec')
                    write_video(video_write_path, frames, fps=30)
                    writing_csv_train(video_write_path, crime_name, self.path_csv, self.csv_file_type)

    def multiprocessing_temporal_splitting(self, temporal_annotation):
        Parallel(n_jobs=os.cpu_count(), prefer='threads', verbose=10)(
            delayed(self.write_videos)(temporal_annotation, video_pth) for video_pth in temporal_annotation)
