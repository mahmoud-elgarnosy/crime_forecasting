# importing
import json
import csv
import pandas as pd
from torchvision.io import read_video, write_video
import numpy as np
import os
from joblib import Parallel, delayed


def video_name(video_path: str) -> (str, str):
    """
    Taking the absolute path of the video and return the
    @param video_path: absolute path of the video
    @return: the type of the video and the video name without extension
    """
    # for ex. splitting this path "Abuse/Abuse001_x264.mp4"
    split_path = video_path.split("/")

    # extracting the full name of the video (name and its extension)
    video_name_ext = split_path[-1]

    # extracting the video name
    vid_name = video_name_ext.split(".")[0]

    # extracting the name of the crime (the name of the folder)
    crime_name = split_path[0]

    return crime_name, vid_name


def writing_csv_train(video_write_path, crime_name, path):
    """
    Write one rwo in csv for training video every row represent the relative path of the video and its crime type
    @param video_write_path: the absolute path of the video
    @param crime_name: the type of the crime
    @param path: path of the csv file that we write on it
    """
    # Define the row data
    row_data = [video_write_path, crime_name]

    # 1. Open a new CSV file or append on it if it exists
    with open(path + 'train.csv', 'a+') as file:
        # 2. Create a CSV writer
        writer = csv.writer(file)
        # 3. Write the row data to the file
        writer.writerow(row_data)


class UCFTemporalSplitting:
    """
    class for splitting the full video of the crime based on the temporal splitting json file of the version 2 of this
    dataset, for ex if the crime take effect from the seconds 5 to sec 20 we're splitting the full video into 3 videos
    5-10, 10-15, 15-20
    """

    def __init__(self):
        # DEFINE CONSTANTS
        # define the frame per seconds of the video
        self.FRAMES_PER_SEC = 30

        # define the path of the temporal json file
        self.TRAIN_ANNOTATIONS_SECS_LEVEL = '../../data/raw/ucfcrime_v2/public_ucfcrimev2/annotations/train.json'

        # define the relative path of the ucf_crime videos
        self.VIDEO_PATHS = '../../data/raw/'

        # define the writing path of the videos splitting based on temporal
        self.PATH_TRAIN_VIDEO_LEVEL = '../../data/interim/train/'

        # CHOSEN CRIME ACTION THAT CAN BE OCCURRED IN CAMPUS
        self.CRIME_ACTION = ['Abuse', 'Fighting', 'Vandalism']

        # reading the temporal annotation
        temporal_annotation = self.read_temporal_annotation(self.TRAIN_ANNOTATIONS_SECS_LEVEL)
        self.path_csv = self.PATH_TRAIN_VIDEO_LEVEL
        self.csv_file_type = 'train'

        # splitting videos by using joblib multi threads
        self.multiprocessing_temporal_splitting(temporal_annotation)

    def read_temporal_annotation(self, annotation_path: str) -> dict:
        """
        reading and filtering the temporal annotation of the training json file
        @param annotation_path: the relative path of the annotation file
        @return: the filtered dictionary of the annotation path based on the chosen crime types
        """

        # open json file
        with open(annotation_path, 'r') as f:
            temporal_annotation = json.load(f)
        temporal_annotation_copy = temporal_annotation.copy()

        # looping of the dic to remove the crime type that doesn't exist in the chosen crime action
        for d in temporal_annotation:
            if all(word not in d for word in self.CRIME_ACTION):
                del temporal_annotation_copy[d]

        return temporal_annotation_copy

    def write_videos(self, temporal_annotation, video_pth):
        """
        split video (in the video path) based on the temporal annotations and write those split videos on the hard
        @param temporal_annotation: the filtered dict of temporal annotations
        @param video_pth: the path of the target video to be divided
        """

        # reading the temporal annotation of this video
        videos_temporal = temporal_annotation[video_pth]
        # putting it in dataframe
        videos_temporal = pd.DataFrame(videos_temporal)
        # filtering the videos that the period of the crime on it is less than 5
        videos_temporal['period'] = videos_temporal['end'] - videos_temporal['start']
        videos_temporal = videos_temporal[videos_temporal['period'] >= 5]

        # define the full path that to read the original video
        full_path = self.VIDEO_PATHS + video_pth

        # defining the videos to be separated from the current video
        videos = {}

        # if the one video have more than one crime
        for i in range(len(videos_temporal)):
            # defining the divided secs that we split the video based on it, for ex. if start = 5, end = 15 [5, 10, 15]
            videos[i] = np.arange(int(videos_temporal['start'].iloc[i]), int(videos_temporal['end'].iloc[i]), 5)

            # looping on the divided secs
            for j in range(len(videos[i]) - 1):

                # define the start and the end of one of the divided videos
                start = videos[i][j]
                end = videos[i][j + 1]

                # defining the writing path of one divided videos
                crime_name, vid_name = video_name(video_pth)
                write_path = self.path_csv + crime_name
                video_write_path = write_path + '/' + vid_name + f'_{start}_{end}' + '.mp4'

                # making this folder if it doesn't exist
                os.makedirs(write_path, exist_ok=True)

                # reading the video based from the start to the end
                frames, _, _ = read_video(full_path, start_pts=start, end_pts=end, pts_unit='sec')

                # writing divided video on the hard
                write_video(video_write_path, frames, fps=30)

                # writing on the csv train
                writing_csv_train(video_write_path, crime_name, self.path_csv)

    def multiprocessing_temporal_splitting(self, temporal_annotation):
        """
        writing divided videos by multi threads to speed up the operation of writing videos
        @param temporal_annotation: the filtered dict of temporal annotations
        """
        Parallel(n_jobs=os.cpu_count(), prefer='threads', verbose=10)(
            delayed(self.write_videos)(temporal_annotation, video_pth) for video_pth in temporal_annotation)


if __name__ == "__main__":
    # splitting videos
    temp_splitting_train = UCFTemporalSplitting()
