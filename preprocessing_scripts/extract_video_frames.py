import json
import pickle
import argparse
import os
import sys
import cv2
import csv
import random
import numpy as np

parser = argparse.ArgumentParser('extract video frames')
parser.add_argument('--video_dir', type=str, default="/scratch2/rxtan/howto100m_videos/", metavar='VD',
                    help='path to video dir')
parser.add_argument('--output_dir', type=str, default="/scratch2/rxtan/howto100m_videos/extracted_frames", metavar='VD',
                    help='path to video dir')                    
parser.add_argument('--num_frames_per_video', type=int, default=128, metavar='NF',
                    help='number of frames to sample from a video')
parser.add_argument('--video_boundary_buffer', type=int, default=2, metavar='NF',
                    help='number of frames to subtract from beginning and end of a video')


def main():
    global args
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    selected_videos = sorted(os.listdir(args.video_dir))

    print('total num videos: ', len(selected_videos))
    print('')

    all_durations = []
    vid_meta_data = {}
    valid = 0
    for idx, video in enumerate(selected_videos):
        if idx % 100 == 0:
            print(idx)

        #video = video + '.mp4'
        video_path = os.path.join(args.video_dir, video)

        if '.mp4' in video:
            output_frame_path = os.path.join(args.output_dir, video.replace('.mp4', '.npy'))
        elif '.webm' in video:
            output_frame_path = os.path.join(args.output_dir, video.replace('.webm', '.npy'))
        elif '.mkv' in video:
            output_frame_path = os.path.join(args.output_dir, video.replace('.mkv', '.npy'))
        elif '.avi' in video:
            output_frame_path = os.path.join(args.output_dir, video.replace('.avi', '.npy'))    

        if os.path.exists(output_frame_path):
            continue

        curr_cmd = 'rm -rf %s' % video_path

        try:
            vid_fps, num_all_frames, duration = extract_video_metadata(video_path)
            num_1fps_frames = extract_video_frames(video_path)

            curr_num_valid_frames = len(num_1fps_frames) - (args.video_boundary_buffer*2)
            if curr_num_valid_frames <= 0:
                continue

            curr_subsampled_frames = num_1fps_frames

            all_durations.append(len(curr_subsampled_frames))

            if len(num_1fps_frames) <= args.num_frames_per_video:
                curr_subsampled_frames = num_1fps_frames
            else:
                curr_selected_indices = np.linspace(0, len(num_1fps_frames)-1, num=args.num_frames_per_video)
                curr_subsampled_frames = []
                for curr_frame_idx in curr_selected_indices:
                    curr_frame_idx = int(curr_frame_idx)
                    curr_subsampled_frames.append(num_1fps_frames[curr_frame_idx])
                curr_subsampled_frames = np.stack(curr_subsampled_frames)

            np.save(output_frame_path, curr_subsampled_frames)
            valid += 1

            curr_cmd = 'rm -rf %s' % video_path
            os.system(curr_cmd)

        except:
            continue
    
    print('valid: ', valid)
    print('total_num: ', len(selected_videos))
    print('final: ', len(os.listdir(args.output_dir)))
    print('max duration: ', np.max(all_durations))
    print('min duration: ', np.min(all_durations))
    print('avg duration: ', np.average(all_durations))
    print('std duration: ', np.std(all_durations))
    

def extract_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    round_vid_fps = round(vid_fps)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = num_frames / round_vid_fps
    return vid_fps, num_frames, duration

def extract_video_frames(video_path, selected_fps=1):
    cap = cv2.VideoCapture(video_path)
    vid_fps = round(cap.get(cv2.CAP_PROP_FPS))
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = num_frames / vid_fps
    hop = round(vid_fps / selected_fps)
    video_frames = []
    curr_frame = 0
    while True:
        success, frame = cap.read()

        if not success:
            break

        if curr_frame % hop == 0:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256,256), interpolation = cv2.INTER_AREA) # resize for egoschema
            video_frames.append(frame)
        curr_frame += 1

    video_frames = np.stack(video_frames)
    return video_frames
    
if __name__ == '__main__':
    main()
