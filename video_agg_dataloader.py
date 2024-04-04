import os
import sys
import json
import random
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms._transforms_video as transforms_video
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip
from pytorchvideo.data.video import VideoPathHandler
from pytorchvideo.transforms.functional import uniform_temporal_subsample
from PIL import Image

def _convert_image_to_rgb(image):
    return image.convert("RGB")
    
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class HowTo100MDataset(Dataset):
    def __init__(self, args, frame_transform, split, first_stage_random_clips=False):
        self.args = args
        self.split = split
        self.num_frames_per_video = self.args.num_frames_per_video
        self.num_frames_per_clip = self.args.num_frames_per_clip
        self.num_segments = self.args.num_segments
        self.frame_size = self.args.frame_size
        self.first_stage_random_clips = first_stage_random_clips
        self.second_stage_random_clips = args.second_stage_random_clips

        if self.first_stage_random_clips:
            if args.single_feature_per_frame:
                self.valid_num_frames = [8, 12, 16, 20, 24, 28, 32]
            else:
                self.valid_num_frames = [8, 12, 16, 20, 24]

        self.video_dir = args.video_dir
        self.second_video_dir = '/net/ivcfs5/mnt/data/long_videos_datasets/howto100m/extracted_frames/all_extracted_frames/'
        self.third_video_dir = '/projectnb/ivc-ml/rxtan/howto100m/extracted_frames/all_extracted_frames/'

        self.data_dir = args.data_dir
        self.all_videos = list(pickle.load(open(os.path.join(self.data_dir, 'all_videos.pkl'), 'rb')))
        self.vid2label = pickle.load(open(os.path.join(self.data_dir, 'vid2label.pkl'), 'rb'))

        self.path_handler = VideoPathHandler()

        self._decode_audio = False
        self._decoder = 'pyav'

        # define video frame transforms
        self.frame_transform = frame_transform

        all_labels = set()
        for vid in self.vid2label:
            all_labels.add(self.vid2label[vid])

    def random_sample_num_frames(self):
        self.num_frames_per_video = random.sample(self.valid_num_frames, 1)[0]
        return

    def parse_text_ann_file(self, filepath):
        data = json.load(open(filepath))
        required_data = {}
        for vid in data:
            if vid in self.all_videos:
                required_data[vid] = data[vid]
        return required_data

    def get_all_videos(self):
        all_videos = set()
        for video in os.listdir(self.video_dir):
            if '.mp4' in video:
                video = video.replace('.mp4', '')
            elif '.npy' in video:
                video = video.replace('.npy', '')
            all_videos.add(video)
        return list(all_videos)
        
    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, index):
        vid = self.all_videos[index]
        
        # decode frames from raw videos on the fly
        '''video_path = os.path.join(self.args.video_dir, '%s.mp4' % vid)
        video = self.path_handler.video_from_path(
                    video_path,
                    decode_audio=self._decode_audio,
                    decoder=self._decoder,
                )
        vid_dur = video.duration.numerator / video.duration.denominator
        clip = video.get_clip(0.0, vid_dur)['video']
        sampled_frames = uniform_temporal_subsample(clip, num_samples=self.num_frames_per_video)

        processed_frames = []
        sampled_frames = sampled_frames.permute(1, 2, 3, 0) # (num_frames, H, W, 3)
        for i in range(len(sampled_frames)):
            curr = np.uint8(sampled_frames[i].numpy())
            curr = self.frame_transform(Image.fromarray(curr))
            processed_frames.append(curr)
        processed_frames = torch.stack(processed_frames)'''

        # reads pre-extracted subsampled frames
        video_path = os.path.join(self.args.video_dir, '%s.npy' % vid)
        if not os.path.exists(video_path):
            video_path = os.path.join(self.second_video_dir, '%s.npy' % vid)
        if not os.path.exists(video_path):
            video_path = os.path.join(self.third_video_dir, '%s.npy' % vid)

        sampled_frames = np.load(video_path)
        selected_num_frames = self.num_frames_per_video

        if len(sampled_frames) <= selected_num_frames:
            selected_indices = np.linspace(0, len(sampled_frames)-1, num=min(selected_num_frames, len(sampled_frames)))
        else:
            selected_indices = np.linspace(0, len(sampled_frames)-1, num=selected_num_frames)

        global_clip_indices = np.linspace(0, len(sampled_frames)-1, num=min(self.num_frames_per_clip, len(sampled_frames)))
        short_window_indices = np.linspace(0, len(sampled_frames)-1, num=min(self.num_frames_per_clip * self.num_segments, len(sampled_frames)))

        global_processed_frames = []
        for i in global_clip_indices:
            i = int(i)
            curr = np.uint8(sampled_frames[i])
            curr = self.frame_transform(Image.fromarray(curr))
            global_processed_frames.append(curr)
        global_processed_frames = torch.stack(global_processed_frames)

        if len(global_processed_frames) < self.num_frames_per_clip:
            diff = self.num_frames_per_clip - len(global_processed_frames)
            pad = global_processed_frames[-1].unsqueeze(0).repeat(diff, 1, 1, 1)
            global_processed_frames = torch.cat((global_processed_frames, pad), dim=0)

        short_window_processed_frames = []
        for i in short_window_indices:
            i = int(i)
            curr = np.uint8(sampled_frames[i])
            curr = self.frame_transform(Image.fromarray(curr))
            short_window_processed_frames.append(curr)
        short_window_processed_frames = torch.stack(short_window_processed_frames)

        if len(short_window_processed_frames) < self.num_frames_per_clip * self.num_segments:
            diff = self.num_frames_per_clip * self.num_segments - len(short_window_processed_frames)
            pad = short_window_processed_frames[-1].unsqueeze(0).repeat(diff, 1, 1, 1)
            short_window_processed_frames = torch.cat((short_window_processed_frames, pad), dim=0)

        global_attn_mask = torch.zeros((self.num_frames_per_clip))
        global_attn_mask[:len(global_processed_frames)] = True

        short_window_attn_mask = torch.zeros((self.num_frames_per_clip * self.num_segments))
        short_window_attn_mask[:len(short_window_processed_frames)] = True

        # get ground-truth goal label
        vid_label = self.vid2label[vid].lower()
        return {'global_video': global_processed_frames, 'global_frame_attn_mask': global_attn_mask, 'segments_video': short_window_processed_frames, 'segments_frame_attn_mask': short_window_attn_mask, 'text': vid_label}