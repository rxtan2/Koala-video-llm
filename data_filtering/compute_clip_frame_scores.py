import json
import pickle
import argparse
import os
import cv2
import csv
import numpy as np
import clip
import torch
from PIL import Image

parser = argparse.ArgumentParser('extract video meta data')
parser.add_argument('--data_dir', type=str, default="/research/reuben/procedural_video_understanding/processed_data/howto100m/", metavar='FC',
                    help='path to output caption dir')
parser.add_argument('--output_data_dir', type=str, default="/research/reuben/procedural_video_understanding/processed_data/howto100m/clip_filtering_scores/", metavar='FC',
                    help='path to output caption dir')
parser.add_argument('--video_dir', type=str, default="/research/reuben/howto100m_videos_subsampled_frames/", metavar='VD',
                    help='path to video dir')
parser.add_argument('--video_split', type=str, default="current video split", metavar='VS',
                    help='specify video split')
parser.add_argument('--clip_model_name', type=str, default="ViT-L/14", metavar='CMN',
                    choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'],
                    help='specify clip model variant')

def main():
    global args
    args = parser.parse_args()

    output_file_path = '%s_frame_scores.pkl' % (args.video_split)
    output_file_path = os.path.join(args.output_data_dir, output_file_path)

    # Loads a dictionary that maps a video id to the corresponding label
    vid2label = pickle.load(open(os.path.join(args.data_dir, 'vid2task_name.pkl'), 'rb'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.clip_model_name, device=device)
    model.eval()

    all_videos = os.listdir(args.video_dir)
    vid_frame_scores = {}
    for idx, video in enumerate(all_videos):
        if idx % 100 == 0:
            print(idx)

        vid_name = video.replace('.npy', '')
        curr_vid_label = vid2label[vid_name].lower()
        curr_input_path = os.path.join(args.video_dir, video)
        curr_frames = preprocess_frames(preprocess, np.load(curr_input_path)).cuda()
        curr_text = clip.tokenize([curr_vid_label]).cuda()

        with torch.no_grad():
            curr_image_features = model.encode_image(curr_frames)
            curr_text_features = model.encode_text(curr_text)

            curr_image_features /= curr_image_features.norm(dim=-1, keepdim=True)
            curr_text_features /= curr_text_features.norm(dim=-1, keepdim=True)
            similarity = (curr_image_features @ curr_text_features.T).squeeze()
            similarity = similarity.cpu().numpy()
            vid_frame_scores[vid_name] = similarity

    pickle.dump(vid_frame_scores, open(output_file_path, 'wb'))
    print('vid_frame_scores: ', len(vid_frame_scores))

def preprocess_frames(frame_transform, frames):
    processed_frames = []
    for idx in range(len(frames)):
        curr = np.uint8(frames[idx])
        curr = frame_transform(Image.fromarray(curr))
        processed_frames.append(curr)
    return torch.stack(processed_frames)
    
if __name__ == '__main__':
    main()
