# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import argparse
import os
import sys
import urllib.request
from collections import OrderedDict
import pickle
import time

import json
import einops
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
import decord
import decord
decord.bridge.set_bridge('torch')

from koala.common.config import Config
from koala.common.dist_utils import get_rank
from koala.common.registry import registry

# imports modules for registration
from koala.datasets.builders import *
from koala.models import *
from koala.processors import *
from koala.runners import *
from koala.tasks import *
from transformers import StoppingCriteria, StoppingCriteriaList

parser = argparse.ArgumentParser('egoschema evaluation script')
parser.add_argument('--caption_output_dir', type=str, default="/path/to/dir/for/saving/predictions", metavar='FC',
                    help='path to output caption dir')
parser.add_argument('--video_dir', type=str, default="/path/to/subsampled/frames", metavar='VD',
                    help='path to video dir which stores subsampled frames')
parser.add_argument('--data_dir', type=str, default="/path/to/egoschema-annotations", metavar='DD',
                    help='path to dir that stores egoschema question and answer annotations')
parser.add_argument('--cfg-path', type=str, default="./train_configs/video_aggregation_finetune.yaml", metavar='CP',
                    help='path to model config file')
parser.add_argument('--options', nargs="+", metavar='CP',
                    help="override some settings in the used config, the key-value pair "
                    "in xxx=yyy format will be merged into config file (deprecate), "
                    "change to --cfg-options instead.")

parser.add_argument('--num_frames', type=int, default=32, metavar='NF',
                    help='specify number of frames to use for each video')
parser.add_argument('--num_frames_per_clip', type=int, default=16, metavar='NPPC',
                    help='specify how frames to use per clip')
parser.add_argument('--num_segments', type=int, default=4, metavar='NS',
                    help='specify number of video segments')
parser.add_argument('--hierarchical_agg_function', type=str, default="without-top-final-global-prompts-region-segment-full-dis-spatiotemporal-prompts-attn-early-attn-linear-learned", metavar='HAF',
                    help='specify function to merge global and clip visual representations')
parser.add_argument('--pretrained_weight_path', type=str, default="/path/to/pretrained/weights", metavar='PWP',
                    help='path to pretrained weight path')
parser.add_argument('--pos_extending_factor', type=int, default=None, metavar='PEF',
                    help='specify scalar value to increase original positional embeddings by')

def main():
    global args
    args = parser.parse_args()

    if not os.path.exists(args.caption_output_dir):
        os.mkdir(args.caption_output_dir)

    # ['q_uid', 'question', 'option 0', 'option 1', 'option 2', 'option 3', 'option 4']
    tmp = json.load(open(os.path.join(args.data_dir, 'questions.json'), 'rb'))
    all_eval_samples = {}
    for i in tmp:
        all_eval_samples[i['q_uid']] = i

    eval_subset_samples = json.load(open(os.path.join(args.data_dir, 'subset_answers.json'), 'rb'))
    final_output_path = os.path.join(args.caption_output_dir, 'segments_%s_frames_%s_predictions.pkl' % (args.num_segments, args.num_frames_per_clip))

    print('final_output_path: ', final_output_path)
    print('')

    cfg = Config(args)
    cfg.pretty_print()

    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)

    model.num_frames_per_clip = args.num_frames_per_clip
    model.num_segments = args.num_segments
    model.hierarchical_agg_function = args.hierarchical_agg_function
    model.global_region_embed_weight = 1e-3

    model.initialize_visual_agg_function()
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    model.pos_extending_factor = args.pos_extending_factor

    if args.pretrained_weight_path:
        best_checkpoint = torch.load(args.pretrained_weight_path, map_location='cpu')['model_state_dict']
        pretrained_dict = {}
        for k, v in best_checkpoint.items():
            pretrained_dict[k.replace('module.', '')] = v
        
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print('Done loading new pretrained weights')

    model = model.cuda().eval()

    stop_words_ids = [torch.tensor([835]).cuda(),
                      torch.tensor([2277, 29937]).cuda()]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    prompt_template = 'Given the question \'%s\', is the answer \'%s\' correct?'
    response_template = 'The most likely answer is %s.'

    model.cuda().eval()

    vid2generations = {}
    correct = 0

    for q_idx, curr_vid in enumerate(all_eval_samples):
        if q_idx % 100 == 0:
            print(q_idx)

        curr_eval_sample = all_eval_samples[curr_vid]
        curr_question = clean_text(curr_eval_sample['question']).lower()
        curr_option_0 = clean_text(curr_eval_sample['option 0'])
        curr_option_1 = clean_text(curr_eval_sample['option 1'])
        curr_option_2 = clean_text(curr_eval_sample['option 2'])
        curr_option_3 = clean_text(curr_eval_sample['option 3'])
        curr_option_4 = clean_text(curr_eval_sample['option 4'])

        curr_all_options = [curr_question, curr_option_0, curr_option_1, curr_option_2, curr_option_3, curr_option_4]

        curr_input_path = os.path.join(args.video_dir, '%s.npy' % curr_vid)
        curr_frames = np.load(curr_input_path)

        curr_scores = extract_frame_captions(vis_processor, args, curr_frames, curr_all_options, model, stopping_criteria)
        curr_scores = curr_scores.tolist()
        vid2generations[curr_vid] = {'log_error_scores': curr_scores}

        if curr_vid in eval_subset_samples:
            curr_correct_answer = eval_subset_samples[curr_vid]
            curr_pred = np.argmin(curr_scores)

            if curr_pred == curr_correct_answer:
                correct += 1

    pickle.dump(vid2generations, open(final_output_path, 'wb'))
    acc = (correct / len(eval_subset_samples)) * 100.
    
    print('final number of eval samples with generated captions: ', len(vid2generations))
    print('acc: ', acc)
    print('final_output_path: ', final_output_path)

def clean_text(text):

    if len(text) >= 2:
        start = text[:2].lower()
        if start == 'c ':
            text = 'The camera wearer ' + text[2:]

    if ' c ' in text:
        text = text.replace(' c ', ' the camera wearer ')
    if  '\'c\'':
        text = text.replace('\'c\'', 'the camera wearer')
    if  '\"c\"':
        text = text.replace('\"c\"', 'the camera wearer')
    if 'C ' in text:
        text = text.replace('C ', 'the camera wearer ')
    if 'c\'s ' in text:
        text = text.replace('c\'s', 'the camera wearer\'s') 
    if 'C\'s ' in text:
        text = text.replace('C\'s', 'The camera wearer\'s') 
    if 'person "c"' in text:
        text = text.replace('person "c"', 'the camera wearer')
    if 'Person "c"' in text:
        text = text.replace('Person "c"', 'the camera wearer')
    if ' c:' in text:
        text = text.replace(' c:', ' the camera wearer:')
    if ' c,' in text:
        text = text.replace(' c,', ' the camera wearer,')
    return text

def get_caption(multimodal_embd, model, stopping_criteria, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        current_max_len = multimodal_embd.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        multimodal_embd = multimodal_embd[:, begin_idx:]

        outputs = model.llama_model.generate(
            inputs_embeds=multimodal_embd,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )

        all_output_text = []
        all_output_token = []
        for idx in range(len(outputs)):
            output_token = outputs[idx]
            if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()

            all_output_text.append(output_text)
            all_output_token.append(output_token.cpu().numpy())

        all_output_token = np.stack(all_output_token)
        return all_output_text, all_output_token

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

def get_multimodal_prompt(model, video_embeds, atts_video, prompt, context=None):
    p_before = '[INST] <Video>'
    p_img = '</Video>'
    p_after = ' [/INST] '

    question = prompt[0]
    responses = prompt[1:]

    question_tokens = model.llama_tokenizer([question], return_tensors="pt", padding="longest", truncation=True, add_special_tokens=False).to(model.device)
    question_input_ids = question_tokens.input_ids
    question_attn_mask = question_tokens.attention_mask
    question_embeds = model.llama_model.model.embed_tokens(question_input_ids)

    responses_tokens = model.llama_tokenizer(responses, return_tensors="pt", padding="longest", truncation=True, add_special_tokens=False).to(model.device)
    responses_input_ids = responses_tokens.input_ids
    responses_attn_mask = responses_tokens.attention_mask

    to_regress_embeds = model.llama_model.model.embed_tokens(responses_input_ids)

    question_embeds = question_embeds.repeat(len(responses_input_ids), 1, 1)
    question_attn_mask = question_attn_mask.repeat(len(responses_input_ids), 1)

    p_before_tokens = model.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(model.device)
    p_before_input_ids = p_before_tokens.input_ids
    p_before_attn_mask = p_before_tokens.attention_mask
    p_before_embeds = model.llama_model.model.embed_tokens(p_before_input_ids)

    p_img_tokens = model.llama_tokenizer(p_img, return_tensors="pt", add_special_tokens=False).to(model.device)
    p_img_input_ids = p_img_tokens.input_ids
    p_img_attn_mask = p_img_tokens.attention_mask
    p_img_embeds = model.llama_model.model.embed_tokens(p_img_input_ids)

    p_after_tokens = model.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(model.device)
    p_after_input_ids = p_after_tokens.input_ids
    p_after_attn_mask = p_after_tokens.attention_mask
    p_after_embeds = model.llama_model.model.embed_tokens(p_after_input_ids)

    bos = torch.ones([1, 1],
                    dtype=responses_input_ids.dtype,
                    device=responses_input_ids.device) * model.llama_tokenizer.bos_token_id
    bos_embeds = model.llama_model.model.embed_tokens(bos)
    atts_bos = p_img_attn_mask[:, :1]

    combined_embeds = torch.cat([bos_embeds.repeat(len(responses_input_ids), 1, 1), p_before_embeds.repeat(len(responses_input_ids), 1, 1), video_embeds.repeat(len(responses_input_ids), 1, 1), p_img_embeds.repeat(len(responses_input_ids), 1, 1), question_embeds, p_after_embeds.repeat(len(responses_input_ids), 1, 1), to_regress_embeds], dim=1)
    combined_attn_mask = torch.cat([atts_bos.repeat(len(responses_input_ids), 1), p_before_attn_mask.repeat(len(responses_input_ids), 1), atts_video.repeat(len(responses_input_ids), 1), p_img_attn_mask.repeat(len(responses_input_ids), 1), question_attn_mask, p_after_attn_mask.repeat(len(responses_input_ids), 1), responses_attn_mask], dim=1)

    targets = responses_input_ids.masked_fill(
                    responses_input_ids == model.llama_tokenizer.pad_token_id,
                    -100)

    empty_targets = (
        torch.ones([combined_attn_mask.shape[0], combined_attn_mask.shape[1] - to_regress_embeds.size(1)],
                    dtype=torch.long).to(responses_input_ids.device).fill_(-100)  # plus one for bos
    )

    targets = torch.cat([empty_targets, targets], dim=1)

    outputs = model.llama_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attn_mask,
            return_dict=True,
            labels=targets,
            no_average=True
        )

    loss = outputs.loss

    loss = loss[:, -to_regress_embeds.size(1):]
    loss = loss.sum(1) / responses_attn_mask.sum(1)

    return loss.cpu()

def process_frame(frame_path, vis_processor):
    frame = Image.open(frame_path).convert('RGB')
    return vis_processor(frame)

def extract_frame_captions(frame_transform, args, all_frames, question, model, stopping_criteria, bs=1, average_frame_feats=False, max_num_frames=32):
    global_clip_indices = np.linspace(0, len(all_frames)-1, num=min(args.num_frames_per_clip, len(all_frames)))
    short_window_indices = np.linspace(0, len(all_frames)-1, num=min(args.num_frames_per_clip * args.num_segments, len(all_frames)))

    global_processed_frames = []
    for i in global_clip_indices:
        i = int(i)
        curr = np.uint8(all_frames[i])
        curr = frame_transform(Image.fromarray(curr))
        global_processed_frames.append(curr)
    global_processed_frames = torch.stack(global_processed_frames)

    if len(global_processed_frames) < args.num_frames_per_clip:
        diff = args.num_frames_per_clip - len(global_processed_frames)
        pad = global_processed_frames[-1].unsqueeze(0).repeat(diff, 1, 1, 1)
        global_processed_frames = torch.cat((global_processed_frames, pad), dim=0)

    short_window_processed_frames = []
    for i in short_window_indices:
        i = int(i)
        curr = np.uint8(all_frames[i])
        curr = frame_transform(Image.fromarray(curr))
        short_window_processed_frames.append(curr)
    short_window_processed_frames = torch.stack(short_window_processed_frames)

    if len(short_window_processed_frames) < args.num_frames_per_clip * args.num_segments:
        diff = args.num_frames_per_clip * args.num_segments - len(short_window_processed_frames)
        pad = short_window_processed_frames[-1].unsqueeze(0).repeat(diff, 1, 1, 1)
        short_window_processed_frames = torch.cat((short_window_processed_frames, pad), dim=0)

    global_attn_mask = torch.zeros((args.num_frames_per_clip))
    global_attn_mask[:len(global_processed_frames)] = True

    short_window_attn_mask = torch.zeros((args.num_frames_per_clip * args.num_segments))
    short_window_attn_mask[:len(short_window_processed_frames)] = True

    global_video = global_processed_frames.unsqueeze(0).cuda()
    global_frame_attn_mask = global_attn_mask.unsqueeze(0).cuda()
    segments_video = short_window_processed_frames.unsqueeze(0).cuda()
    segments_frame_attn_mask = short_window_attn_mask.unsqueeze(0).cuda()

    with torch.no_grad():
        samples = {'global_video': global_video, 'global_frame_attn_mask': global_frame_attn_mask, 'segments_video': segments_video, 'segments_frame_attn_mask': segments_frame_attn_mask}
        merged_video_embeds, merged_video_embeds_mask = model.compute_merged_video_embeds(samples)
        return get_multimodal_prompt(model, merged_video_embeds, merged_video_embeds_mask, question)

if __name__ == '__main__':
    main()
