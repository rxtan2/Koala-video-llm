import logging
import random
import sys

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

from koala.common.registry import registry
from koala.models.blip2 import Blip2Base, disabled_train
from koala.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer,BertConfig
import einops
import copy
from koala.models.Qformer import BertConfig, BertLMHeadModel
from koala.models.ImageBind.models.imagebind_model import ImageBindModel,ModalityType
from koala.models.ImageBind.models import imagebind_model
# from flamingo_pytorch import PerceiverResampler
@registry.register_model("video_aggregation_llama")
class VideoAggregationLLAMA(Blip2Base):
    """
    Koala model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/video_llama.yaml",
        "pretrain_llama_v2": "configs/models/video_llama.yaml",
    }

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width,num_hidden_layers =2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.

        frozen_llama_proj=True,
        frozen_video_Qformer=True,
        frozen_audio_Qformer=True,

        llama_proj_model='',
        fusion_header_type= "seqTransf",
        max_frame_pos= 32,
        fusion_head_layers = 2,
        num_video_query_token = 32,
        num_audio_query_token = 8,
        imagebind_ckpt_path = '/mnt/workspace/ckpt',
        equip_audio_branch = True
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        logging.info('Loading Q-Former Done')

        logging.info('Loading LLAMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token 
        DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
        DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
        self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.llama_tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        
        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]
        self.AUDIO_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_AUDIO_PATCH_TOKEN]

        logging.info('Loading LLAMA Model')
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLAMA Done')


        logging.info('Loading LLAMA proj')
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        if llama_proj_model:
            print("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            msg = self.load_state_dict(llama_proj_weight['model'], strict=False)

        if frozen_llama_proj:
            #  todo frozen  llama_proj
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            logging.info('LLAMA proj is frozen')
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            logging.info('LLAMA proj is not frozen')

        logging.info('Loading llama_proj Done')

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.Qformer.config.hidden_size)

        self.num_video_query_token = num_video_query_token
        self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = num_video_query_token,\
            vision_width=self.Qformer.config.hidden_size, num_hidden_layers =2)

        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None


        if frozen_video_Qformer:
            #  todo frozen  llama_proj
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = False
            self.video_query_tokens.requires_grad = False
            
            logging.info('video_Qformer is frozen')
        else:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = True
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = True
            self.video_query_tokens.requires_grad = True
            logging.info('video_Qformer is not frozen')

        if frozen_video_Qformer and (not frozen_audio_Qformer):
            self.train_flag = 1 # 只训练audio_Qformer
        elif not(frozen_video_Qformer) and frozen_audio_Qformer:
            self.train_flag = 0 # 训练video_Qformer
        elif not(frozen_video_Qformer) and not(frozen_audio_Qformer):
            self.train_flag = 2 # video_Qformer and AL trained
        else:
            self.train_flag = 3

        if equip_audio_branch:
            print (f'Initializing audio encoder from {imagebind_ckpt_path} ...')
            self.audio_encoder,self.audio_hidden_size = \
                imagebind_model.imagebind_huge()
            self.audio_encoder.load_state_dict(torch.load("{}/imagebind_huge.pth".format(imagebind_ckpt_path)))
            # free vision encoder
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
            print ('audio encoder initialized.')
            
            self.num_audio_query_token = num_audio_query_token
            self.audio_Qformer,self.audio_query_tokens = self.init_video_Qformer(num_query_token = self.num_audio_query_token,\
                vision_width=self.audio_hidden_size, num_hidden_layers =2)
            self.audio_Qformer.cls = None
            self.audio_Qformer.bert.embeddings.word_embeddings = None
            self.audio_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.audio_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.audio_llama_proj = nn.Linear(
                self.audio_Qformer.config.hidden_size, self.llama_model.config.hidden_size
            )
            self.audio_position_embedding = nn.Embedding(8, self.audio_hidden_size)

            if frozen_audio_Qformer:
                #  todo frozen  llama_proj
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = False
                self.audio_query_tokens.requires_grad = False
                for name, param in self.audio_llama_proj.named_parameters():
                    param.requires_grad = False
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = False
                logging.info('audio_Qformer and audio-LLAMA proj is frozen')
            else:
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = True
                self.audio_query_tokens.requires_grad = True
                for name, param in self.audio_llama_proj.named_parameters():
                    param.requires_grad = True
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = True
                logging.info('audio_Qformer is not frozen')

        # initialize additional arguments
        self.pos_extending_factor = None

        self.prompt_list = ['[INST] <Video><ImageHere></Video> What is the most likely objective in the video? [/INST]',
                            '[INST] <Video><ImageHere></Video> What is the most likely goal in the video? [/INST]',
                            '[INST] <Video><ImageHere></Video> What is the person trying to do in the video? [/INST]',
                            '[INST] <Video><ImageHere></Video> What is happening in the video? [/INST]',
                            '[INST] <Video><ImageHere></Video> Describe the most likely objective in the video. [/INST]',
                            '[INST] <Video><ImageHere></Video> Describe the most likely goal in the video. [/INST]',
                            '[INST] <Video><ImageHere></Video> Describe what the person is trying to do in the video. [/INST]',
                            '[INST] <Video><ImageHere></Video> Describe what is happening in the video. [/INST]',]

        self.video_level_reply_prefix_list = ['The most likely objective in the video is to ',
                                              'The most likely goal is to ',
                                              'The person is trying to ',
                                              'This video demonstrates the steps to ',
                                              'The most likely objective in the video is to ',
                                              'The most likely goal is to ',
                                              'The person is trying to ',
                                              'This video demonstrates the steps to ',]

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def initialize_visual_agg_function(self):
        if self.hierarchical_agg_function == 'without-top-final-global-prompts-region-segment-full-dis-spatiotemporal-prompts-attn-early-attn-linear-learned':
            self.video_global_proj = nn.Linear(self.Qformer.config.hidden_size, self.llama_model.config.hidden_size)
            self.video_global_proj.load_state_dict(self.llama_proj.state_dict())

            for name, param in self.video_global_proj.named_parameters():
                param.requires_grad = True

            if 'without-top' not in self.hierarchical_agg_function:
                self.global_region_prompts = nn.Parameter(torch.zeros(1, self.video_query_tokens.size(1), self.video_query_tokens.size(-1)))
                self.global_region_prompts.data = self.video_query_tokens.data.clone()
                self.global_region_prompts.requires_grad = True

            self.segment_region_prompts = nn.Parameter(torch.zeros(1, self.video_query_tokens.size(1), self.video_query_tokens.size(-1)))
            self.segment_region_prompts.data = self.video_query_tokens.data.clone()
            self.segment_region_prompts.requires_grad = True

            if 'region-prompts' in self.hierarchical_agg_function:
                self.segment_attn_queries = nn.Parameter(torch.zeros(1, self.num_segments, self.video_query_tokens.size(-1)))
                self.segment_attn_queries.data = self.video_query_tokens.data[:, :self.num_segments].clone()
                self.segment_attn_queries.requires_grad = True

            if 'spatiotemporal-prompts' in self.hierarchical_agg_function:
                if 'full-dis-spatiotemporal' in self.hierarchical_agg_function:
                    self.spatial_segment_prompts = nn.Parameter(torch.zeros(1, self.video_query_tokens.size(1), self.video_query_tokens.size(-1)))
                    self.spatial_segment_prompts.data = self.video_query_tokens.data.clone()
                    self.spatial_segment_prompts.requires_grad = True

                    self.temporal_segment_prompts = nn.Parameter(torch.zeros(1, self.num_segments, self.video_query_tokens.size(-1)))
                    self.temporal_segment_prompts.data = self.video_query_tokens.data.clone().mean(1, keepdim=True).repeat(1, self.num_segments, 1)
                    self.temporal_segment_prompts.requires_grad = True

                elif 'full-spatiotemporal' not in self.hierarchical_agg_function:
                    self.temporal_segment_prompts = nn.Parameter(torch.zeros(1, self.num_segments, self.video_query_tokens.size(-1)))
                    self.temporal_segment_prompts.data = self.video_query_tokens.data[:, :self.num_segments].clone()
                    self.temporal_segment_prompts.requires_grad = True

                    self.spatial_segment_prompts = nn.Parameter(torch.zeros(1, self.video_query_tokens.size(1), self.video_query_tokens.size(-1)))
                    self.spatial_segment_prompts.data = self.video_query_tokens.data.clone()
                    self.spatial_segment_prompts.requires_grad = True
                else:
                    self.spatial_segment_prompts = nn.Parameter(torch.zeros(self.num_segments, self.video_query_tokens.size(1), self.video_query_tokens.size(-1)))
                    self.spatial_segment_prompts.data = self.video_query_tokens.data.clone().repeat(self.num_segments, 1, 1)
                    self.spatial_segment_prompts.requires_grad = True

            if 'final-global-prompts' in self.hierarchical_agg_function:
                self.global_to_segment_prompts = nn.Parameter(torch.zeros(1, self.video_query_tokens.size(1), self.video_query_tokens.size(-1)))
                self.global_to_segment_prompts.data = self.video_query_tokens.data.clone()
                self.global_to_segment_prompts.requires_grad = True

            if 'proj-' in self.hierarchical_agg_function:
                self.global_frame_proj = nn.Linear(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size)
                self.global_segment_proj = nn.Linear(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size)

        if '-learned' in self.hierarchical_agg_function:
            if self.global_region_embed_weight is None:
                self.global_region_embed_weight = nn.Parameter(data=torch.rand(1))
            else:
                self.global_region_embed_weight = nn.Parameter(data=torch.Tensor([self.global_region_embed_weight]))
            self.global_region_embed_weight.requires_grad = True

        return

    def encode_videoQformer_visual(self, image, frame_attn_mask, global_video=True):
        device = image.device

        # input shape b,t,c,h,w
        batch_size, time_length, _, _, _ = image.size()

        image = einops.rearrange(image, 'b t c h w -> (b t) c h w')

        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

            # frame attention
            frame_hidden_state = einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)

            frame_attn_mask = frame_attn_mask.unsqueeze(-1).repeat(1, 1, q_hidden_state.size(1))
            frame_attn_mask = frame_attn_mask.view(frame_attn_mask.size(0), -1)
            frame_atts = frame_atts * frame_attn_mask

            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                #output_attentions=True,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state
            atts_llama = torch.ones(video_hidden.size()[:-1], dtype=torch.long).to(image_embeds.device)

        return video_hidden, atts_llama

    def encode_frame_level_visual(self, image, frame_attn_mask, return_attn=False):
        device = image.device

        # input shape b,t,c,h,w
        batch_size, time_length, _, _, _ = image.size()
        image = einops.rearrange(image, 'b t c h w -> (b t) c h w')

        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                output_attentions=return_attn,
                return_dict=True,
            )

            q_hidden_state = query_output.last_hidden_state
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)

        if return_attn:
            with self.maybe_autocast():
                _, frame_patch_attn = self.visual_encoder.get_attn_weights(image)
            return frame_hidden_state, frame_atts, query_output['attentions'], query_output['cross_attentions'], frame_patch_attn
        else:
            return frame_hidden_state, frame_atts
    
    
    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            # print(prompt)
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img
    #  input audio shape [b t c h w] 
    def encode_audioQformer(self, audio,modality_type=ModalityType.AUDIO):
        device = audio.device
        with self.maybe_autocast():
            audio_feature, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(audio,modality_type=modality_type)
            batch_size,time_length = audio.size()[:2]


            position_ids = torch.arange(time_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            audio_position_embeddings = self.audio_position_embedding(position_ids)
            audio_imagebind_finalout = audio_imagebind_finalout + audio_position_embeddings

            audio_query_tokens = self.audio_query_tokens.expand(audio_imagebind_finalout.shape[0], -1, -1)
            frame_atts = torch.ones(audio_imagebind_finalout.size()[:-1], dtype=torch.long).to(device)

            audio_query_output = self.audio_Qformer.bert(
                query_embeds=audio_query_tokens, #[32,768]
                encoder_hidden_states=audio_imagebind_finalout,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            audio_hidden = audio_query_output.last_hidden_state

            inputs_llama = self.audio_llama_proj(audio_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
    
        return inputs_llama, atts_llama

    def encode_videoQformer_audiovideo(self, image, audio):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            
            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

            # encode audio 
            audio_feature, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(audio,modality_type=ModalityType.AUDIO) # [batch,8*1,768]    8*32, 768
            audio_frame_position_embeddings = frame_position_embeddings.squeeze(-2)
            audio_feature = audio_feature + audio_frame_position_embeddings

            # frame attention a
            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
            frame_hidden_state = torch.cat([frame_hidden_state,audio_feature],dim = 1)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)

            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens, #[32,768]
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            inputs_llama = self.llama_proj(video_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
    
        return inputs_llama, atts_llama

    def forward(self, samples):

        global_video = samples['global_video'].cuda()
        global_frame_attn_mask = samples['global_frame_attn_mask'].cuda()
        segments_video = samples['segments_video'].cuda()
        segments_frame_attn_mask = samples['segments_frame_attn_mask'].cuda()
        text = samples['text']

        batch_size = global_video.size(0)

        global_video_embeds, global_video_embeds_mask = self.encode_videoQformer_visual(global_video, global_frame_attn_mask)
        segments_video = segments_video.view(-1, self.num_frames_per_clip, segments_video.size(-3), segments_video.size(-2), segments_video.size(-1))
        segments_frame_attn_mask = segments_frame_attn_mask.view(-1, self.num_frames_per_clip)

        segments_video_embeds, segments_video_embeds_mask = self.encode_frame_level_visual(segments_video, segments_frame_attn_mask)
        segments_video_embeds = segments_video_embeds.view(-1, self.num_segments, segments_video_embeds.size(-3), segments_video_embeds.size(-2), segments_video_embeds.size(-1))
        segments_video_embeds_mask = segments_video_embeds_mask.view(-1, self.num_segments, segments_video_embeds_mask.size(-2), segments_video_embeds_mask.size(-1))

        if self.hierarchical_agg_function == 'without-top-final-global-prompts-region-segment-full-dis-spatiotemporal-prompts-attn-early-attn-linear-learned':
            # add segment pos embedding
            position_ids = torch.arange(segments_video_embeds.size(2), dtype=torch.long, device=self.video_query_tokens.device)
            segments_video_embeds = segments_video_embeds.view(batch_size*self.num_segments, segments_video_embeds.size(-3), segments_video_embeds.size(-2), segments_video_embeds.size(-1))
            segments_video_embeds_mask = segments_video_embeds_mask.view(batch_size*self.num_segments, segments_video_embeds_mask.size(-2), segments_video_embeds_mask.size(-1))

            position_ids = position_ids.unsqueeze(0).expand(batch_size*self.num_segments, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            segments_video_embeds = frame_position_embeddings + segments_video_embeds
            segments_video_embeds_mask = segments_video_embeds_mask * segments_frame_attn_mask.unsqueeze(-1)

            segments_video_embeds = segments_video_embeds.view(segments_video_embeds.size(0), -1, segments_video_embeds.size(-1))
            segments_video_embeds_mask = segments_video_embeds_mask.view(segments_video_embeds_mask.size(0), -1)
            video_query_tokens = self.video_query_tokens.expand(segments_video_embeds.shape[0], -1, -1)
            num_region_queries = video_query_tokens.size(1)

            # add short video segment prompts
            curr_segment_query_tokens = self.segment_region_prompts.expand(global_video_embeds.shape[0], -1, -1)
            global_context = global_video_embeds + curr_segment_query_tokens
            global_context = global_context.unsqueeze(1).repeat(1, self.num_segments, 1, 1)
            global_context = global_context.view(-1, global_context.size(-2), global_context.size(-1))

            video_query_tokens = torch.cat([video_query_tokens, global_context], dim=1)

            global_region_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=segments_video_embeds,
                encoder_attention_mask=segments_video_embeds_mask,
                #output_attentions=True,
                return_dict=True,
                )
            global_region_segment_embeds = global_region_query_output.last_hidden_state[:, :num_region_queries]
            global_region_segment_embeds = global_region_segment_embeds.view(batch_size, self.num_segments, global_region_segment_embeds.size(-2), global_region_segment_embeds.size(-1))

            # add segment pos embedding
            position_ids = torch.arange(self.num_segments, dtype=torch.long, device=segments_video_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(len(global_video_embeds), -1)
            segments_position_embeddings = self.video_frame_position_embedding(position_ids)

            segments_position_embeddings = segments_position_embeddings.unsqueeze(-2)
            segments_hidden_state = segments_position_embeddings + global_region_segment_embeds

            segment_temporal_context = segments_hidden_state.mean(1)
            segment_spatial_context = segments_hidden_state.mean(2)

            if 'spatiotemporal-prompts' in self.hierarchical_agg_function:

                if 'full-dis-spatiotemporal' in self.hierarchical_agg_function:
                    temporal_context_prompts = self.temporal_segment_prompts.unsqueeze(-2).expand(global_video_embeds.shape[0], -1, segments_hidden_state.size(-2), -1)
                    spatial_context_prompts = self.spatial_segment_prompts.unsqueeze(0).expand(global_video_embeds.shape[0], self.num_segments, -1, -1)

                    final_context = segments_hidden_state + temporal_context_prompts + spatial_context_prompts
                    final_context = final_context.view(final_context.size(0), -1, final_context.size(-1))

            if 'without' in self.hierarchical_agg_function and 'full-dis-spatiotemporal' in self.hierarchical_agg_function:
                final_top_down_context = final_context

            final_top_down_context_mask = torch.ones(final_top_down_context.size()[:-1], dtype=torch.long).to(final_top_down_context.device)
            if 'without-top' in self.hierarchical_agg_function:
                merged_query_tokens = self.video_query_tokens.expand(len(global_video_embeds), -1, -1)

            if 'final-global-prompts' in self.hierarchical_agg_function:
                global_to_segment_prompts = self.global_to_segment_prompts.expand(global_video_embeds.shape[0], -1, -1)
                global_to_segment_context = global_video_embeds + global_to_segment_prompts

                merged_query_tokens = torch.cat([merged_query_tokens, global_to_segment_context], dim=1)

            global_region_output = self.video_Qformer.bert(
                query_embeds=merged_query_tokens,
                encoder_hidden_states=final_top_down_context,
                encoder_attention_mask=final_top_down_context_mask,
                #output_attentions=True,
                return_dict=True,
            )
            global_region_merged_embeds = global_region_output.last_hidden_state

            if 'final-global-prompts' in self.hierarchical_agg_function:
                global_region_merged_embeds = global_region_merged_embeds[:, :num_region_queries]

            global_region_merged_embeds = self.video_global_proj(global_region_merged_embeds)

            merged_video_embeds = global_video_embeds
            merged_video_embeds_mask = global_video_embeds_mask

        merged_video_embeds = self.llama_proj(merged_video_embeds)
        merged_video_embeds = merged_video_embeds + self.global_region_embed_weight * global_region_merged_embeds

        prompt_idx = random.randint(0, len(self.prompt_list)-1)
        prompt = self.prompt_list[prompt_idx]
        response_prefix = self.video_level_reply_prefix_list[prompt_idx]
        text = [response_prefix + t + '.' + self.end_sym for t in samples["text"]]

        merged_video_embeds, merged_atts_video = self.prompt_wrap(merged_video_embeds, merged_video_embeds_mask, prompt)

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(global_video.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([merged_atts_video.shape[0], merged_atts_video.shape[1]+1],
                       dtype=torch.long).to(global_video.device).fill_(-100)  # plus one for bos
        )

        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = merged_video_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                        dtype=to_regress_tokens.input_ids.dtype,
                        device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = merged_atts_video[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, merged_video_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, merged_atts_video, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

        return {"loss": loss}

    def compute_merged_video_embeds(self, samples):

        global_video = samples['global_video'].cuda()
        global_frame_attn_mask = samples['global_frame_attn_mask'].cuda()
        segments_video = samples['segments_video'].cuda()
        segments_frame_attn_mask = samples['segments_frame_attn_mask'].cuda()

        global_video_embeds, global_video_embeds_mask = self.encode_videoQformer_visual(global_video, global_frame_attn_mask)

        segments_video = segments_video.view(-1, self.num_frames_per_clip, segments_video.size(-3), segments_video.size(-2), segments_video.size(-1))
        segments_frame_attn_mask = segments_frame_attn_mask.view(-1, self.num_frames_per_clip)

        if 'early-attn' not in self.hierarchical_agg_function:
            segments_video_embeds, segments_video_embeds_mask = self.encode_videoQformer_visual(segments_video, segments_frame_attn_mask, global_video=False)

            segments_video_embeds = segments_video_embeds.view(-1, self.num_segments, segments_video_embeds.size(-2), segments_video_embeds.size(-1))
            segments_video_embeds_mask = segments_video_embeds_mask.view(-1, self.num_segments, segments_video_embeds_mask.size(-1))
        else:
            segments_video_embeds, segments_video_embeds_mask = self.encode_frame_level_visual(segments_video, segments_frame_attn_mask)
            segments_video_embeds = segments_video_embeds.view(-1, self.num_segments, segments_video_embeds.size(-3), segments_video_embeds.size(-2), segments_video_embeds.size(-1))
            segments_video_embeds_mask = segments_video_embeds_mask.view(-1, self.num_segments, segments_video_embeds_mask.size(-2), segments_video_embeds_mask.size(-1))

        if 'fq-without' in self.hierarchical_agg_function:
            curr_frame_query_prompts = self.frame_query_prompts.unsqueeze(0).unsqueeze(0)
            curr_frame_query_prompts = curr_frame_query_prompts.repeat(segments_video_embeds.size(0), segments_video_embeds.size(1), segments_video_embeds.size(2), 1, 1)
            segments_video_embeds = segments_video_embeds + curr_frame_query_prompts

        batch_size = global_video_embeds.size(0)

        if self.hierarchical_agg_function == 'without-top-final-global-prompts-region-segment-full-dis-spatiotemporal-prompts-attn-early-attn-linear-learned':
            # add segment pos embedding
            position_ids = torch.arange(segments_video_embeds.size(2), dtype=torch.long, device=self.video_query_tokens.device)
            segments_video_embeds = segments_video_embeds.view(batch_size*self.num_segments, segments_video_embeds.size(-3), segments_video_embeds.size(-2), segments_video_embeds.size(-1))
            segments_video_embeds_mask = segments_video_embeds_mask.view(batch_size*self.num_segments, segments_video_embeds_mask.size(-2), segments_video_embeds_mask.size(-1))

            position_ids = position_ids.unsqueeze(0).expand(batch_size*self.num_segments, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            segments_video_embeds = frame_position_embeddings + segments_video_embeds
            segments_video_embeds_mask = segments_video_embeds_mask * segments_frame_attn_mask.unsqueeze(-1)

            segments_video_embeds = segments_video_embeds.view(segments_video_embeds.size(0), -1, segments_video_embeds.size(-1))
            segments_video_embeds_mask = segments_video_embeds_mask.view(segments_video_embeds_mask.size(0), -1)
            video_query_tokens = self.video_query_tokens.expand(segments_video_embeds.shape[0], -1, -1)
            num_region_queries = video_query_tokens.size(1)

            # add short video segment prompts
            if 'n1key' not in self.hierarchical_agg_function and 'n2key' not in self.hierarchical_agg_function:
                curr_segment_query_tokens = self.segment_region_prompts.expand(global_video_embeds.shape[0], -1, -1)
                global_context = global_video_embeds + curr_segment_query_tokens

                global_context = global_context.unsqueeze(1).repeat(1, self.num_segments, 1, 1)

                global_context = global_context.view(-1, global_context.size(-2), global_context.size(-1))

                video_query_tokens = torch.cat([video_query_tokens, global_context], dim=1)

            global_region_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=segments_video_embeds,
                encoder_attention_mask=segments_video_embeds_mask,
                #output_attentions=True,
                return_dict=True,
                )
            global_region_segment_embeds = global_region_query_output.last_hidden_state[:, :num_region_queries]
            global_region_segment_embeds = global_region_segment_embeds.view(batch_size, self.num_segments, global_region_segment_embeds.size(-2), global_region_segment_embeds.size(-1))

            # add segment pos embedding
            position_ids = torch.arange(self.num_segments, dtype=torch.long, device=segments_video_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(len(global_video_embeds), -1)
            segments_position_embeddings = self.video_frame_position_embedding(position_ids)

            segments_position_embeddings = segments_position_embeddings.unsqueeze(-2)
            segments_hidden_state = segments_position_embeddings + global_region_segment_embeds

            segment_temporal_context = segments_hidden_state.mean(1)
            segment_spatial_context = segments_hidden_state.mean(2)

            if 'spatiotemporal-prompts' in self.hierarchical_agg_function:

                if 'full-dis-spatiotemporal' in self.hierarchical_agg_function and 'none-without' not in self.hierarchical_agg_function:
                    temporal_context_prompts = self.temporal_segment_prompts.unsqueeze(-2).expand(global_video_embeds.shape[0], -1, segments_hidden_state.size(-2), -1)
                    spatial_context_prompts = self.spatial_segment_prompts.unsqueeze(0).expand(global_video_embeds.shape[0], self.num_segments, -1, -1)

                    final_context = segments_hidden_state + temporal_context_prompts + spatial_context_prompts
                    final_context = final_context.view(final_context.size(0), -1, final_context.size(-1))
                elif 'full-dis-spatiotemporal' in self.hierarchical_agg_function and 'none-without' in self.hierarchical_agg_function:
                    final_context = segments_hidden_state
                    final_context = final_context.view(final_context.size(0), -1, final_context.size(-1))

            if 'without' in self.hierarchical_agg_function and 'full-dis-spatiotemporal' in self.hierarchical_agg_function:
                final_top_down_context = final_context
            elif 'without' in self.hierarchical_agg_function and 'full-spatiotemporal' not in self.hierarchical_agg_function:
                final_top_down_context = torch.cat([segment_temporal_context, segment_spatial_context], dim=1)
            elif 'without' in self.hierarchical_agg_function and 'full-spatiotemporal' in self.hierarchical_agg_function:
                final_top_down_context = segment_spatial_context
            else:
                final_top_down_context = torch.cat([global_video_embeds, segment_temporal_context, segment_spatial_context], dim=1)
            final_top_down_context_mask = torch.ones(final_top_down_context.size()[:-1], dtype=torch.long).to(final_top_down_context.device)

            if 'without-top' in self.hierarchical_agg_function:
                merged_query_tokens = self.video_query_tokens.expand(len(global_video_embeds), -1, -1)
            else:
                merged_query_tokens = self.global_region_prompts.expand(len(global_video_embeds), -1, -1)

            if 'final-global-prompts' in self.hierarchical_agg_function:
                global_to_segment_prompts = self.global_to_segment_prompts.expand(global_video_embeds.shape[0], -1, -1)
                global_to_segment_context = global_video_embeds + global_to_segment_prompts

                if 'proj-' in self.hierarchical_agg_function:
                    global_to_segment_context = self.global_segment_proj(global_to_segment_context)

                merged_query_tokens = torch.cat([merged_query_tokens, global_to_segment_context], dim=1)

            global_region_output = self.video_Qformer.bert(
                query_embeds=merged_query_tokens,
                encoder_hidden_states=final_top_down_context,
                encoder_attention_mask=final_top_down_context_mask,
                #output_attentions=True,
                return_dict=True,
            )
            global_region_merged_embeds = global_region_output.last_hidden_state

            if 'keep-without' in self.hierarchical_agg_function:
                global_region_merged_embeds = global_region_merged_embeds
            elif 'final-global-prompts' in self.hierarchical_agg_function:
                global_region_merged_embeds = global_region_merged_embeds[:, :num_region_queries]

            global_region_merged_embeds = self.video_global_proj(global_region_merged_embeds)

            merged_video_embeds = global_video_embeds
            merged_video_embeds_mask = global_video_embeds_mask

        merged_video_embeds = self.llama_proj(merged_video_embeds)

        if 'keep-without' in self.hierarchical_agg_function:
            tmp = merged_video_embeds + self.global_region_embed_weight * global_region_merged_embeds[:, :num_region_queries]
            merged_video_embeds = torch.cat([tmp, global_region_merged_embeds[:, num_region_queries:]], dim=1)
            merged_video_embeds_mask = torch.cat([merged_video_embeds_mask, merged_video_embeds_mask], dim=1)
        else:
            merged_video_embeds = merged_video_embeds + self.global_region_embed_weight * global_region_merged_embeds

        return merged_video_embeds, merged_video_embeds_mask

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        
        frozen_llama_proj = cfg.get("frozen_llama_proj", True)
        frozen_video_Qformer = cfg.get("frozen_video_Qformer", True)
        frozen_audio_Qformer = cfg.get("frozen_audio_Qformer", True)

        llama_proj_model = cfg.get("llama_proj_model", '')
        
        fusion_header_type = cfg.get("fusion_header_type", 'seqTransf')
        max_frame_pos = cfg.get("max_frame_pos", 32)
        fusion_head_layers = cfg.get("fusion_head_layers", 2)
        num_video_query_token =  cfg.get("num_video_query_token", 32)

        equip_audio_branch= cfg.get("equip_audio_branch", True)
        num_audio_query_token =  cfg.get("num_audio_query_token", 8)
        imagebind_ckpt_path = cfg.get("imagebind_ckpt_path", '/mnt/workspace/ckpt')
        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            fusion_header_type=fusion_header_type,
            max_frame_pos=max_frame_pos,
            fusion_head_layers=fusion_head_layers,
            frozen_llama_proj=frozen_llama_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            frozen_audio_Qformer=frozen_audio_Qformer,
            num_video_query_token=num_video_query_token,
            num_audio_query_token = num_audio_query_token,
            imagebind_ckpt_path = imagebind_ckpt_path,
            equip_audio_branch = equip_audio_branch,
            llama_proj_model = llama_proj_model
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        ckpt_path_2 = cfg.get("ckpt_2", "")  
        if ckpt_path_2:
            print("Load second Checkpoint: {}".format(ckpt_path_2))
            ckpt = torch.load(ckpt_path_2, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        return model
