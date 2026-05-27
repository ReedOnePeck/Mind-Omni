import os
import sys
from dataclasses import dataclass
import json
from safetensors.torch import load_file
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import PIL.Image
import torch
import PIL
import numpy as np
from MindOmni_utils.trainer_utils import load_images_to_tensor

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)


from diffusers.image_processor import VaeImageProcessor
from diffusers.models import VQModel
from train_fMRI_tokenizer_perceptual.fMRI_tokenizer_perceptual import VQ_fMRI
from diffusers.utils import replace_example_docstring
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput

from MindOmni_utils.scheduler import Scheduler
from MindOmni_src.tri_modal_transformer import Trimodal_SymmetricTransformer2DModel

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> image = pipe(prompt).images[0]
        ```
"""

def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def prepare_brain_ids(brain_sequence_length: int, device, dtype) -> torch.Tensor:
    """
    为大脑信号 token 创建位置 ID。
    将 brain token 序列视为沿着 't' 轴排列。
    坐标格式: (t, h, w)
    """
    # 1. 创建一个 (brain_seq_len, 3) 的零张量
    brain_ids = torch.zeros(brain_sequence_length, 3, dtype=dtype, device=device)

    # 2. 填充 't' 坐标 (第一个维度)
    #    每个 token 的 t 坐标就是它在序列中的索引
    brain_ids[:, 0] = torch.arange(brain_sequence_length, device=device, dtype=dtype)

    # h 和 w 坐标保持为 0
    return brain_ids


def dedup_consecutive_words(text: str) -> str:
    words = text.split()
    if not words:
        return text

    out = [words[0]]
    for w in words[1:]:
        if w != out[-1]:
            out.append(w)
    return " ".join(out)


def keep_upto_last_period(text: str) -> str:
    # Weired problem
    text = text.replace("is such is", "").replace("such is", "").replace("such as", "").replace("such", "")
    text = text.strip()
    # Fallback to the ASCII period
    idx = -1
    if idx == -1:
        idx = text.rfind(".")
    # If still not found, return original text
    if idx == -1:
        return text
    # Keep everything up to (and including) the last period
    return text[:idx + 1]



@dataclass
class UnifiedPipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    prompts: List[str]
    brain: torch.Tensor



class UnifiedPipeline(DiffusionPipeline):
    def __init__(
            self,
            vqvae: VQModel,
            tokenizer: CLIPTokenizer,
            text_encoder: CLIPTextModelWithProjection,
            brain_tokenizer: VQ_fMRI,
            transformer: Trimodal_SymmetricTransformer2DModel,
            scheduler: Scheduler,
    ):
        super().__init__()
        #self.register_modules(
            #vqvae=vqvae,
            #tokenizer=tokenizer,
            #text_encoder=text_encoder,
            #brain_tokenizer=brain_tokenizer,
            #transformer=transformer,
            #scheduler=scheduler,
        #)
        self.brain_tokenizer = brain_tokenizer
        self.vqvae=vqvae
        self.tokenizer=tokenizer
        self.text_encoder=text_encoder
        self.transformer=transformer
        self.scheduler=scheduler
        self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False)

    @property
    def device(self):
        """
        覆盖基类的 device 属性，使其从核心组件动态推断设备。
        """
        try:
            # next(self.transformer.parameters()) 会获取第一个模型参数
            # .device 就能得到该参数所在的设备
            return next(self.transformer.parameters()).device
        except StopIteration:
            # 如果 transformer 没有任何参数，提供一个备选方案
            try:
                return next(self.vqvae.parameters()).device
            except StopIteration:
                # 如果所有模型都没有参数，返回一个默认的CPU设备
                return torch.device("cpu")

    @property
    def _execution_device(self):
        # 直接调用我们上面定义的 device 属性
        return self.device


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Optional[Union[List[str], str]] = None,
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            image: Optional[Union[torch.Tensor, PIL.Image.Image]] = None,
            brain_data: Optional[torch.Tensor] = None,
            num_brain_token: int = 64,
            num_inference_steps: int = 48,
            guidance_scale: float = 9.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[torch.Generator] = None,
            latents: Optional[torch.IntTensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            negative_encoder_hidden_states: Optional[torch.Tensor] = None,
            output_type="pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            callback_steps: int = 1,
            micro_conditioning_aesthetic_score: int = 6,
            micro_conditioning_crop_coord: Tuple[int, int] = (0, 0),
            temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),
            mask_token_embedding: Optional[str] = None,
            brain_mask_token_path:  Optional[str] = None,
            is_multimodal_decoding:  Optional[str] = None,
            is_brain_to_img_decoding:  Optional[str] = None,
            is_brain_to_text_decoding:  Optional[str] = None,
            is_multimodal_encoding:  Optional[str] = None,
            is_img_to_brain_encoding:  Optional[str] = None,
            is_text_to_brain_encoding:  Optional[str] = None
    ):
        """
                The call function to the pipeline for generation.

                Args:
                    prompt (`str` or `List[str]`, *optional*):
                        The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
                    height (`int`, *optional*, defaults to `self.transformer.config.sample_size * self.vae_scale_factor`):
                        The height in pixels of the generated image.
                    width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                        The width in pixels of the generated image.
                    num_inference_steps (`int`, *optional*, defaults to 16):
                        The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                        expense of slower inference.
                    guidance_scale (`float`, *optional*, defaults to 10.0):
                        A higher guidance scale value encourages the model to generate images closely linked to the text
                        `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
                    negative_prompt (`str` or `List[str]`, *optional*):
                        The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                        pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
                    num_images_per_prompt (`int`, *optional*, defaults to 1):
                        The number of images to generate per prompt.
                    generator (`torch.Generator`, *optional*):
                        A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                        generation deterministic.
                    latents (`torch.IntTensor`, *optional*):
                        Pre-generated tokens representing latent vectors in `self.vqvae`, to be used as inputs for image
                        gneration. If not provided, the starting latents will be completely masked.
                    prompt_embeds (`torch.Tensor`, *optional*):
                        Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                        provided, text embeddings are generated from the `prompt` input argument. A single vector from the
                        pooled and projected final hidden states.
                    encoder_hidden_states (`torch.Tensor`, *optional*):
                        Pre-generated penultimate hidden states from the text encoder providing additional text conditioning.
                    negative_prompt_embeds (`torch.Tensor`, *optional*):
                        Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                        not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
                    negative_encoder_hidden_states (`torch.Tensor`, *optional*):
                        Analogous to `encoder_hidden_states` for the positive prompt.
                    output_type (`str`, *optional*, defaults to `"pil"`):
                        The output format of the generated image. Choose between `PIL.Image` or `np.array`.
                    return_dict (`bool`, *optional*, defaults to `True`):
                        Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                        plain tuple.
                    callback (`Callable`, *optional*):
                        A function that calls every `callback_steps` steps during inference. The function is called with the
                        following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
                    callback_steps (`int`, *optional*, defaults to 1):
                        The frequency at which the `callback` function is called. If not specified, the callback is called at
                        every step.
                    cross_attention_kwargs (`dict`, *optional*):
                        A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                        [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                    micro_conditioning_aesthetic_score (`int`, *optional*, defaults to 6):
                        The targeted aesthetic score according to the laion aesthetic classifier. See
                        https://laion.ai/blog/laion-aesthetics/ and the micro-conditioning section of
                        https://arxiv.org/abs/2307.01952.
                    micro_conditioning_crop_coord (`Tuple[int]`, *optional*, defaults to (0, 0)):
                        The targeted height, width crop coordinates. See the micro-conditioning section of
                        https://arxiv.org/abs/2307.01952.
                    temperature (`Union[int, Tuple[int, int], List[int]]`, *optional*, defaults to (2, 0)):
                        Configures the temperature scheduler on `self.scheduler` see `Scheduler#set_timesteps`.

                Examples:

                Returns:
                    [`~pipelines.pipeline_utils.ImagePipelineOutput`] or `tuple`:
                        If `return_dict` is `True`, [`~pipelines.pipeline_utils.ImagePipelineOutput`] is returned, otherwise a
                        `tuple` is returned where the first element is a list with the generated images.
                """
        # ---------------------------------------------------------------------------------
        # Part 0: 参数校验和初始设置
        # ---------------------------------------------------------------------------------
        if (prompt_embeds is not None and encoder_hidden_states is None) or (
                prompt_embeds is None and encoder_hidden_states is not None
        ):
            raise ValueError("pass either both `prompt_embeds` and `encoder_hidden_states` or neither")

        if (negative_prompt_embeds is not None and negative_encoder_hidden_states is None) or (
                negative_prompt_embeds is None and negative_encoder_hidden_states is not None
        ):
            raise ValueError(
                "pass either both `negatve_prompt_embeds` and `negative_encoder_hidden_states` or neither"
            )
        decoding_encoding_flags = [
            is_multimodal_decoding,
            is_brain_to_img_decoding,
            is_brain_to_text_decoding,
            is_multimodal_encoding,
            is_img_to_brain_encoding,
            is_text_to_brain_encoding
        ]

        # 检查是否所有变量都是None
        if all(flag is None for flag in decoding_encoding_flags):
            print("所有解码和编码标志都是None，执行特定判断...")
            # 确定任务类型
            is_multimodal_decoding = (brain_data is not None) and (image is None) and (prompt is None)
            is_brain_to_img_decoding = (brain_data is not None) and (image is None) and (prompt is not None)
            is_brain_to_text_decoding = (brain_data is not None) and (image is not None) and (prompt is None)

            is_multimodal_encoding = (brain_data is None) and (image is not None) and (prompt is not None)
            is_img_to_brain_encoding = (brain_data is None) and (image is not None) and (prompt is None)
            is_text_to_brain_encoding = (brain_data is None) and (image is None) and (prompt is not None)


        else:
            print("至少有一个解码或编码标志不是None，跳过判断...")



        # =================================================================================
        # ========================== Brain to text and image GENERATION =============================
        # =================================================================================
        if is_multimodal_decoding:
            batch_size = len(brain_data)

            # -----------------------------------------------------------------------------
            # Part 1: 准备 Mask Token 和 Batch Size
            # -----------------------------------------------------------------------------

            #获取图像的mask token id
            img_mask_token_id = self.transformer.config.vocab_size - 1

            #获取文本的mask token id
            mask_token = "<mask>"
            self.tokenizer.add_tokens(mask_token, special_tokens=False)
            clip_mask_id = self.tokenizer.convert_tokens_to_ids(mask_token)
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))

            if mask_token_embedding is not None:
                try:
                    if mask_token_embedding.endswith(".pth"):
                        mask_token_embedding = torch.load(mask_token_embedding)
                    else:
                        mask_token_embedding_path = os.path.join(mask_token_embedding, "mask_token_embedding.pth")
                        assert os.path.exists(
                            mask_token_embedding_path), f"{mask_token_embedding_path} doesn't exists!"
                        mask_token_embedding = torch.load(mask_token_embedding_path)

                    mask_token_embedding = mask_token_embedding.to(self._execution_device,
                                                                   dtype=self.text_encoder.dtype)
                    self.text_encoder.get_input_embeddings().weight.data[clip_mask_id].copy_(mask_token_embedding)

                except Exception as e:
                    print(f"Error loading mask token embedding: {e}")
                    print("Using random initialized mask token embedding")
                    mask_token_embedding = None

            text_mask_token_id = clip_mask_id

            # -----------------------------------------------------------------------------
            # Part 2: 准备brain, 初始化图像和文本 Latents
            # -----------------------------------------------------------------------------

            shape = (batch_size * num_images_per_prompt, height // self.vae_scale_factor, width // self.vae_scale_factor)
            image_latents = torch.full(shape, img_mask_token_id, dtype=torch.long, device=self._execution_device)

            text_latents = torch.ones((batch_size, self.tokenizer.model_max_length), dtype=torch.long, device=self._execution_device) * text_mask_token_id
            question_len = [0] * batch_size

            brain_latents,_ = self.brain_tokenizer.forward_for_inference(brain_data.to(self._execution_device))

            # -----------------------------------------------------------------------------
            # Part 3: 扩散循环 (Denoising Loop)
            # -----------------------------------------------------------------------------
            self.scheduler.set_timesteps(num_inference_steps, temperature, self._execution_device)
            num_warmup_steps = len(self.scheduler.timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, timestep in enumerate(self.scheduler.timesteps):

                    outputs = self.text_encoder(text_latents, return_dict=True, output_hidden_states=True)
                    prompt_embeds = outputs.text_embeds
                    encoder_hidden_states = outputs.hidden_states[-2]  # 文本自身的 hidden states

                    micro_conds = torch.tensor(
                        [width, height, micro_conditioning_crop_coord[0], micro_conditioning_crop_coord[1],
                         micro_conditioning_aesthetic_score],
                        device=self._execution_device, dtype=encoder_hidden_states.dtype,
                    ).unsqueeze(0).expand(batch_size, -1)


                    # 准备位置编码
                    img_ids = _prepare_latent_image_ids(
                        image_latents.shape[0], image_latents.shape[-2], image_latents.shape[-1],
                        image_latents.device, torch.long  # ids should be long
                    )
                    txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(device=image_latents.device, dtype=torch.long)
                    brain_ids = prepare_brain_ids(brain_latents.shape[1], device=image_latents.device,dtype=torch.long)

                    img_logits, text_logits, _   = self.transformer(
                        hidden_states=image_latents,
                        encoder_hidden_states=encoder_hidden_states,
                        thirdmodal_hidden_states=brain_latents,
                        micro_conds=micro_conds,
                        pooled_projections=prompt_embeds,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        thirdmodal_ids=brain_ids,
                        timestep=torch.tensor([timestep / num_inference_steps], device=self._execution_device),
                    )

                    # 调用调度器更新文本latent
                    self.scheduler.config.mask_token_id = text_mask_token_id
                    text_latents = self.scheduler.step(
                        model_output=text_logits,
                        timestep=timestep,
                        sample=text_latents,
                        generator=generator,
                    ).prev_sample

                    self.scheduler.config.mask_token_id = img_mask_token_id
                    image_latents = self.scheduler.step(
                        model_output=img_logits,
                        timestep=timestep,
                        sample=image_latents,
                        generator=generator,
                    ).prev_sample

                    # 更新进度条和回调
                    if i == len(self.scheduler.timesteps) - 1 or (
                            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

            # -----------------------------------------------------------------------------
            #Part 4: 解码和后处理
            # -----------------------------------------------------------------------------
            #图像后处理
            needs_upcasting = self.vqvae.dtype == torch.float16 and self.vqvae.config.force_upcast
            if needs_upcasting: self.vqvae.float()

            images = self.vqvae.decode(image_latents, force_not_quantize=True,shape=(batch_size, height // self.vae_scale_factor, width // self.vae_scale_factor,
                       self.vqvae.config.latent_channels)).sample.clip(0, 1)
            images = self.image_processor.postprocess(images, output_type)

            if needs_upcasting: self.vqvae.half()

            #文本后处理
            decoded_input_ids = text_latents

            prompts = []
            for i, p_ids in enumerate(decoded_input_ids):
                q_len = question_len[i]
                decoded_text = self.tokenizer.decode(p_ids.tolist()[q_len:], skip_special_tokens=True)
                prompts.append(keep_upto_last_period(dedup_consecutive_words(decoded_text)))

            #self.maybe_free_model_hooks()

            return UnifiedPipelineOutput(images=images, prompts=prompts, brain = brain_data)




        # =================================================================================
        # ========================== Brain to image GENERATION =============================
        # =================================================================================
        if is_brain_to_img_decoding:
            batch_size = len(brain_data)
            # -----------------------------------------------------------------------------
            # Part 1: 准备 Mask Token 和 Batch Size
            # -----------------------------------------------------------------------------

            # 获取图像的mask token id
            img_mask_token_id = self.transformer.config.vocab_size - 1

            # 获取文本的mask token id
            mask_token = "<mask>"
            self.tokenizer.add_tokens(mask_token, special_tokens=False)
            clip_mask_id = self.tokenizer.convert_tokens_to_ids(mask_token)
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))

            if mask_token_embedding is not None:
                try:
                    if mask_token_embedding.endswith(".pth"):
                        mask_token_embedding = torch.load(mask_token_embedding)
                    else:
                        mask_token_embedding_path = os.path.join(mask_token_embedding, "mask_token_embedding.pth")
                        assert os.path.exists(
                            mask_token_embedding_path), f"{mask_token_embedding_path} doesn't exists!"
                        mask_token_embedding = torch.load(mask_token_embedding_path)

                    mask_token_embedding = mask_token_embedding.to(self._execution_device,
                                                                   dtype=self.text_encoder.dtype)
                    self.text_encoder.get_input_embeddings().weight.data[clip_mask_id].copy_(mask_token_embedding)

                except Exception as e:
                    print(f"Error loading mask token embedding: {e}")
                    print("Using random initialized mask token embedding")
                    mask_token_embedding = None

            text_mask_token_id = clip_mask_id

            # -----------------------------------------------------------------------------
            # Part 2: 准备brain, 初始化图像和文本 Latents
            # -----------------------------------------------------------------------------

            shape = (batch_size * num_images_per_prompt, height // self.vae_scale_factor, width // self.vae_scale_factor)
            image_latents = torch.full(shape, img_mask_token_id, dtype=torch.long, device=self._execution_device)

            text_latents = torch.ones((batch_size, self.tokenizer.model_max_length), dtype=torch.long,device=self._execution_device) * text_mask_token_id
            question_len = [0] * batch_size
            outputs = self.text_encoder(text_latents, return_dict=True, output_hidden_states=True)
            prompt_embeds = outputs.text_embeds
            encoder_hidden_states = outputs.hidden_states[-2]  # 文本自身的 hidden states

            brain_latents, _ = self.brain_tokenizer.forward_for_inference(brain_data.to(self._execution_device))

            micro_conds = torch.tensor(
                [width, height, micro_conditioning_crop_coord[0], micro_conditioning_crop_coord[1],
                 micro_conditioning_aesthetic_score],
                device=self._execution_device, dtype=encoder_hidden_states.dtype,
            ).unsqueeze(0).expand(batch_size, -1)

            # 准备位置编码
            img_ids = _prepare_latent_image_ids(
                image_latents.shape[0], image_latents.shape[-2], image_latents.shape[-1],
                image_latents.device, torch.long  # ids should be long
            )
            txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(device=image_latents.device, dtype=torch.long)
            brain_ids = prepare_brain_ids(brain_latents.shape[1], device=image_latents.device, dtype=torch.long)


            # 创建 Attention Mask (屏蔽文本模态)
            image_seq_len_after_downsample = 256
            text_mask = torch.ones(batch_size, encoder_hidden_states.shape[1],  dtype=torch.float32) * -torch.inf
            image_mask = torch.zeros(batch_size, image_seq_len_after_downsample, dtype=torch.float32)
            brain_mask = torch.zeros(batch_size, brain_latents.shape[1], dtype=torch.float32)
            attention_mask = torch.cat([text_mask, image_mask, brain_mask], dim=1).unsqueeze(1).unsqueeze(1).to(device=image_latents.device)

            # -----------------------------------------------------------------------------
            # Part 3: 扩散循环 (Denoising Loop)
            # -----------------------------------------------------------------------------
            self.scheduler.set_timesteps(num_inference_steps, temperature, self._execution_device)
            num_warmup_steps = len(self.scheduler.timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, timestep in enumerate(self.scheduler.timesteps):

                    img_logits, _, _ = self.transformer(
                        hidden_states=image_latents,
                        encoder_hidden_states=encoder_hidden_states,
                        thirdmodal_hidden_states=brain_latents,
                        micro_conds=micro_conds,
                        pooled_projections=prompt_embeds,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        thirdmodal_ids=brain_ids,
                        timestep=torch.tensor([timestep / num_inference_steps], device=self._execution_device),
                        attention_mask=attention_mask,
                    )


                    self.scheduler.config.mask_token_id = img_mask_token_id
                    image_latents = self.scheduler.step(
                        model_output=img_logits,
                        timestep=timestep,
                        sample=image_latents,
                        generator=generator,
                    ).prev_sample

                    # 更新进度条和回调
                    if i == len(self.scheduler.timesteps) - 1 or (
                            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

            # -----------------------------------------------------------------------------
            # Part 4: 解码和后处理
            # -----------------------------------------------------------------------------
            # 图像后处理
            needs_upcasting = self.vqvae.dtype == torch.float16 and self.vqvae.config.force_upcast
            if needs_upcasting: self.vqvae.float()

            images = self.vqvae.decode(image_latents, force_not_quantize=True, shape=(
            batch_size, height // self.vae_scale_factor, width // self.vae_scale_factor,
            self.vqvae.config.latent_channels)).sample.clip(0, 1)
            images = self.image_processor.postprocess(images, output_type)

            if needs_upcasting: self.vqvae.half()

            # 文本后处理
            prompts = ["Placeholder"] * batch_size

            # self.maybe_free_model_hooks()

            return UnifiedPipelineOutput(images=images, prompts=prompts, brain=brain_data)




        # =================================================================================
        # ========================== Brain to text GENERATION =============================
        # =================================================================================
        if is_brain_to_text_decoding:
            batch_size = len(brain_data)
            # -----------------------------------------------------------------------------
            # Part 1: 准备 Mask Token 和 Batch Size
            # -----------------------------------------------------------------------------

            # 获取图像的mask token id
            img_mask_token_id = self.transformer.config.vocab_size - 1

            # 获取文本的mask token id
            mask_token = "<mask>"
            self.tokenizer.add_tokens(mask_token, special_tokens=False)
            clip_mask_id = self.tokenizer.convert_tokens_to_ids(mask_token)
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))

            if mask_token_embedding is not None:
                try:
                    if mask_token_embedding.endswith(".pth"):
                        mask_token_embedding = torch.load(mask_token_embedding)
                    else:
                        mask_token_embedding_path = os.path.join(mask_token_embedding, "mask_token_embedding.pth")
                        assert os.path.exists(
                            mask_token_embedding_path), f"{mask_token_embedding_path} doesn't exists!"
                        mask_token_embedding = torch.load(mask_token_embedding_path)

                    mask_token_embedding = mask_token_embedding.to(self._execution_device,
                                                                   dtype=self.text_encoder.dtype)
                    self.text_encoder.get_input_embeddings().weight.data[clip_mask_id].copy_(mask_token_embedding)

                except Exception as e:
                    print(f"Error loading mask token embedding: {e}")
                    print("Using random initialized mask token embedding")
                    mask_token_embedding = None

            text_mask_token_id = clip_mask_id

            # -----------------------------------------------------------------------------
            # Part 2: 准备brain, 初始化图像和文本 Latents
            # -----------------------------------------------------------------------------
            shape = (batch_size * num_images_per_prompt, height // self.vae_scale_factor, width // self.vae_scale_factor)
            image_latents = torch.full(shape, img_mask_token_id, dtype=torch.long, device=self._execution_device)

            text_latents = torch.ones((batch_size, self.tokenizer.model_max_length), dtype=torch.long,
                                      device=self._execution_device) * text_mask_token_id
            question_len = [0] * batch_size
            outputs = self.text_encoder(text_latents, return_dict=True, output_hidden_states=True)
            prompt_embeds = outputs.text_embeds
            encoder_hidden_states = outputs.hidden_states[-2]  # 文本自身的 hidden states

            brain_latents, _ = self.brain_tokenizer.forward_for_inference(brain_data.to(self._execution_device))

            micro_conds = torch.tensor(
                [width, height, micro_conditioning_crop_coord[0], micro_conditioning_crop_coord[1],
                 micro_conditioning_aesthetic_score],
                device=self._execution_device, dtype=encoder_hidden_states.dtype,
            ).unsqueeze(0).expand(batch_size, -1)

            # 准备位置编码
            img_ids = _prepare_latent_image_ids(
                image_latents.shape[0], image_latents.shape[-2], image_latents.shape[-1],
                image_latents.device, torch.long  # ids should be long
            )
            txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(device=image_latents.device, dtype=torch.long)
            brain_ids = prepare_brain_ids(brain_latents.shape[1], device=image_latents.device, dtype=torch.long)

            # 创建 Attention Mask (屏蔽图像模态)
            image_seq_len_after_downsample = 256
            text_mask = torch.zeros(batch_size, encoder_hidden_states.shape[1], dtype=torch.float32)
            image_mask = torch.ones(batch_size, image_seq_len_after_downsample, dtype=torch.float32) * -torch.inf
            brain_mask = torch.zeros(batch_size, brain_latents.shape[1], dtype=torch.float32)
            attention_mask = torch.cat([text_mask, image_mask, brain_mask], dim=1).unsqueeze(1).unsqueeze(1).to(device=image_latents.device)

            # -----------------------------------------------------------------------------
            # Part 3: 扩散循环 (Denoising Loop)
            # -----------------------------------------------------------------------------
            self.scheduler.set_timesteps(num_inference_steps, temperature, self._execution_device)
            num_warmup_steps = len(self.scheduler.timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, timestep in enumerate(self.scheduler.timesteps):

                    _, text_logits, _ = self.transformer(
                        hidden_states=image_latents,
                        encoder_hidden_states=encoder_hidden_states,
                        thirdmodal_hidden_states=brain_latents,
                        micro_conds=micro_conds,
                        pooled_projections=prompt_embeds,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        thirdmodal_ids=brain_ids,
                        timestep=torch.tensor([timestep / num_inference_steps], device=self._execution_device),
                        attention_mask=attention_mask,
                    )

                    self.scheduler.config.mask_token_id = text_mask_token_id
                    text_latents = self.scheduler.step(
                        model_output=text_logits,
                        timestep=timestep,
                        sample=text_latents,
                        generator=generator,
                    ).prev_sample

                    # 更新进度条和回调
                    if i == len(self.scheduler.timesteps) - 1 or (
                            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

            # -----------------------------------------------------------------------------
            # Part 4: 解码和后处理
            # -----------------------------------------------------------------------------
            images = np.array([1,2,3])

            # 文本后处理
            decoded_input_ids = text_latents

            prompts = []
            for i, p_ids in enumerate(decoded_input_ids):
                q_len = question_len[i]
                decoded_text = self.tokenizer.decode(p_ids.tolist()[q_len:], skip_special_tokens=True)
                prompts.append(keep_upto_last_period(dedup_consecutive_words(decoded_text)))

            # self.maybe_free_model_hooks()

            return UnifiedPipelineOutput(images=images, prompts=prompts, brain=brain_data)



        # =================================================================================
        # ========================== image and text to Brain GENERATION =============================
        # =================================================================================
        if is_multimodal_encoding :
            #-----------------------------------------------------------------------------
            # Part 1: 准备 image 和 text的embedding
            # -----------------------------------------------------------------------------
            batch_size = len(image)
            image_latents = self.vqvae.quantize(self.vqvae.encode(image.to(self._execution_device, dtype=self.vqvae.dtype)).latents
                                )[2][2].reshape(batch_size, height // self.vae_scale_factor, width // self.vae_scale_factor)

            # 确保形状是 (B, H_lat, W_lat)
            if image_latents.ndim == 4:
                # 如果 VQVAE 输出 (B, 1, H, W), 则挤压掉通道维度
                image_latents = image_latents.squeeze(1)



            input_ids = self.tokenizer(
                prompt, return_tensors="pt", padding="max_length",
                truncation=True, add_special_tokens=True, max_length=77,
            ).input_ids.to(self._execution_device)
            outputs = self.text_encoder(input_ids, return_dict=True, output_hidden_states=True)
            prompt_embeds = outputs.text_embeds
            encoder_hidden_states = outputs.hidden_states[-2]

            prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1)
            encoder_hidden_states = encoder_hidden_states.repeat(num_images_per_prompt, 1, 1)

            # -----------------------------------------------------------------------------
            # Part 2: 准备 brain mask
            # -----------------------------------------------------------------------------
            brain_vocab = self.brain_tokenizer.quantize.embedding.weight.data    #tensor  (128,16)
            brain_mask_id = int(brain_vocab.shape[0])

            self.scheduler.config.mask_token_id = brain_mask_id

            #brain_mask_token = self.transformer.fmri_mask_token.data             #tensor  (16,)

            brain_mask_token = torch.load(brain_mask_token_path, map_location="cpu").to(self._execution_device)
            updated_brain_vocab = torch.cat([brain_vocab, brain_mask_token.unsqueeze(0)], dim=0).to(self._execution_device)

            brain_latent_ids = torch.ones((batch_size, num_brain_token), dtype=torch.long, device=self._execution_device) * brain_mask_id
            brain_latents = updated_brain_vocab[brain_latent_ids]


            micro_conds = torch.tensor(
                [width, height, micro_conditioning_crop_coord[0], micro_conditioning_crop_coord[1],
                 micro_conditioning_aesthetic_score],
                device=self._execution_device, dtype=encoder_hidden_states.dtype,
            ).unsqueeze(0).expand( batch_size, -1)

            # 准备位置编码
            img_ids = _prepare_latent_image_ids(
                image_latents.shape[0], image_latents.shape[-2], image_latents.shape[-1],
                image_latents.device, torch.long  # ids should be long
            )
            txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(device=image_latents.device, dtype=torch.long)
            brain_ids = prepare_brain_ids(brain_latents.shape[1], device=image_latents.device, dtype=torch.long)

            # -----------------------------------------------------------------------------
            # Part 3: 扩散循环 (Denoising Loop)
            # -----------------------------------------------------------------------------
            self.scheduler.set_timesteps(num_inference_steps, temperature, self._execution_device)
            num_warmup_steps = len(self.scheduler.timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, timestep in enumerate(self.scheduler.timesteps):
                    _, _, brain_logits = self.transformer(
                        hidden_states=image_latents,
                        encoder_hidden_states=encoder_hidden_states,
                        thirdmodal_hidden_states=brain_latents,
                        micro_conds=micro_conds,
                        pooled_projections=prompt_embeds,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        thirdmodal_ids=brain_ids,
                        timestep=torch.tensor([timestep / num_inference_steps], device=self._execution_device),
                    )

                    brain_latent_ids = self.scheduler.step(
                        model_output=brain_logits,
                        timestep=timestep,
                        sample=brain_latent_ids,
                        generator=generator,
                    ).prev_sample

                    brain_latents = updated_brain_vocab[brain_latent_ids]


                    # 更新进度条和回调
                    if i == len(self.scheduler.timesteps) - 1 or (
                            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

            # -----------------------------------------------------------------------------
            # Part 4: 解码和后处理
            # -----------------------------------------------------------------------------
            images = np.array([1,2,3])
            prompts = ["Placeholder"] * batch_size
            brain = brain_latents

            return UnifiedPipelineOutput(images=images, prompts=prompts, brain=brain)



        # =================================================================================
        # ========================== image to Brain GENERATION =============================
        # =================================================================================
        if is_img_to_brain_encoding:
            # -----------------------------------------------------------------------------
            # Part 1: 准备 image 和 text的embedding
            # -----------------------------------------------------------------------------
            batch_size = len(image)
            image_latents = self.vqvae.quantize(self.vqvae.encode(image.to(self._execution_device, dtype=self.vqvae.dtype)).latents
                                )[2][2].reshape(batch_size, height // self.vae_scale_factor,
                                                width // self.vae_scale_factor)

            # 确保形状是 (B, H_lat, W_lat)
            if image_latents.ndim == 4:
                # 如果 VQVAE 输出 (B, 1, H, W), 则挤压掉通道维度
                image_latents = image_latents.squeeze(1)


            # 获取文本的mask token id
            mask_token = "<mask>"
            self.tokenizer.add_tokens(mask_token, special_tokens=False)
            clip_mask_id = self.tokenizer.convert_tokens_to_ids(mask_token)
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))

            if mask_token_embedding is not None:
                try:
                    if mask_token_embedding.endswith(".pth"):
                        mask_token_embedding = torch.load(mask_token_embedding)
                    else:
                        mask_token_embedding_path = os.path.join(mask_token_embedding, "mask_token_embedding.pth")
                        assert os.path.exists(
                            mask_token_embedding_path), f"{mask_token_embedding_path} doesn't exists!"
                        mask_token_embedding = torch.load(mask_token_embedding_path)

                    mask_token_embedding = mask_token_embedding.to(self._execution_device,
                                                                   dtype=self.text_encoder.dtype)
                    self.text_encoder.get_input_embeddings().weight.data[clip_mask_id].copy_(mask_token_embedding)

                except Exception as e:
                    print(f"Error loading mask token embedding: {e}")
                    print("Using random initialized mask token embedding")
                    mask_token_embedding = None

            text_mask_token_id = clip_mask_id

            text_latents = torch.ones((batch_size, self.tokenizer.model_max_length), dtype=torch.long,
                                      device=self._execution_device) * text_mask_token_id
            question_len = [0] * batch_size
            outputs = self.text_encoder(text_latents, return_dict=True, output_hidden_states=True)
            prompt_embeds = outputs.text_embeds
            encoder_hidden_states = outputs.hidden_states[-2]  # 文本自身的 hidden states

            # -----------------------------------------------------------------------------
            # Part 2: 准备 brain mask
            # -----------------------------------------------------------------------------
            brain_vocab = self.brain_tokenizer.quantize.embedding.weight.data  # tensor  (128,16)
            brain_mask_id = int(brain_vocab.shape[0])

            self.scheduler.config.mask_token_id = brain_mask_id

            # brain_mask_token = self.transformer.fmri_mask_token.data             #tensor  (16,)

            brain_mask_token = torch.load(brain_mask_token_path, map_location="cpu").to(self._execution_device)
            updated_brain_vocab = torch.cat([brain_vocab, brain_mask_token.unsqueeze(0)], dim=0).to(
                self._execution_device)

            brain_latent_ids = torch.ones((batch_size, num_brain_token), dtype=torch.long,
                                          device=self._execution_device) * brain_mask_id
            brain_latents = updated_brain_vocab[brain_latent_ids]

            micro_conds = torch.tensor(
                [width, height, micro_conditioning_crop_coord[0], micro_conditioning_crop_coord[1],
                 micro_conditioning_aesthetic_score],
                device=self._execution_device, dtype=encoder_hidden_states.dtype,
            ).unsqueeze(0).expand(batch_size, -1)

            # 准备位置编码
            img_ids = _prepare_latent_image_ids(
                image_latents.shape[0], image_latents.shape[-2], image_latents.shape[-1],
                image_latents.device, torch.long  # ids should be long
            )
            txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(device=image_latents.device,
                                                                        dtype=torch.long)
            brain_ids = prepare_brain_ids(brain_latents.shape[1], device=image_latents.device, dtype=torch.long)

            # 创建 Attention Mask (屏蔽文本模态)
            image_seq_len_after_downsample = 256
            text_mask = torch.ones(batch_size, encoder_hidden_states.shape[1], dtype=torch.float32) * -torch.inf
            image_mask = torch.zeros(batch_size, image_seq_len_after_downsample, dtype=torch.float32)
            brain_mask = torch.zeros(batch_size, brain_latents.shape[1], dtype=torch.float32)
            attention_mask = torch.cat([text_mask, image_mask, brain_mask], dim=1).unsqueeze(1).unsqueeze(1).to(device=image_latents.device)

            # -----------------------------------------------------------------------------
            # Part 3: 扩散循环 (Denoising Loop)
            # -----------------------------------------------------------------------------
            self.scheduler.set_timesteps(num_inference_steps, temperature, self._execution_device)
            num_warmup_steps = len(self.scheduler.timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, timestep in enumerate(self.scheduler.timesteps):
                    _, _, brain_logits = self.transformer(
                        hidden_states=image_latents,
                        encoder_hidden_states=encoder_hidden_states,
                        thirdmodal_hidden_states=brain_latents,
                        micro_conds=micro_conds,
                        pooled_projections=prompt_embeds,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        thirdmodal_ids=brain_ids,
                        timestep=torch.tensor([timestep / num_inference_steps], device=self._execution_device),
                        attention_mask=attention_mask,
                    )

                    brain_latent_ids = self.scheduler.step(
                        model_output=brain_logits,
                        timestep=timestep,
                        sample=brain_latent_ids,
                        generator=generator,
                    ).prev_sample

                    brain_latents = updated_brain_vocab[brain_latent_ids]

                    # 更新进度条和回调
                    if i == len(self.scheduler.timesteps) - 1 or (
                            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

            # -----------------------------------------------------------------------------
            # Part 4: 解码和后处理
            # -----------------------------------------------------------------------------
            images = self.image_processor.pt_to_numpy(self.image_processor.preprocess(image))
            prompts = ["Placeholder"] * batch_size
            brain = brain_latents

            return UnifiedPipelineOutput(images=images, prompts=prompts, brain=brain)



        # =================================================================================
        # ========================== text to Brain GENERATION =============================
        # =================================================================================
        if is_text_to_brain_encoding:
            # -----------------------------------------------------------------------------
            # Part 1: 准备 image 和 text的embedding
            # -----------------------------------------------------------------------------
            batch_size = len(prompt)

            # 获取图像的mask token id
            img_mask_token_id = self.transformer.config.vocab_size - 1
            shape = (batch_size * num_images_per_prompt, height // self.vae_scale_factor, width // self.vae_scale_factor)
            image_latents = torch.full(shape, img_mask_token_id, dtype=torch.long, device=self._execution_device)



            input_ids = self.tokenizer(
                prompt, return_tensors="pt", padding="max_length",
                truncation=True, add_special_tokens=True, max_length=77,
            ).input_ids.to(self._execution_device)
            outputs = self.text_encoder(input_ids, return_dict=True, output_hidden_states=True)
            prompt_embeds = outputs.text_embeds
            encoder_hidden_states = outputs.hidden_states[-2]

            prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1)
            encoder_hidden_states = encoder_hidden_states.repeat(num_images_per_prompt, 1, 1)

            # -----------------------------------------------------------------------------
            # Part 2: 准备 brain mask
            # -----------------------------------------------------------------------------
            brain_vocab = self.brain_tokenizer.quantize.embedding.weight.data  # tensor  (128,16)
            brain_mask_id = int(brain_vocab.shape[0])

            self.scheduler.config.mask_token_id = brain_mask_id

            # brain_mask_token = self.transformer.fmri_mask_token.data             #tensor  (16,)

            brain_mask_token = torch.load(brain_mask_token_path, map_location="cpu").to(self._execution_device)
            updated_brain_vocab = torch.cat([brain_vocab, brain_mask_token.unsqueeze(0)], dim=0).to(
                self._execution_device)

            brain_latent_ids = torch.ones((batch_size, num_brain_token), dtype=torch.long,
                                          device=self._execution_device) * brain_mask_id
            brain_latents = updated_brain_vocab[brain_latent_ids]

            micro_conds = torch.tensor(
                [width, height, micro_conditioning_crop_coord[0], micro_conditioning_crop_coord[1],
                 micro_conditioning_aesthetic_score],
                device=self._execution_device, dtype=encoder_hidden_states.dtype,
            ).unsqueeze(0).expand(batch_size, -1)

            # 准备位置编码
            img_ids = _prepare_latent_image_ids(
                image_latents.shape[0], image_latents.shape[-2], image_latents.shape[-1],
                image_latents.device, torch.long  # ids should be long
            )
            txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(device=image_latents.device,
                                                                        dtype=torch.long)
            brain_ids = prepare_brain_ids(brain_latents.shape[1], device=image_latents.device, dtype=torch.long)


            # 创建 Attention Mask (屏蔽图像模态)
            image_seq_len_after_downsample = 256
            text_mask = torch.zeros(batch_size, encoder_hidden_states.shape[1], dtype=torch.float32)
            image_mask = torch.ones(batch_size, image_seq_len_after_downsample, dtype=torch.float32) * -torch.inf
            brain_mask = torch.zeros(batch_size, brain_latents.shape[1], dtype=torch.float32)
            attention_mask = torch.cat([text_mask, image_mask, brain_mask], dim=1).unsqueeze(1).unsqueeze(1).to(
                device=image_latents.device)

            # -----------------------------------------------------------------------------
            # Part 3: 扩散循环 (Denoising Loop)
            # -----------------------------------------------------------------------------
            self.scheduler.set_timesteps(num_inference_steps, temperature, self._execution_device)
            num_warmup_steps = len(self.scheduler.timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, timestep in enumerate(self.scheduler.timesteps):
                    _, _, brain_logits = self.transformer(
                        hidden_states=image_latents,
                        encoder_hidden_states=encoder_hidden_states,
                        thirdmodal_hidden_states=brain_latents,
                        micro_conds=micro_conds,
                        pooled_projections=prompt_embeds,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        thirdmodal_ids=brain_ids,
                        timestep=torch.tensor([timestep / num_inference_steps], device=self._execution_device),
                        attention_mask=attention_mask,
                    )

                    brain_latent_ids = self.scheduler.step(
                        model_output=brain_logits,
                        timestep=timestep,
                        sample=brain_latent_ids,
                        generator=generator,
                    ).prev_sample

                    brain_latents = updated_brain_vocab[brain_latent_ids]

                    # 更新进度条和回调
                    if i == len(self.scheduler.timesteps) - 1 or (
                            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

            # -----------------------------------------------------------------------------
            # Part 4: 解码和后处理
            # -----------------------------------------------------------------------------
            images = np.array([1,1,1])
            prompts = ["Placeholder"] * batch_size
            brain = brain_latents

            return UnifiedPipelineOutput(images=images, prompts=prompts, brain=brain)






if __name__ == '__main__':
    from train_stage1.train_mind_omni_stage1 import load_pretrained_weights_for_trimodal_model

    device = 'cuda:5'
    tokenizer = CLIPTokenizer.from_pretrained("/nfs/diskstation/DataStation/ChangdeDu/Muddit/tokenizer")
    text_encoder = CLIPTextModelWithProjection.from_pretrained("/nfs/diskstation/DataStation/ChangdeDu/Muddit/text_encoder")
    text_encoder.requires_grad_(False)
    text_encoder = text_encoder.to(device)


    vq_model = VQModel.from_pretrained("/nfs/diskstation/DataStation/ChangdeDu/Muddit/vqvae")
    vq_model.requires_grad_(False)
    vq_model = vq_model.to(device)

    brain_vae = VQ_fMRI.from_pretrained(
        "/nfs/diskstation/DataStation/ChangdeDu/LYZ/UniBrain/fMRI_tokenizer/train_with_semantic/token_concat_codebook_size_128_code_dim_16_num_token_64/checkpoint-9500/VQ_fMRI")
    brain_vae.requires_grad_(False)
    brain_vae = brain_vae.to(device)



    model = load_pretrained_weights_for_trimodal_model(
                                                        config_path="/nfs/diskstation/DataStation/ChangdeDu/Muddit/512/transformer/config.json",
                                                        pretrained_weights_path="/nfs/diskstation/DataStation/ChangdeDu/Muddit/512/transformer/diffusion_pytorch_model.safetensors",
                                                        third_modal_codebook_size=128,
                                                        third_modal_token_dim=16
                                                        )
    model.requires_grad_(False)
    model = model.to(device)

    scheduler = Scheduler.from_pretrained("/nfs/diskstation/DataStation/ChangdeDu/Muddit/scheduler/")

    pipe = UnifiedPipeline(
            vqvae=vq_model,
            tokenizer=tokenizer,
            brain_tokenizer = brain_vae,
            text_encoder=text_encoder,
            transformer=model,
            scheduler=scheduler
        )
    #pipe.to(device)
    #pipe.brain_tokenizer.to(device)


    prompts = [
            "A man in a black wetsuit is kiteboarding, soaring high above the ocean waves. He's upside down, with his board displaying vibrant graphics. Below him, a surfer in a black wetsuit watches the kiteboarder's impressive aerial maneuver. The backdrop features a quaint coastal town with white houses and a gray sky, casting a serene atmosphere over the dynamic scene." ,
            "A fluffy black cat with striking green eyes sits on the rim of a white toilet, curiously gazing at the open bathroom door.The bathroom features a white bathtub with a yellow bath mat, a white shower curtain, and a blue object on the floor.The scene is illuminated by natural light, creating a cozy atmosphere." ,
            "A person slices a pizza on a wooden board, with a bowl of salad, two wine glasses, and bottles on a wooden table.The scene is illuminated by warm lighting, creating a cozy atmosphere.",
            "A blue Pepsi cup sits on a wooden table, next to a plate with a grilled sandwich, a side salad, and a tray of fresh vegetables.The background features a person and chairs, suggesting a casual dining setting.The lighting is bright, indicating daytime.",
            "A bustling city intersection features a white and blue ice cream truck driving by, with a man in a white shirt and blue jeans crossing the street.The scene is illuminated by bright sunlight, casting shadows on the asphalt.Surrounding buildings and vehicles add to the urban atmosphere."

        ]

    image_path_or_dir = ["/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_imgs/46002.png","/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_imgs/48617.png","/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_imgs/44980.png","/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_imgs/32625.png","/nfs/diskstation/DataStation/public_dataset/NSD_complete/NSD_imgs/06431.png"]

    images = load_images_to_tensor(image_path_or_dir, target_size=(512, 512))


    fmri_data = torch.randn(4, 16127)


    #解码
    output1 = pipe(
            brain_data=fmri_data,
            num_brain_token=64,
            height=512,
            width=512,
            num_inference_steps=64,
            mask_token_embedding='/nfs/diskstation/DataStation/ChangdeDu/Muddit/1024/mask_token_embedding.pth',
            brain_mask_token_path=os.path.join("/nfs/diskstation/DataStation/ChangdeDu/LYZ/UniBrain/train_stage1/" , "fmri_mask_embedding.pt"),
            generator=torch.manual_seed(42)
        )


    output2 = pipe(
            prompt=prompts,
            image=images,
            height=512,
            width=512,
            num_inference_steps=64,
            mask_token_embedding='/nfs/diskstation/DataStation/ChangdeDu/Muddit/1024/mask_token_embedding.pth',
            brain_mask_token_path=os.path.join("/nfs/diskstation/DataStation/ChangdeDu/LYZ/UniBrain/train_stage1/" , "fmri_mask_embedding.pt"),
            generator=torch.manual_seed(42)
        )





