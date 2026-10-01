# SPDX-License-Identifier: GPL-3.0-or-later
#
# Single-step VideoMaMa inference pipeline.
#
# Taken from Sammie-Roto 2 (videomama/pipeline_svd_mask_numpy.py,
# https://github.com/Zarxrax/Sammie-Roto-2, GPL-3.0), with the PySide6 import
# removed so it runs headless inside ComfyUI. Model: VideoMaMa
# (https://github.com/cvlab-kaist/VideoMaMa) on top of Stable Video Diffusion XT.

import numpy as np
import torch

from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel


class VideoInferencePipeline:
    """Loads the SVD VAE + VideoMaMa UNet once and runs single-step matting on a frame window."""

    def __init__(self, base_model_path: str, unet_checkpoint_path: str, device: str = "cuda",
                 weight_dtype: torch.dtype = torch.float16, enable_model_cpu_offload: bool = False,
                 vae_encode_chunk_size: int = 1, attention_mode: str = "auto",
                 enable_vae_tiling: bool = False, enable_vae_slicing: bool = True):
        self.device = torch.device(device)
        self.weight_dtype = weight_dtype
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.vae_encode_chunk_size = max(1, int(vae_encode_chunk_size))

        try:
            self.vae = AutoencoderKLTemporalDecoder.from_pretrained(base_model_path, subfolder="vae", variant="fp16")
            self.unet = UNetSpatioTemporalConditionModel.from_pretrained(unet_checkpoint_path, subfolder="unet")
        except Exception as e:
            raise IOError(f"Fatal error loading VideoMaMa models: {e}")

        self.vae.eval()
        self.unet.eval()
        self._apply_attention_optimization(attention_mode)

        if enable_vae_slicing:
            try:
                self.vae.enable_slicing()
            except (AttributeError, NotImplementedError):
                pass
        if enable_vae_tiling:
            try:
                self.vae.enable_tiling()
            except (AttributeError, NotImplementedError):
                pass

        target = "cpu" if self.enable_model_cpu_offload else self.device
        self.vae.to(target, dtype=self.weight_dtype)
        self.unet.to(target, dtype=self.weight_dtype)

    def _apply_attention_optimization(self, attention_mode: str):
        if attention_mode == "none":
            return
        if attention_mode in ("auto", "xformers"):
            try:
                import xformers  # noqa: F401
                self.unet.enable_xformers_memory_efficient_attention()
                return
            except Exception:
                pass
        if attention_mode in ("auto", "sdpa") and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            try:
                from diffusers.models.attention_processor import AttnProcessor2_0
                self.unet.set_attn_processor(AttnProcessor2_0())
            except Exception as e:
                print(f"[AIFX VideoMaMa] SDPA setup failed ({e}); using default attention")

    @staticmethod
    def _clear_device_cache(device):
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device.type == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    @torch.no_grad()
    def run(self, cond_frames, mask_frames, seed=42, fps=7, motion_bucket_id=127, noise_aug_strength=0.0):
        """
        cond_frames: list of RGB uint8 (H, W, 3); mask_frames: list of uint8 (H, W) or (H, W, 3).
        Returns a list of single-channel float32 alpha arrays (H, W) in [0, 1].
        """
        cond_video = self._frames_to_tensor(cond_frames).to(self.device)
        mask_video = self._frames_to_tensor(mask_frames).to(self.device)

        hidden_size = self.unet.config.cross_attention_dim
        encoder_hidden_states = torch.zeros((1, 1, hidden_size), dtype=self.weight_dtype, device=self.device)

        if self.enable_model_cpu_offload:
            self.vae.to(self.device)
        cond_latents = self._encode(cond_video.to(self.weight_dtype)) / self.vae.config.scaling_factor
        mask_latents = self._encode(mask_video.to(self.weight_dtype)) / self.vae.config.scaling_factor
        del cond_video, mask_video
        if self.enable_model_cpu_offload:
            self.vae.to("cpu")
        self._clear_device_cache(self.device)

        if self.enable_model_cpu_offload:
            self.unet.to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        noisy_latents = torch.randn(cond_latents.shape, generator=generator, device=self.device, dtype=self.weight_dtype)
        timesteps = torch.full((1,), 1.0, device=self.device, dtype=torch.long)
        added_time_ids = self._get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, batch_size=1)

        unet_input = torch.cat([noisy_latents, cond_latents, mask_latents], dim=2)
        del noisy_latents, cond_latents, mask_latents
        self._clear_device_cache(self.device)

        pred_latents = self.unet(unet_input, timesteps, encoder_hidden_states, added_time_ids=added_time_ids).sample
        del unet_input
        if self.enable_model_cpu_offload:
            self.unet.to("cpu")
        self._clear_device_cache(self.device)

        if self.enable_model_cpu_offload:
            self.vae.to(self.device)
        pred_latents = (1 / self.vae.config.scaling_factor) * pred_latents.squeeze(0)
        frames = []
        step = min(self.vae_encode_chunk_size, pred_latents.shape[0])
        for i in range(0, pred_latents.shape[0], step):
            chunk = pred_latents[i: i + step]
            frames.append(self.vae.decode(chunk, num_frames=chunk.shape[0]).sample.float().cpu())
        del pred_latents
        if self.enable_model_cpu_offload:
            self.vae.to("cpu")
        self._clear_device_cache(self.device)

        video = torch.cat(frames, dim=0)
        alpha = (video / 2.0 + 0.5).clamp(0, 1).mean(dim=1)  # (F, H, W)
        return [a.numpy() for a in alpha]

    @staticmethod
    def _frames_to_tensor(frames) -> torch.Tensor:
        """List of uint8 (H, W[, 3]) -> (1, F, 3, H, W) float in [-1, 1]."""
        tensors = []
        for f in frames:
            arr = np.asarray(f)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            tensors.append(torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0)
        return torch.stack(tensors).unsqueeze(0) * 2.0 - 1.0

    def _encode(self, t: torch.Tensor) -> torch.Tensor:
        b, f = t.shape[0], t.shape[1]
        t = t.reshape(b * f, *t.shape[2:])
        latents = [self.vae.encode(t[i: i + self.vae_encode_chunk_size]).latent_dist.sample()
                   for i in range(0, t.shape[0], self.vae_encode_chunk_size)]
        latents = torch.cat(latents, dim=0)
        return latents.reshape(b, f, *latents.shape[1:]) * self.vae.config.scaling_factor

    def _get_add_time_ids(self, fps, motion_bucket_id, noise_aug_strength, batch_size):
        ids = [fps, motion_bucket_id, noise_aug_strength]
        if self.unet.config.addition_time_embed_dim * len(ids) != self.unet.add_embedding.linear_1.in_features:
            raise ValueError("VideoMaMa UNet expects a different added time embedding size")
        return torch.tensor([ids], dtype=self.weight_dtype, device=self.device).repeat(batch_size, 1)
