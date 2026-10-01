# SPDX-License-Identifier: GPL-3.0-or-later
#
# AIFX VideoMaMa nodes: high-resolution, any-length VideoMaMa matting for ComfyUI.
#
# The crop / proportional resize / sliding-window / soft-mask feedback / boundary
# blend strategy follows Sammie-Roto 2 (sammie/matting.py, GPL-3.0,
# https://github.com/Zarxrax/Sammie-Roto-2):
#   1. one fixed crop rect = union of the mask over the whole range + buffer,
#   2. the crop is resized so its SHORT side is `max_resolution` (multiple of 8),
#   3. frames go through the model in windows of `chunk_size` with `overlap`,
#      the overlap frames of each window being guided by the previous window's
#      soft alpha instead of the binary mask,
#   4. overlap frames are linearly blended across the boundary,
#   5. the alpha is resized back to the crop and pasted into the full frame.
#
# Input names match ComfyUI-VideoMaMa's VideoMaMaPipelineLoader / VideoMaMaSampler
# so an existing workflow only needs its class_type swapped.

import gc
import os

import cv2
import numpy as np
import torch

import comfy.model_management
import folder_paths
from comfy.utils import ProgressBar

from .pipeline import VideoInferencePipeline

_PIPELINE_CACHE = {"key": None, "pipeline": None}


def _resolve_model_dir(path: str) -> str:
    """Absolute path as-is; otherwise look next to ComfyUI-VideoMaMa, then in ComfyUI/models."""
    if os.path.isabs(path) and os.path.isdir(path):
        return path
    custom_nodes = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(custom_nodes, "ComfyUI-VideoMaMa", path),
        os.path.join(folder_paths.models_dir, path),
        os.path.join(folder_paths.base_path, path),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        f"VideoMaMa model folder '{path}' not found. Looked in: " + "; ".join(candidates) +
        ". Run ComfyUI-VideoMaMa's loader once to download it, or give an absolute path.")


class AIFXVideoMaMaPipelineLoader:
    """Loads (and caches) the SVD VAE + VideoMaMa UNet. Inputs mirror VideoMaMaPipelineLoader."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": ("STRING", {"default": "checkpoints/stabilityai/stable-video-diffusion-img2vid-xt"}),
                "unet_checkpoint_path": ("STRING", {"default": "checkpoints/VideoMaMa"}),
                "precision": (["fp16", "bf16"], {"default": "fp16"}),
                "enable_model_cpu_offload": ("BOOLEAN", {"default": False}),
                "vae_encode_chunk_size": ("INT", {"default": 1, "min": 1, "max": 32}),
                "attention_mode": (["auto", "xformers", "sdpa", "none"], {"default": "auto"}),
                "enable_vae_tiling": ("BOOLEAN", {"default": False}),
                "enable_vae_slicing": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AIFX_VIDEOMAMA_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "AIFX/VideoMaMa"

    def load(self, base_model_path, unet_checkpoint_path, precision, enable_model_cpu_offload,
             vae_encode_chunk_size, attention_mode, enable_vae_tiling, enable_vae_slicing):
        base = _resolve_model_dir(base_model_path)
        unet = _resolve_model_dir(unet_checkpoint_path)
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        device = comfy.model_management.get_torch_device()
        key = (base, unet, precision, enable_model_cpu_offload, vae_encode_chunk_size,
               attention_mode, enable_vae_tiling, enable_vae_slicing, str(device))

        if _PIPELINE_CACHE["key"] != key:
            _PIPELINE_CACHE["pipeline"] = None
            gc.collect()
            comfy.model_management.soft_empty_cache()
            # Make room: other ComfyUI models (e.g. SAM3) may still sit in VRAM.
            comfy.model_management.unload_all_models()
            _PIPELINE_CACHE["pipeline"] = VideoInferencePipeline(
                base_model_path=base, unet_checkpoint_path=unet, device=str(device),
                weight_dtype=dtype, enable_model_cpu_offload=enable_model_cpu_offload,
                vae_encode_chunk_size=vae_encode_chunk_size, attention_mode=attention_mode,
                enable_vae_tiling=enable_vae_tiling, enable_vae_slicing=enable_vae_slicing)
            _PIPELINE_CACHE["key"] = key
            print(f"[AIFX VideoMaMa] pipeline loaded ({precision}, offload={enable_model_cpu_offload}) on {device}")
        return (_PIPELINE_CACHE["pipeline"],)


def _srgb_encode(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def _crop_rect(masks: np.ndarray, buffer: float):
    """Union bbox of all masks, padded by `buffer` of its own size, snapped to multiples of 8.
    Returns (x1, y1, x2, y2) with x2/y2 exclusive, or None when every mask is empty."""
    union = (masks > 0.5).any(axis=0)
    ys, xs = np.nonzero(union)
    if len(xs) == 0:
        return None
    h, w = union.shape
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad_x, pad_y = int((x2 - x1) * buffer), int((y2 - y1) * buffer)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    x1, y1 = (x1 // 8) * 8, (y1 // 8) * 8
    x2, y2 = min(w, -(-x2 // 8) * 8), min(h, -(-y2 // 8) * 8)
    return int(x1), int(y1), int(x2), int(y2)


def _windows(total: int, size: int, overlap: int):
    overlap = min(overlap, size - 1)
    if total <= size:
        return [(0, total)], overlap
    out, pos = [], 0
    while pos < total:
        end = min(pos + size, total)
        out.append((pos, end))
        if end >= total:
            break
        pos += size - overlap
    return out, overlap


class AIFXVideoMaMaChunked:
    """VideoMaMa on a mask-tight crop, at a short-side resolution, in overlapping windows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("AIFX_VIDEOMAMA_PIPELINE",),
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "max_resolution": ("INT", {"default": 1080, "min": 0, "max": 4096, "step": 8,
                                           "tooltip": "Short side of the processed crop (0 = native). SammieRoto default: 1080."}),
                "fps": ("INT", {"default": 7, "min": 1, "max": 60}),
                "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 255}),
                "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "chunk_size": ("INT", {"default": 16, "min": 2, "max": 128,
                                       "tooltip": "Frames per model window; lower = less VRAM."}),
                "overlap": ("INT", {"default": 2, "min": 0, "max": 16,
                                    "tooltip": "Frames shared by consecutive windows (soft-mask feedback + linear blend)."}),
                "crop_to_mask": ("BOOLEAN", {"default": True,
                                             "tooltip": "Process only the union bbox of the mask over the clip."}),
                "crop_buffer": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "input_transfer": (["as-is (display-referred)", "linear -> sRGB"], {"default": "as-is (display-referred)",
                                   "tooltip": "VideoMaMa expects display-referred RGB. Pick linear -> sRGB for scene-linear plates."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "run"
    CATEGORY = "AIFX/VideoMaMa"

    def run(self, pipeline, images, masks, seed, max_resolution, fps, motion_bucket_id, noise_aug_strength,
            chunk_size=16, overlap=2, crop_to_mask=True, crop_buffer=0.10,
            input_transfer="as-is (display-referred)"):
        n, full_h, full_w = images.shape[0], images.shape[1], images.shape[2]
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
        if masks.shape[0] != n:
            raise ValueError(f"{n} image frames but {masks.shape[0]} mask frames")
        if masks.shape[1:] != (full_h, full_w):
            masks = torch.nn.functional.interpolate(masks[:, None], size=(full_h, full_w), mode="nearest")[:, 0]

        mask_np = masks.cpu().numpy().astype(np.float32)
        rect = _crop_rect(mask_np, crop_buffer) if crop_to_mask else None
        x1, y1, x2, y2 = rect if rect else (0, 0, full_w, full_h)
        crop_w, crop_h = x2 - x1, y2 - y1

        scale = 1.0
        if max_resolution > 0 and min(crop_w, crop_h) > max_resolution:
            scale = max_resolution / min(crop_w, crop_h)
        proc_w = max(8, (int(crop_w * scale) // 8) * 8)
        proc_h = max(8, (int(crop_h * scale) // 8) * 8)

        wins, overlap = _windows(n, chunk_size, overlap)
        print(f"[AIFX VideoMaMa] {n} frames {full_w}x{full_h}, crop ({x1},{y1})-({x2},{y2}) {crop_w}x{crop_h} "
              f"-> model {proc_w}x{proc_h}, {len(wins)} window(s) of {chunk_size} / overlap {overlap}")

        linear = input_transfer.startswith("linear")

        def load_frame(i):
            rgb = images[i, y1:y2, x1:x2, :3].cpu().numpy().astype(np.float32)
            rgb = _srgb_encode(rgb) if linear else np.clip(rgb, 0.0, 1.0)  # clip: >1.0 must not wrap in uint8
            rgb = (rgb * 255.0 + 0.5).astype(np.uint8)
            return cv2.resize(rgb, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        def load_mask(i):
            m = (mask_np[i, y1:y2, x1:x2] > 0.5).astype(np.uint8) * 255
            return cv2.resize(m, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)

        alphas = [None] * n                     # model-resolution float alpha, committed
        prev_soft = None                        # uint8 soft alpha of the previous window's tail
        pbar = ProgressBar(len(wins))

        for wi, (s, e) in enumerate(wins):
            comfy.model_management.throw_exception_if_processing_interrupted()
            cond = [load_frame(i) for i in range(s, e)]
            guide = [load_mask(i) for i in range(s, e)]
            first = wi == 0
            if not first and prev_soft is not None:
                for k in range(min(overlap, len(guide), len(prev_soft))):
                    guide[k] = prev_soft[k]

            out = pipeline.run(cond, guide, seed=seed, fps=fps, motion_bucket_id=motion_bucket_id,
                               noise_aug_strength=noise_aug_strength)

            if overlap > 0:
                prev_soft = [(a * 255.0 + 0.5).astype(np.uint8) for a in out[-overlap:]]

            if first:
                for k, a in enumerate(out):
                    alphas[s + k] = a
            else:
                for k, a in enumerate(out):
                    idx = s + k
                    if k < overlap and alphas[idx] is not None:
                        w_new = (k + 1) / (overlap + 1)
                        alphas[idx] = (1.0 - w_new) * alphas[idx] + w_new * a
                    else:
                        alphas[idx] = a
            pbar.update(1)

        result = torch.zeros((n, full_h, full_w), dtype=torch.float32)
        for i, a in enumerate(alphas):
            up = cv2.resize(a.astype(np.float32), (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            result[i, y1:y2, x1:x2] = torch.from_numpy(np.clip(up, 0.0, 1.0))
        return (result,)


NODE_CLASS_MAPPINGS = {
    "AIFXVideoMaMaPipelineLoader": AIFXVideoMaMaPipelineLoader,
    "AIFXVideoMaMaChunked": AIFXVideoMaMaChunked,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIFXVideoMaMaPipelineLoader": "AIFX VideoMaMa Loader",
    "AIFXVideoMaMaChunked": "AIFX VideoMaMa (Crop + Chunk)",
}
