# AIFX VideoMaMa (Crop + Chunk)

ComfyUI nodes that run VideoMaMa matting at high resolution on clips of any
length, using Sammie-Roto 2's strategy: a fixed mask-tight crop, short-side
resizing (default 1080), overlapping windows with soft-alpha feedback, and a
linear blend at window boundaries. Drop-in replacement for ComfyUI-VideoMaMa's
`VideoMaMaPipelineLoader` / `VideoMaMaSampler` (same input names).

Install: copy this folder to `ComfyUI/custom_nodes/aifx_videomama` and restart
ComfyUI. Uses the models ComfyUI-VideoMaMa already downloaded.

## License

GPL-3.0-or-later. `pipeline.py` and the chunking strategy in `nodes.py` are
derived from [Sammie-Roto 2](https://github.com/Zarxrax/Sammie-Roto-2) (GPL-3.0).
This folder is therefore GPL-licensed, unlike the BSD-3-Clause AIFX plugin code.
