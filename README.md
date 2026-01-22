# ComfyUI-Gemini-VAE-Fix (Temporal-Aware VAE Decoding)

This node provides a professional solution for decoding long video sequences in ComfyUI using **Temporal-Aware Batching**. Unlike standard VAE decoders or simple tiling methods, this node ensures perfect frame synchronization and eliminates flickering between batches.

## 🚀 Key Features
* **Temporal Context Overlap:** Every batch "sees" a bit of the previous and next frames, preventing temporal artifacts at batch boundaries.
* **Smart Stitching (v2.3):** Uses a global index mapping system to ensure that the output frame count perfectly matches the latent input (e.g., exactly 681 frames for a 681-frame latent).
* **VRAM Efficiency:** Decode videos of any length (4K, 8K) by splitting them into small temporal batches without losing audio/video sync.
* **Auto-Correction:** Safety guards prevent crashes if users set invalid overlap or batch parameters.
* **Universal Compatibility:** Tested with SVD, LTX-Video, CogVideo, and standard SDXL VAEs.

## 🛠 Usage
1.  **frames_per_batch:** How many latent frames to decode at once. Lower for less VRAM, higher for speed.
2.  **overlap:** Number of context frames. Recommended: `2` or `4`.
3.  **tile_mode:** Keep `True` for high-resolution videos to avoid "Out of Memory" errors.

## Why is this different?
Most decoders treat video batches as independent image lists. This node treats video as a **continuous signal**, performing precise "surgical" trims on decoded batches to reconstruct a seamless temporal stream.
