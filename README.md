# ComfyUI-Gemini-VAE-Fix
A robust, universal VAE Decoder for ComfyUI designed with Gemini AI to solve 5D tensor errors and OOM issues in LTX-Video and other video models. Features auto-scale detection and manual spatial tiling.
# 🎬 ComfyUI Gemini Universal VAE Decode (v1.1)

This custom node for ComfyUI provides a "smarter" and more stable way to decode VAE latents, especially for video models like **LTX-Video**, **SVD**, or high-resolution **SDXL** renders. 

It was developed to solve the notorious `RuntimeError: The size of tensor a must match the size of tensor b` and out-of-memory (OOM) issues when dealing with 5D video tensors.

## 🚀 Key Features

* **5D Video Support:** Specifically handles `[Batch, Channel, Frames, Height, Width]` tensors that often crash standard VAE nodes.
* **Auto-Scale Detection:** Automatically detects if your VAE uses x8, x16, or x32 compression (tested with LTX-Video, SDXL, and Flux). No manual configuration needed!
* **Temporal & Spatial Tiling:** * **Temporal:** Decodes video in chunks of frames (frames_per_batch) to save VRAM.
    * **Spatial:** Manually tiles large frames into smaller pieces during decoding.
* **Memory Management:** Forced garbage collection and cache clearing after each batch to prevent crashes during long video renders.

## 🛠️ Installation

1. Navigate to your `ComfyUI/custom_nodes/` directory.
2. Clone this repo: 
   ```bash
   git clone [https://github.com/uczensokratesa/ComfyUI-Gemini-VAE-Fix.git](https://github.com/uczensokratesa/ComfyUI-Gemini-VAE-Fix.git)
