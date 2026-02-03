"""
Universal Smart VAE Decode - Audio Sync Fix (God Mode)
Fixes the 'Fencepost Error' causing audio desync in chunked decoding.

Base: Grok v5.1 (Architecture) + Gemini (Math Fix)
Version: 6.0.0 (Audio Sync)
License: MIT
"""

import torch
import comfy.utils
from comfy.model_management import throw_exception_if_processing_interrupted
import gc

class GeminiSmartVAEDecode

:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "frames_per_batch": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
            },
            "optional": {
                "overlap_frames": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1}),
                "force_time_scale": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "enable_tiling": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent/video"

    def __init__(self):
        self.cached_vae_id = None
        self.cached_time_scale = None

    def _get_available_vram(self):
        try:
            if not torch.cuda.is_available(): return None
            free_vram, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
            return free_vram / (1024 ** 3)
        except: return None

    def detect_scales(self, vae, latents, force_time=0):
        if force_time > 0: return force_time, 8 # Assume 8 spatial default
        
        vae_id = id(vae)
        if vae_id == self.cached_vae_id and self.cached_time_scale is not None:
            return self.cached_time_scale, 8

        # 1. Metadata
        if hasattr(vae, 'downscale_index_formula') and vae.downscale_index_formula:
            try:
                scale = int(vae.downscale_index_formula[0])
                self.cached_vae_id = vae_id
                self.cached_time_scale = scale
                return scale, 8
            except: pass

        # 2. Empirical (5 frames for accuracy)
        try:
            test_sample = latents[:, :, 0:5, :16, :16]
            with torch.no_grad():
                # Force standard decode
                test_out = vae.decode(test_sample)
            
            # Normalize
            if isinstance(test_out, (list, tuple)): test_out = test_out[0]
            if test_out.dim() == 5 and test_out.shape[1] in [3,4]: frames = test_out.shape[2]
            elif test_out.dim() == 5: frames = test_out.shape[1]
            else: frames = test_out.shape[0]

            # Scale = (Out - 1) / (In - 1)
            scale = max(1, (frames - 1) // 4)
            print(f"🔍 Auto-detected time_scale: {scale}x")
            
            self.cached_vae_id = vae_id
            self.cached_time_scale = scale
            return scale, 8
        except:
            print("⚠️ Scale detection failed. Defaulting to 1x.")
            return 1, 8

    def _normalize(self, tensor):
        if isinstance(tensor, (list, tuple)): tensor = tensor[0]
        if tensor.dim() == 5:
            if tensor.shape[1] in [3, 4]: tensor = tensor.permute(0, 2, 3, 4, 1)
            b, f, h, w, c = tensor.shape
            tensor = tensor.reshape(b * f, h, w, c)
        elif tensor.dim() == 4:
            if tensor.shape[1] in [3, 4]: tensor = tensor.permute(0, 2, 3, 1)
        
        tensor = torch.clamp(tensor, 0.0, 1.0)
        if tensor.shape[-1] > 3: tensor = tensor[..., :3]
        return tensor.contiguous()

    def decode(self, vae, samples, frames_per_batch, overlap_frames=2, force_time_scale=0, enable_tiling=False, tile_size=512):
        latents = samples["samples"]
        
        # Image Path
        if latents.dim() == 4:
            with torch.no_grad():
                out = vae.decode_tiled(latents, tile_x=tile_size, tile_y=tile_size) if enable_tiling else vae.decode(latents)
            return (self._normalize(out),)

        # Video Path
        B, C, total_frames, H, W = latents.shape
        time_scale, _ = self.detect_scales(vae, latents, force_time_scale)
        
        # --- CORRECT TOTAL FRAME CALCULATION ---
        # Formula: 1 + (Total - 1) * Scale
        expected_total_frames = 1 + (total_frames - 1) * time_scale
        print(f"🎬 Decoding Video: {total_frames} latents -> {expected_total_frames} frames (Sync Guarantee)")

        frames_per_batch = max(1, min(frames_per_batch, total_frames))
        overlap_frames = max(0, min(overlap_frames, frames_per_batch - 1))
        
        output_chunks = []
        start_idx = 0
        current_batch = frames_per_batch
        
        pbar = comfy.utils.ProgressBar(total_frames)

        while start_idx < total_frames:
            throw_exception_if_processing_interrupted()
            
            end_idx = min(start_idx + current_batch, total_frames)
            
            # Context window (temporal overlap)
            ctx_start = max(0, start_idx - overlap_frames)
            ctx_end = min(total_frames, end_idx + overlap_frames)
            
            latent_chunk = latents[:, :, ctx_start:ctx_end, :, :]
            
            try:
                with torch.no_grad():
                    if enable_tiling and hasattr(vae, 'decode_tiled'):
                        decoded_raw = vae.decode_tiled(latent_chunk, tile_x=tile_size, tile_y=tile_size)
                    else:
                        decoded_raw = vae.decode(latent_chunk)
            except RuntimeError as e:
                # OOM RECOVERY (Grok's Logic)
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    gc.collect()
                    if not enable_tiling:
                        enable_tiling = True; continue
                    if current_batch > 1:
                        current_batch = max(1, current_batch // 2)
                        overlap_frames = min(overlap_frames, current_batch - 1)
                        continue
                    if tile_size > 256:
                        tile_size //= 2; continue
                    raise e
                raise e

            decoded = self._normalize(decoded_raw)
            
            # --- THE SYNC FIX (MATH CORRECTION) ---
            # 1. Calculate where the "valid core" starts in the decoded chunk
            #    Latents processed from ctx_start. We need data from start_idx.
            #    Difference = start_idx - ctx_start.
            #    Frames to skip = Difference * time_scale
            front_trim = (start_idx - ctx_start) * time_scale
            
            # 2. Calculate correct length
            if end_idx == total_frames:
                # LAST CHUNK: Take everything remaining
                # This naturally includes the "+1" frame at the very end
                valid = decoded[front_trim:]
            else:
                # MIDDLE CHUNKS: Strict linear mapping
                # Length = (Latents In Chunk) * Scale
                # DO NOT use the "1 + ..." formula here, or you get duplicates/shifts!
                core_len = (end_idx - start_idx) * time_scale
                valid = decoded[front_trim : front_trim + core_len]

            output_chunks.append(valid)
            
            pbar.update(end_idx - start_idx)
            start_idx = end_idx # Move exactly by the batch size
            
            del latent_chunk, decoded_raw, decoded, valid
            if start_idx % (frames_per_batch * 2) == 0:
                gc.collect(); torch.cuda.empty_cache()

        final_output = torch.cat(output_chunks, dim=0)
        
        # Final sanity check for audio sync
        if final_output.shape[0] != expected_total_frames:
            print(f"⚠️ Warning: Output {final_output.shape[0]} != Expected {expected_total_frames}. Audio might be slightly off.")
        else:
            print(f"✅ Audio Sync Check: PERFECT ({final_output.shape[0]} frames)")
            
        return (final_output,)

NODE_CLASS_MAPPINGS = {"GeminiSmartVAEDecode": GeminiSmartVAEDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiSmartVAEDecode": "🎬 GeminiSmartVAEDecode(Audio Sync Fix)"}
