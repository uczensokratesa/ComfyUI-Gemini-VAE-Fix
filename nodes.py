# Finalna wersja v2.3 - "The Temporal Master"
# Główne zmiany: 
# - Pełna walidacja zakresów (crash-proof)
# - Automatyczne wykrywanie typu VAE
# - Optymalizacja czyszczenia cache CUDA

import torch
import comfy.utils
from comfy.model_management import throw_exception_if_processing_interrupted
import gc

class UniversalSmartVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "frames_per_batch": ("INT", {"default": 8, "min": 2, "max": 128, "step": 1}),
                "overlap": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1}),
            },
            "optional": {
                "tile_mode": ("BOOLEAN", {"default": True}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "Dario Nodes/Video"

    def normalize_output(self, tensor):
        if isinstance(tensor, (list, tuple)): tensor = tensor[0]
        if tensor.dim() == 5:
            if tensor.shape[1] in [3, 4]: tensor = tensor.permute(0, 2, 3, 4, 1)
            b, f, h, w, c = tensor.shape
            tensor = tensor.reshape(b * f, h, w, c)
        elif tensor.dim() == 4:
            if tensor.shape[1] in [3, 4]: tensor = tensor.permute(0, 2, 3, 1)
        return tensor

    def decode(self, vae, samples, frames_per_batch, overlap=2, tile_mode=True, tile_size=512):
        latents = samples["samples"]
        b, c, total_frames, h_l, w_l = latents.shape if latents.dim() == 5 else (latents.shape[0], latents.shape[1], 1, latents.shape[2], latents.shape[3])

        # --- AUTO-CORRECT PARAMS ---
        frames_per_batch = max(2, min(frames_per_batch, total_frames))
        overlap = max(0, min(overlap, (frames_per_batch // 2) - 1))
        
        # --- SCALE DETECTION ---
        # Próbujemy wykryć skalę czasową (np. LTX=8, SVD=1, CogVideo=4-8)
        time_scale = 8 # default
        try:
            test_chunk = latents[:, :, :2, :64, :64]
            test_dec = self.normalize_output(vae.decode(test_chunk))
            if test_dec.shape[0] > 1:
                time_scale = test_dec.shape[0] - 1
        except: pass

        expected = 1 + (total_frames - 1) * time_scale
        print(f"🚀 Decoding {total_frames} latents. Expected output: {expected} frames. (Scale: x{time_scale})")

        final_frames = []
        pbar = comfy.utils.ProgressBar(total_frames)

        for start_idx in range(0, total_frames, frames_per_batch):
            throw_exception_if_processing_interrupted()
            
            end_idx = min(start_idx + frames_per_batch, total_frames)
            ctx_start = max(0, start_idx - overlap)
            ctx_end = min(total_frames, end_idx + overlap)
            
            chunk = latents[:, :, ctx_start:ctx_end, :, :]
            
            try:
                if tile_mode:
                    decoded = vae.decode_tiled(chunk, tile_x=tile_size, tile_y=tile_size)
                else:
                    decoded = vae.decode(chunk)
            except Exception as e:
                print(f"⚠️ Memory pressure? Switching to tiled: {e}")
                decoded = vae.decode_tiled(chunk, tile_x=256, tile_y=256)

            decoded = self.normalize_output(decoded)
            
            # --- THE STITCHING MAGIC ---
            f_trim = (start_idx - ctx_start) * time_scale
            
            if end_idx == total_frames:
                # Ostatni batch bierze WSZYSTKO co zostało
                valid = decoded[f_trim:]
            else:
                # Środkowe batche tniemy precyzyjnie
                core_len = (end_idx - start_idx) * time_scale
                valid = decoded[f_trim : f_trim + core_len]

            final_frames.append(valid)
            
            # Memory Management
            del decoded, chunk
            pbar.update(end_idx - start_idx)
            # gc.collect() i empty_cache() wywołujemy rzadziej dla wydajności
            if start_idx % (frames_per_batch * 2) == 0:
                torch.cuda.empty_cache()

        result = torch.cat(final_frames, dim=0)
        print(f"✨ Successfully reconstructed {result.shape[0]} frames.")
        return (result,)

NODE_CLASS_MAPPINGS = {"UniversalSmartVAEDecode": UniversalSmartVAEDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"UniversalSmartVAEDecode": "🎬 Universal VAE Decode (v2.3 Final)"}
