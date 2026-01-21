import torch
import comfy.utils
from comfy.model_management import throw_exception_if_processing_interrupted

class UniversalSmartVAEDecode:
    """
    Universal VAE Decode v1.1
    - Supports 5D video tensors (Batch, Channel, Frames, Height, Width)
    - Manual Spatial Tiling to prevent OOM on high-res video
    - Auto-detection of VAE compression scale (works with SD, SDXL, SVD, LTX)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", ),
                "vae": ("VAE", ),
                "frames_per_batch": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1, "tooltip": "Liczba klatek dekodowana jednorazowo (VRAM intensive)."}),
            },
            "optional": {
                "tile_mode": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64, "tooltip": "Rozmiar kafelka w pikselach."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "Dario Nodes/Video"

    def decode(self, vae, samples, frames_per_batch, tile_mode=True, tile_size=512):
        latents = samples["samples"]
        decoded_parts = []
        
        # Wykrywanie czy to wideo 5D
        is_video_5d = len(latents.shape) == 5
        
        # --- AUTO-SCALE DETECTION ---
        # Zamiast zgadywać 8, sprawdzamy faktyczną kompresję VAE
        # Pobieramy malutką próbkę latentu
        try:
            if is_video_5d:
                test_sample = latents[:, :, 0, :8, :8] # [B, C, 8, 8] slice from first frame
            else:
                test_sample = latents[:, :, :8, :8]    # [B, C, 8, 8]
            
            # Szybki decode próbki
            test_decoded = vae.decode(test_sample)
            
            # Obliczamy scale factor (np. 64px output / 8px latent = 8x scale)
            # Obsługa różnych formatów wyjścia VAE (czasem [B,H,W,C], czasem [B,C,H,W])
            if test_decoded.shape[-1] in [1, 3, 4]: # Channels last
                out_size = test_decoded.shape[-2]
            else:
                out_size = test_decoded.shape[-1]
                
            scale_factor = out_size // 8
            print(f"🕵️ [Universal VAE] Detected VAE Scale Factor: x{scale_factor}")
        except Exception as e:
            print(f"⚠️ [Universal VAE] Scale detection failed ({e}). Defaulting to x8.")
            scale_factor = 8

        if is_video_5d:
            b, c, total_frames, h_latent, w_latent = latents.shape
            print(f"🎬 [Universal VAE] Processing 5D Video. Frames: {total_frames}, Tile Mode: {tile_mode}")
            
            pbar = comfy.utils.ProgressBar(total_frames)

            for start_idx in range(0, total_frames, frames_per_batch):
                throw_exception_if_processing_interrupted()
                end_idx = min(start_idx + frames_per_batch, total_frames)
                
                # Wycinamy fragment czasu: [B, C, F_sub, H, W]
                chunk = latents[:, :, start_idx:end_idx, :, :]
                
                if tile_mode:
                    tile_l = tile_size // scale_factor
                    
                    # Ręczne kafelkowanie
                    h_chunks = []
                    for y in range(0, h_latent, tile_l):
                        w_chunks = []
                        for x in range(0, w_latent, tile_l):
                            # Crop latentu
                            crop = chunk[:, :, :, y:y+tile_l, x:x+tile_l]
                            
                            try:
                                decoded_crop = vae.decode(crop)
                            except Exception as e:
                                print(f"⚠️ Tile decode error at {x},{y}: {e}. Retrying fallback...")
                                decoded_crop = vae.decode(crop) # Retry or could be fallback to CPU
                            
                            w_chunks.append(decoded_crop)
                        
                        # Sklejanie w poziomie
                        # Sprawdzamy gdzie jest wymiar Width. Dla Comfy Image jest to zazwyczaj [..., W, C] lub [..., W]
                        # VAE decode zwraca zazwyczaj [Batch, Frames, Height, Width, Channels] dla wideo
                        cat_dim_w = -2 if w_chunks[0].shape[-1] in [1,3,4] else -1
                        h_chunks.append(torch.cat(w_chunks, dim=cat_dim_w))
                    
                    # Sklejanie w pionie
                    cat_dim_h = -3 if h_chunks[0].shape[-1] in [1,3,4] else -2
                    decoded_chunk = torch.cat(h_chunks, dim=cat_dim_h)
                    
                else:
                    decoded_chunk = vae.decode(chunk)

                # Normalizacja wymiarów wyjściowych (usuwanie Batch=1 jeśli istnieje i jest zbędny)
                if len(decoded_chunk.shape) == 5 and decoded_chunk.shape[0] == 1:
                    decoded_chunk = decoded_chunk.squeeze(0) 
                
                decoded_parts.append(decoded_chunk)
                pbar.update(end_idx - start_idx)
                
                # Zwolnienie pamięci po każdym batchu
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            final_tensor = torch.cat(decoded_parts, dim=0)

        else:
            # Tryb 4D (Zwykłe obrazy)
            print(f"🖼️ [Universal VAE] Standard 4D processing.")
            if tile_mode:
                final_tensor = vae.decode_tiled(latents, tile_x=tile_size, tile_y=tile_size)
            else:
                final_tensor = vae.decode(latents)

        return (final_tensor,)

NODE_CLASS_MAPPINGS = {
    "UniversalSmartVAEDecode": UniversalSmartVAEDecode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalSmartVAEDecode": "🎬 Universal VAE Decode (v1.1)"
}
