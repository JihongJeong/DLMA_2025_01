import os
import dotenv

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"

if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    print("No Gemini API key provided. Please set the GEMINI_API_KEY environment variable.")

GLOBAL_WEBTOON_STYLE = {
    'art_style': '현대적이고 깔끔한 웹툰 스타일',
    'color_palette': '밝고 선명한 색상 위주'
}

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "YOUR_STABILITY_AI_KEY_HERE")

IMAGE_GEN_CONFIG = {
    "stability_ai": {
        "api_key": STABILITY_API_KEY,
        "engine_id": "stable-diffusion-xl-1024-v1-0",
        # "engine_id":  "stable-diffusion-xl-1024-v1-0" (SDXL 1.0)
        #               "stable-diffusion-v1-6" (SD 1.6)
        #               "stable-diffusion-3-medium" (SD3 Medium)
        "samples": 1,
        "width": 1024, 
        "height": 1024, 
        "steps": 30,  
        "cfg_scale": 7.0,
        "seed": 0, 
        "style_preset": "comic-book",
        # style_preset: "3d-model", "analog-film", "anime", "cinematic", "comic-book",
        #               "digital-art", "enhance", "fantasy-art", "isometric", "line-art",
        #               "low-poly", "modeling-compound", "neon-punk", "origami",
        #               "photographic", "pixel-art", "tile-texture"
        # "negative_prompt": "blurry, low quality, text, watermark, signature, ugly, deformed",
        # "sampler": "K_DPMPP_2M" 
    }
}

if STABILITY_API_KEY == "YOUR_STABILITY_AI_KEY_HERE":
    print("No Stability API key provided. Please set the STABILITY_API_KEY environment variable.")