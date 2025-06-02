import base64
import requests 
import json
import traceback
from typing import Dict, Any, Optional, List

STABILITY_API_HOST = "https://api.stability.ai"

class ImageGenerator:
    def __init__(self):
        """
        Stability AI API를 사용하는 ImageGenerator를 초기화합니다.
        """
        print("ImageGenerator initialized for Stability AI.")

    def create_image_from_prompt(
        self,
        image_prompt_text: str,
        image_model_configuration: Dict[str, Any]
    ) -> Optional[List[bytes]]:
        """
        Stability AI API를 사용하여 텍스트 프롬프트로부터 이미지(들)를 생성합니다.

        Args:
            image_prompt_text (str): 주된 긍정적 텍스트 프롬프트.
            image_model_configuration (Dict[str, Any]): Stability AI 모델 설정.
                필수 키:
                - 'api_key' (str): Stability AI API 키.
                - 'engine_id' (str): 사용할 모델의 엔진 ID.
                  (예: "stable-diffusion-xl-1024-v1-0", "stable-diffusion-v1-6", "stable-diffusion-3-medium").
                  Stability AI 문서에서 사용 가능한 최신 엔진 ID를 확인하세요.
                선택적 키:
                - 'samples' (int): 생성할 이미지 수 (기본값 1).
                - 'width' (int): 이미지 너비 (SDXL의 경우 기본값 1024, 이전 모델은 512).
                - 'height' (int): 이미지 높이 (SDXL의 경우 기본값 1024, 이전 모델은 512).
                - 'steps' (int): 확산 단계 수 (기본값 30-50, SD3는 더 적을 수 있음).
                - 'cfg_scale' (float): Classifier-Free Guidance 스케일 (기본값 7).
                - 'seed' (int): 재현성을 위한 시드 값 (0은 랜덤).
                - 'style_preset' (str): 스타일 프리셋 (예: "comic-book", "anime", "photographic", "line-art").
                                         웹툰에는 "comic-book", "anime", "line-art" 등이 유용할 수 있습니다.
                - 'negative_prompt' (str): 부정적 프롬프트 텍스트.
                - 'sampler' (str): 사용할 샘플러 (예: "K_DPM_2_ANCESTRAL", "DDIM"). 모델마다 지원 샘플러가 다를 수 있습니다.

        Returns:
            Optional[List[bytes]]: 생성된 이미지 데이터(bytes)의 리스트. 실패 시 None.
        """
        api_key = image_model_configuration.get("api_key")
        engine_id = image_model_configuration.get("engine_id")

        if not api_key:
            print("  [ERROR] Stability AI: 'api_key'가 image_model_configuration에 제공되어야 합니다.")
            return None
        if not engine_id:
            print("  [ERROR] Stability AI: 'engine_id'가 제공되어야 합니다 (예: 'stable-diffusion-xl-1024-v1-0').")
            return None

        if "stable-diffusion-3" in engine_id:
            default_width, default_height = (1024, 1024)
        elif "xl" in engine_id:
            default_width, default_height = (1024, 1024)
        else:
            default_width, default_height = (512, 512)

        text_prompts = [{"text": image_prompt_text, "weight": 1.0}]
        negative_prompt_text = image_model_configuration.get("negative_prompt")
        if negative_prompt_text:
            text_prompts.append({"text": negative_prompt_text, "weight": -1.0})

        payload = {
            "text_prompts": text_prompts,
            "cfg_scale": image_model_configuration.get("cfg_scale", 7.0),
            "height": image_model_configuration.get("height", default_height),
            "width": image_model_configuration.get("width", default_width),
            "samples": image_model_configuration.get("samples", 1),
            "steps": image_model_configuration.get("steps", 40), 
        }

        if "seed" in image_model_configuration:
            payload["seed"] = image_model_configuration["seed"]
        if "style_preset" in image_model_configuration:
            payload["style_preset"] = image_model_configuration["style_preset"]
        if "sampler" in image_model_configuration:
            payload["sampler"] = image_model_configuration["sampler"]

        print(f"\n[ImageGenSAI] Requesting image(s) from Stability AI engine '{engine_id}'...")
        print(f"  Prompt: '{image_prompt_text[:200]}...'")
        if negative_prompt_text:
            print(f"  Negative Prompt: '{negative_prompt_text[:100]}...'")
        print(f"  Payload (partial): samples={payload['samples']}, HxW=({payload['height']}x{payload['width']}), steps={payload['steps']}, cfg_scale={payload['cfg_scale']}")

        api_url = f"{STABILITY_API_HOST}/v1/generation/{engine_id}/text-to-image"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json", 
            "Authorization": f"Bearer {api_key}"
        }

        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=180 
            )
            response.raise_for_status() 
            
            response_json = response.json()
            
            generated_images_bytes: List[bytes] = []
            if "artifacts" in response_json:
                for artifact in response_json["artifacts"]:
                    if artifact.get("finishReason") == "SUCCESS":
                        image_b64 = artifact.get("base64")
                        if image_b64:
                            generated_images_bytes.append(base64.b64decode(image_b64))
                    elif artifact.get("finishReason") == "CONTENT_FILTERED":
                        print("  [WARN] An image was filtered by Stability AI's content policy.")
                    else:
                        print(f"  [WARN] Artifact has finishReason: {artifact.get('finishReason')}. Artifact details: {artifact}")
            
            if not generated_images_bytes:
                print("  -> No images successfully generated or found in the response.")
                print(f"  Raw API response JSON: {json.dumps(response_json, indent=2)}")
                return None

            print(f"  -> Decoded {len(generated_images_bytes)} image(s) from Stability AI.")
            return generated_images_bytes

        except requests.exceptions.HTTPError as http_err:
            error_message = f"HTTP error occurred: {http_err}"
            if http_err.response is not None:
                try:
                    error_detail = http_err.response.json()
                    error_message += f" - Detail: {json.dumps(error_detail, indent=2)}"
                except ValueError: 
                    error_message += f" - Raw response: {http_err.response.text}"
            print(f"  -> {error_message}")
            return None
        except requests.exceptions.RequestException as req_err: 
            print(f"  -> Request exception occurred: {req_err}")
            return None
        except Exception as e: 
            print(f"  -> An unexpected error occurred: {e}")
            print(traceback.format_exc())
            return None