# image_prompt_generator.py
from typing import Dict, Any, Optional
import google.generativeai as genai # 만약 이 모듈도 LLM을 직접 사용한다면 필요

class ImagePromptGenerator:
    def __init__(self, gemini_model: Optional[genai.GenerativeModel] = None):
        self.model = gemini_model # LLM을 활용한 프롬프트 최적화 등에 사용 가능
        if self.model:
            print("ImagePromptGenerator initialized with Gemini Model.")
        else:
            print("ImagePromptGenerator initialized W/O Gemini Model (will use rule-based generation).")


    def generate_prompt(self, extracted_elements_for_cut: Dict[str, Any], 
                        global_style_settings: Optional[Dict[str, str]] = None) -> str:
        print(f"\n[PromptGen] Generating image prompt for cut_id: {extracted_elements_for_cut.get('cut_id')}")
        
        char_descs = []
        if extracted_elements_for_cut.get('characters'):
            for char in extracted_elements_for_cut['characters']:
                desc = f"{char.get('name', 'Unknown character')} ({char.get('appearance', 'N/A')}, wearing {char.get('outfit', 'N/A')}) is {char.get('action', 'N/A')} with an expression of {char.get('expression', 'N/A')}."
                char_descs.append(desc)
        
        comp = extracted_elements_for_cut.get('composition', {})
        bg = extracted_elements_for_cut.get('background', {})

        prompt_core_parts = []
        if char_descs: prompt_core_parts.append(", ".join(char_descs))
        
        comp_desc = f"Scene composition: {comp.get('shot_type', 'medium shot')}, {comp.get('camera_angle', 'eye level')}"
        if comp.get('focus_element'): comp_desc += f", focusing on {comp.get('focus_element')}"
        prompt_core_parts.append(comp_desc + ".")

        bg_desc = f"Background: {bg.get('specific_place', 'a generic place')} ({bg.get('location_type', 'outdoor')}) at {bg.get('time_of_day', 'daytime')}"
        if bg.get('weather'): bg_desc += f", weather is {bg.get('weather')}"
        if bg.get('key_props'): bg_desc += f". Key props: {', '.join(bg.get('key_props',[]))}"
        if bg.get('atmosphere'): bg_desc += f". Overall atmosphere: {bg.get('atmosphere')}"
        prompt_core_parts.append(bg_desc + ".")
        
        prompt_core = " ".join(prompt_core_parts)

        style_prompt_parts = []
        if global_style_settings:
            if global_style_settings.get('art_style'):
                style_prompt_parts.append(f"Art style: {global_style_settings['art_style']}.")
            if global_style_settings.get('color_palette'):
                style_prompt_parts.append(f"Color palette: {global_style_settings['color_palette']}.")
        
        final_image_prompt = prompt_core + " " + " ".join(style_prompt_parts)

        for_better_prompt = f"""
        [지시사항]
        기존 프롬프트를 이용해서 Text2Image 모델에 사용하기 적합한 프롬프트를 생성합니다.
        
        [기존 프롬프트]
        {final_image_prompt}\n
        
        [응답 형식] 
        **반드시 프롬프트 텍스트만을 출력해 주십시오.**

        """
        enhanced_prompt = self.model.generate_content(for_better_prompt).text
        # enhanced_prompt = enhanced_prompt.replace("최종 프롬프트:", "").strip()
        print(f"  -> Generated Prompt (first 100 chars): {enhanced_prompt[:100].strip()}...")
        return enhanced_prompt.strip()