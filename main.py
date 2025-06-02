import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import config
from webtoon_element_extractor import WebtoonElementExtractor
from image_prompt_generator import ImagePromptGenerator
from image_generator import ImageGenerator
from final_cut_composer import FinalCutComposer

def webnovel_to_webtoon_pipeline(
    novel_text: str,
    gemini_api_key: str,
    gemini_model_name: str,
    global_webtoon_style: Optional[Dict[str, str]] = None,
    image_gen_config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    웹소설 텍스트를 입력받아 웹툰 컷 리스트를 생성하는 전체 파이프라인입니다.
    """
    print("🚀 STARTING WEBNOVEL-TO-WEBTOON PIPELINE 🚀")

    gemini_llm_model = None
    if gemini_api_key and gemini_api_key != "YOUR_GEMINI_API_KEY_HERE":
        try:
            genai.configure(api_key=gemini_api_key)
            gemini_llm_model = genai.GenerativeModel(gemini_model_name)
            print(f"Gemini model '{gemini_model_name}' initialized successfully.")
        except Exception as e:
            print(f"Error initializing Gemini model: {e}. Proceeding without LLM for some operations.")
            return []
    else:
        print("Gemini API key not provided or invalid. Proceeding without LLM for some operations.")


    extractor = WebtoonElementExtractor(gemini_model=gemini_llm_model)
    prompt_gen = ImagePromptGenerator(gemini_model=gemini_llm_model)
    img_gen = ImageGenerator() 
    composer = FinalCutComposer(gemini_model=gemini_llm_model) 

    current_global_style = global_webtoon_style if global_webtoon_style is not None else config.GLOBAL_WEBTOON_STYLE
    current_image_config = image_gen_config if image_gen_config is not None else config.IMAGE_GEN_CONFIG

    # 1. 입력 텍스트 분석 및 웹툰 요소 추출 (컷 분할)
    segmented_scenes = extractor.segment_scenes(novel_text)

    completed_webtoon_cuts = []
    character_database: Dict[str, Dict[str, Any]] = {} # 캐릭터 정보 누적

    initial_scene_context = novel_text[:500] # 예시로 처음 500자

    for scene_info in segmented_scenes:
        cut_id = scene_info.get('id', f'unknown_cut_{i}')
        original_text = scene_info.get('text', '')

        previous_cut_text = segmented_scenes[i-1]['text'] if i > 0 else ""
        current_scene_context = f"이전 컷 내용: {previous_cut_text}\n\n현재 분석할 장면의 전체적인 문맥: {initial_scene_context}"
        
        print(f"\n--- Processing Cut ID: {cut_id} ---")
        
        extracted_elements = extractor.process_single_cut_text(
            cut_id,
            original_text,
            current_scene_context, 
            character_database 
        )
        
        image_prompt = prompt_gen.generate_prompt(extracted_elements, current_global_style)
        generated_image = img_gen.create_image_from_prompt(image_prompt, current_image_config)
        final_cut_data = composer.add_dialogues_to_image(
            base_image_data=generated_image,
            dialogues_for_cut=extracted_elements.get('dialogues', []),
            speech_bubble_guidance_for_cut=extracted_elements.get('speech_bubble_guidance', []),
        )

        completed_webtoon_cuts.append({
            "cut_id": cut_id,
            "processed_elements": extracted_elements,
            "image_prompt": image_prompt,
            "generated_image_mock_len": len(generated_image), # 실제 이미지 대신 길이만 저장
            "final_cut_data_mock_len": len(final_cut_data)    # 실제 데이터 대신 길이만 저장
        })
        
        print(f"✅ Completed processing for cut_id: {cut_id}")
        print(f"  Updated Character DB keys: {list(character_database.keys())}") # DB 상태 확인용

        # 실제 데이터 저장시:
        # "generated_image": generated_image,
        # "final_cut_data": final_cut_data

    print("\n🎉 WEBNOVEL-TO-WEBTOON PIPELINE COMPLETED 🎉")
    return completed_webtoon_cuts


if __name__ == '__main__':
    sample_novel_text = """
    오후 세 시, 탐정 사무소의 문이 조용히 열렸다. 들어선 것은 긴 트렌치코트를 입은 의문의 여인이었다. 
    그녀의 이름은 이영희. "김철수 탐정님... 제 고양이를 찾아주세요." 그녀의 목소리는 절박했다. 
    김철수는 책상에서 일어나 그녀를 맞이했다. 그의 날카로운 눈빛이 그녀를 훑었다. "고양이라... 자세히 말씀해주시죠." 
    사무실 창밖으로는 비가 내리기 시작했다. 분위기는 점점 더 무거워졌다. 
    이영희는 떨리는 목소리로 말을 이었다. "그 아이의 이름은... 루비예요. 어제 저녁부터 보이지 않아요." 
    그녀의 얼굴이 화면 가득 클로즈업되었다. 눈가에는 눈물이 그렁그렁했다.
    """

    # config.py 또는 환경 변수에서 API 키를 가져옵니다.
    api_key = config.GEMINI_API_KEY
    model_name = config.GEMINI_MODEL_NAME
    
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        print("메인 실행: Gemini API 키가 설정되지 않았습니다. LLM 기능 없이 실행되거나 제한적으로 동작할 수 있습니다.")

    
    webtoon_cuts_output = webnovel_to_webtoon_pipeline(
        novel_text=sample_novel_text,
        gemini_api_key=api_key,
        gemini_model_name=model_name,
        global_webtoon_style=config.GLOBAL_WEBTOON_STYLE,
        image_gen_config=config.IMAGE_GEN_CONFIG
    )

    print("\n--- 최종 생성된 웹툰 컷 요약 ---")
    for i, cut_data in enumerate(webtoon_cuts_output):
        print(f"\n--- 컷 #{i+1} (ID: {cut_data.get('cut_id', 'N/A')}) ---")
        if "error" in cut_data:
            print(f"  Error: {cut_data['error']}")
            continue
        
        elements = cut_data.get('processed_elements', {})
        print(f"  Original Text Snippet: {elements.get('original_text_for_cut', 'N/A')[:50]}...")
        
        characters = elements.get('characters')
        if characters:
             print(f"  Characters: {[(c.get('name', 'N/A'), c.get('action', 'N/A')) for c in characters if c]}")
        else:
            print(f"  Characters: 정보 없음")

        dialogues = elements.get('dialogues')
        if dialogues:
            print(f"  Dialogues: {[(d.get('speaker', 'N/A'), d.get('text', 'N/A')) for d in dialogues if d]}")
        else:
            print(f"  Dialogues: 정보 없음")
            
        print(f"  Image Prompt Snippet: {cut_data.get('image_prompt', 'N/A')[:70]}...")
        print(f"  Generated Image (mocked length): {cut_data.get('generated_image_mock_len', 'N/A')}")
        print(f"  Final Cut Data (mocked length): {cut_data.get('final_cut_data_mock_len', 'N/A')}")