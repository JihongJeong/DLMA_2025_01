import json
import re
from typing import List, Dict, Any, Optional
import google.generativeai as genai

class WebtoonElementExtractor:
    def __init__(self, gemini_model: Optional[genai.GenerativeModel] = None):
        self.model = gemini_model
        if self.model:
            print("WebtoonElementExtractor initialized with Gemini Model.")
        else:
            print("WebtoonElementExtractor initialized W/O Gemini Model (will use fallback for some operations).")
        self.character_id_counter = 0
        self.dialogue_id_counter = 0
        self.scene_id_counter = 0 

    def _get_new_character_id(self) -> str:
        self.character_id_counter += 1
        return f"char_{self.character_id_counter:03d}"

    def _get_new_dialogue_id(self, cut_id_prefix: str) -> str:
        self.dialogue_id_counter +=1
        return f"dlg_{cut_id_prefix}_{self.dialogue_id_counter:03d}"

    def _get_new_scene_id(self) -> str:
        self.scene_id_counter += 1
        return f"cut_{self.scene_id_counter:03d}"

    def _reset_dialogue_id_counter_for_cut(self):
        self.dialogue_id_counter = 0

    def _reset_scene_id_counter(self): 
        self.scene_id_counter = 0

    def _call_gemini(self, prompt: str, task_description: str) -> Any:
        if not self.model:
            return None
        try:
            response = self.model.generate_content(prompt)
            if not response.parts:
                print(f"  -> Gemini API response for {task_description} has no parts (possibly due to safety settings or empty response).")
                return [] if task_description == "scene segmentation" or "dialogue separation" in task_description or "speech bubble guidance" in task_description else {}
            
            raw_response_text = response.text
            clean_json_text = raw_response_text.strip()
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_json_text, re.IGNORECASE)
            if match:
                clean_json_text = match.group(1).strip() 
            else:
                if clean_json_text.startswith("```") and clean_json_text.endswith("```"):
                    clean_json_text = clean_json_text[3:-3].strip()
            
            return json.loads(clean_json_text.strip())
        except json.JSONDecodeError as je:
            print(f"  -> JSONDecodeError parsing response for {task_description}: {je}. Response was: {raw_response_text[:500]}")
            return [] if task_description == "scene segmentation" or "dialogue separation" in task_description or "speech bubble guidance" in task_description else {}
        except Exception as e:
            print(f"  -> Error calling Gemini API or parsing response for {task_description}: {e}")
            return [] if task_description == "scene segmentation" or "dialogue separation" in task_description or "speech bubble guidance" in task_description else {}

    def segment_scenes(self, novel_text: str) -> List[Dict[str, Any]]:
        """
        전체 웹소설 텍스트를 의미론적 단위인 개별 컷(장면)으로 분할합니다.
        """
        self._reset_scene_id_counter()
        print(f"\n[Extractor] Segmenting novel text into scenes...")
        prompt = f"""
        주어진 "웹소설 텍스트"를 웹툰의 개별 컷(장면)으로 분할해주세요. 
        각 컷은 시간, 장소, 주요 사건 또는 서술의 전환점을 기준으로 구분되어야 합니다.
        분할된 각 컷에 대해 다음 정보를 포함하는 JSON 객체를 생성하고, 이 객체들의 리스트로 응답해주세요:
        - `id_placeholder`: 각 컷의 임시 ID placeholder (실제 ID는 나중에 부여됨). LLM은 이 필드를 그냥 "temp_id_X" 와 같이 두어도 됩니다.
        - `text`: 해당 컷의 내용인 소설 텍스트.

        [웹소설 텍스트]
        ---
        {novel_text}
        ---

        [응답 형식] 
        **반드시 다음 예시와 같이 대괄호 `[` 와 `]` 로 전체를 감싼 JSON 배열 형식으로만 응답해주십시오.**
        각 JSON 객체는 "id_placeholder"와 "text" 키만을 가져야 합니다.
        예시:
        [
        {{"id_placeholder": "temp_id_1", "text": "첫 번째 컷의 내용입니다..."}},
        {{"id_placeholder": "temp_id_2", "text": "두 번째 컷의 내용..."}}
        ]
        """
        llm_response_list = self._call_gemini(prompt, "scene segmentation")
        print(llm_response_list, "\n")
        final_scenes = []
        if isinstance(llm_response_list, list):
            for scene_data in llm_response_list:
                if isinstance(scene_data, dict) and 'text' in scene_data:
                    final_scenes.append({
                        "id": self._get_new_scene_id(), 
                        "text": scene_data['text']
                    })
        if not final_scenes and not self.model: 
            final_scenes = [{"id": self._get_new_scene_id(), "text": "Wrong Response"}]
        elif not final_scenes and self.model: 
             print("  Warning: Scene segmentation by LLM resulted in empty list. Check LLM response or prompt.")
             final_scenes = [{"id": self._get_new_scene_id(), "text": "LLM으로부터 장면 분할 정보를 받지 못했습니다."}]

        print(f"-> Segmented into {len(final_scenes)} scenes.")
        return final_scenes

    def configure_characters(self,
                             cut_text: str,
                             scene_context: str,
                             existing_character_database: Dict[str, Dict[str, Any]]
                            ) -> List[Dict[str, Any]]:
        print(f"  Configuring characters for cut: '{cut_text[:30]}...' with scene context.")
        db_summary_parts = []
        for char_id, char_data in existing_character_database.items():
            aliases_str = f"(별칭: {', '.join(char_data.get('aliases',[]))})" if char_data.get('aliases') else ""
            db_summary_parts.append(
                f"- ID: {char_id}, 이름: {char_data.get('name', '알 수 없음')} {aliases_str}, 주요 특징: {char_data.get('appearance', '')}, {char_data.get('outfit', '')}"
            )
        db_summary = "\n".join(db_summary_parts) if db_summary_parts else "없음"

        prompt = f"""
        당신은 소설을 분석하여 등장인물의 연속성을 파악하고 상세 정보를 추출하는 전문가입니다.

        [지시사항]
        1.  아래 제공되는 "현재 컷 텍스트"와 "장면 문맥"을 바탕으로 등장하는 모든 인물을 식별합니다.
        2.  "기존 캐릭터 데이터베이스"를 참조하여, 현재 컷의 인물이 데이터베이스의 기존 인물과 동일한지 판단합니다.
            -   동일 인물 판단 기준: 이름, 별칭, 외모/행동 묘사의 일관성, 문맥적 흐름 등.
            -   "의문의 여인", "그녀"와 같은 대명사나 일반 명사가 특정 인물을 지칭한다면, 해당 인물 정보에 포함시키세요.
        3.  각 인물에 대해 다음 정보를 포함하는 JSON 객체를 생성합니다:
            -   `id`: 캐릭터의 고유 ID. 기존 인물과 동일하면 해당 ID를 사용하고, 새로운 인물이면 "NEW"라고 표시합니다.
            -   `name`: 인물의 주된 이름. (예: "이영희")
            -   `aliases`: 작중 사용된 다른 이름이나 지칭 표현 리스트. (예: ["의문의 여인", "그녀"])
            -   `appearance`: 외모 묘사. (기존 정보에 추가/보강)
            -   `outfit`: 복장 묘사. (기존 정보에 추가/보강)
            -   `expression`: 표정.
            -   `emotion`: 감정 상태.
            -   `action`: 주요 행동.
            -   `is_new_character_suggestion`: LLM이 판단하기에 새로운 캐릭터이면 true, 기존 캐릭터의 업데이트이면 false.
            -   `reasoning`: 해당 인물 정보 추출 및 동일 인물 판단(또는 신규 인물 판단)에 대한 간략한 근거.
            -   `confidence_for_merge`: 만약 기존 캐릭터와 동일하다고 판단했다면, 그 판단에 대한 신뢰도 (0.0 ~ 1.0). 새로운 캐릭터는 0.0.
            -   `identified_in_current_cut`: 현재 컷에서 직접적으로 언급/묘사된 정보인지 여부 (true).

        [기존 캐릭터 데이터베이스]
        {db_summary}

        [장면 문맥 (이전 상황 또는 전체 줄거리 요약)]
        {scene_context}

        [현재 컷 텍스트]
        {cut_text}

        [응답 형식] JSON 객체 리스트로 응답해주세요.
        """
        llm_response_list = self._call_gemini(prompt, "character configuration with continuity")

        processed_characters = []
        if isinstance(llm_response_list, list):
            for char_info_from_llm in llm_response_list:
                if not isinstance(char_info_from_llm, dict): continue

                char_id = char_info_from_llm.get('id')
                
                if char_id == "NEW" or not isinstance(char_id, str) or not char_id.startswith("char_"):
                    final_char_id = self._get_new_character_id()
                    char_info_from_llm['id'] = final_char_id
                    char_info_from_llm['is_actually_new'] = True
                elif char_id not in existing_character_database: 
                    final_char_id = self._get_new_character_id() 
                    char_info_from_llm['id'] = final_char_id
                    char_info_from_llm['is_actually_new'] = True
                    print(f"  Warning: LLM provided ID '{char_id}' not in DB. Treated as new: {final_char_id}")
                else:
                    final_char_id = char_id
                    char_info_from_llm['is_actually_new'] = False

                processed_characters.append(char_info_from_llm)
        return processed_characters

    def configure_composition(self, cut_text: str, scene_context: Optional[str] = "") -> Dict[str, Any]:
        print(f"  Configuring composition for cut: '{cut_text[:30]}...'")
        prompt = f"""
        다음 "현재 컷 텍스트"와 "장면 문맥"을 바탕으로 웹툰 컷의 시각적 연출 및 구도를 설정해주세요.
        응답은 다음 키를 포함하는 JSON 객체로 해주세요: "camera_angle", "shot_type" (예: 클로즈업, 풀샷, 버스트샷 등), "character_placement" (등장인물들의 화면 내 주요 배치), "focus_element" (해당 컷에서 시각적으로 가장 강조되어야 할 요소).

        [장면 문맥]
        {scene_context}

        [현재 컷 텍스트]
        {cut_text}

        [응답 형식] JSON 객체.
        """
        result = self._call_gemini(prompt, "composition configuration")
        return result if isinstance(result, dict) else {}

    def configure_background(self, cut_text: str, scene_context: Optional[str] = "") -> Dict[str, Any]:
        print(f"  Configuring background for cut: '{cut_text[:30]}...'")
        prompt = f"""
        다음 "현재 컷 텍스트"와 "장면 문맥"을 바탕으로 웹툰 컷의 배경 정보를 상세히 설정해주세요.
        응답은 다음 키를 포함하는 JSON 객체로 해주세요: "location_type" (실내/실외), "specific_place" (구체적인 장소명 또는 묘사), "time_of_day" (시간대), "weather" (날씨, 해당될 경우), "key_props" (배경에 등장하는 주요 소품 리스트), "atmosphere" (전체적인 분위기).

        [장면 문맥]
        {scene_context}

        [현재 컷 텍스트]
        {cut_text}

        [응답 형식] JSON 객체.
        """
        result = self._call_gemini(prompt, "background configuration")
        return result if isinstance(result, dict) else {}

    def separate_dialogues(self, cut_text: str, cut_id_prefix: str) -> List[Dict[str, Any]]:
        self._reset_dialogue_id_counter_for_cut()
        print(f"  Separating dialogues for cut: '{cut_text[:30]}...'")
        prompt = f"""
        다음 "현재 컷 텍스트"에서 직접 인용된 모든 대사("...")를 추출해주세요.
        각 대사에 대해 다음 정보를 포함하는 JSON 객체를 생성하고, 이 객체들의 리스트로 응답해주세요:
        - `id_placeholder`: 각 대사의 임시 ID placeholder (실제 ID는 나중에 부여됨).
        - `speaker_name_guess`: 해당 대사의 화자로 추정되는 인물의 이름. (정확한 이름이 안나오면 "남자1", "목소리" 등으로 추정)
        - `text`: 실제 대사 내용.
        - `nuance`: 대사의 어조나 내포된 감정 (예: 외침, 속삭임, 단호함, 기쁨, 슬픔 등).

        [현재 컷 텍스트]
        {cut_text}

        [응답 형식] JSON 객체 리스트.
        """
        llm_response_list = self._call_gemini(prompt, "dialogue separation")
        
        final_dialogues = []
        if isinstance(llm_response_list, list):
            for dlg_info in llm_response_list:
                if isinstance(dlg_info, dict):
                    dlg_info['id'] = self._get_new_dialogue_id(cut_id_prefix)
                    final_dialogues.append(dlg_info)
        return final_dialogues


    def guide_speech_bubble_placement(self,
                                      dialogues_in_cut: List[Dict[str, Any]],
                                      characters_in_cut: List[Dict[str, Any]],
                                      composition_settings: Dict[str, Any]
                                     ) -> List[Dict[str, Any]]:
        if not dialogues_in_cut:
            return []
        print(f"  Guiding speech bubble placement for {len(dialogues_in_cut)} dialogues.")

        char_summary_for_bubbles = []
        for char_info in characters_in_cut:
            char_summary_for_bubbles.append(
                f"캐릭터 ID: {char_info.get('id')}, 이름: {char_info.get('name')}, 현재 행동/위치 단서: {char_info.get('action')} ({char_info.get('expression', '')} 표정)"
            )
        char_summary_str = "\n".join(char_summary_for_bubbles)

        dialogue_summary_for_bubbles = []
        for dlg_info in dialogues_in_cut:
            dialogue_summary_for_bubbles.append(
                f"대사 ID: {dlg_info.get('id')}, 화자 ID(추정): {dlg_info.get('speaker_id', dlg_info.get('speaker_name_guess', '미상'))}, 내용: \"{dlg_info.get('text')}\", 뉘앙스: {dlg_info.get('nuance')}"
            )
        dialogue_summary_str = "\n".join(dialogue_summary_for_bubbles)

        prompt = f"""
        웹툰 컷의 말풍선 배치를 위한 가이드를 생성합니다. 아래 제공된 "현재 컷의 캐릭터 정보", "현재 컷의 대사 정보", "구도 설정"을 종합적으로 고려해주세요.
        각 대사별로 말풍선의 시각적 요소에 대한 제안을 JSON 객체 리스트로 응답해주세요. 각 객체는 다음 키를 포함해야 합니다:
        - `dialogue_id`: 해당하는 대사의 ID.
        - `speaker_ref_id`: 해당 대사의 화자 캐릭터 ID. (위 "현재 컷의 캐릭터 정보"에서 ID 참조)
        - `suggested_area`: 말풍선이 위치할 대략적인 영역 (예: "캐릭터 A의 우측 상단", "화면 하단 중앙"). 캐릭터의 실제 위치, 주요 시각 요소와의 관계를 고려.
        - `bubble_style_hint`: 말풍선의 모양이나 스타일에 대한 힌트 (예: "일반형", "생각형 구름모양", "뾰족한 외침형", "떨리는 선"). 대사의 뉘앙스 반영.
        - `tail_direction`: 말풍선 꼬리가 향해야 할 방향 (예: "화자 입 방향", "화자 머리 근처").

        [현재 컷의 캐릭터 정보]
        {char_summary_str}

        [현재 컷의 대사 정보]
        {dialogue_summary_str}

        [구도 설정]
        {json.dumps(composition_settings, ensure_ascii=False)}

        [응답 형식] JSON 객체 리스트.
        """
        result = self._call_gemini(prompt, "speech bubble guidance")
        return result if isinstance(result, list) else []

    def process_single_cut_text(self,
                                cut_id: str,
                                cut_text: str,
                                scene_context: str,
                                character_database: Dict[str, Dict[str, Any]]
                               ) -> Dict[str, Any]:
        print(f"\n[Extractor] Processing cut_id: {cut_id}")

        current_cut_characters_info = self.configure_characters(cut_text, scene_context, character_database)
        
        active_characters_in_cut = []
        for char_data_llm in current_cut_characters_info:
            char_id = char_data_llm['id'] 

            if char_data_llm.get('is_actually_new'):
                character_database[char_id] = {
                    "id": char_id, "name": char_data_llm.get('name'),
                    "aliases": list(set(char_data_llm.get('aliases', []))),
                    "appearance": char_data_llm.get('appearance'), "outfit": char_data_llm.get('outfit'),
                    "first_seen_cut": cut_id, "last_seen_cut": cut_id,
                    "all_actions": {cut_id: char_data_llm.get('action')},
                    "all_emotions": {cut_id: char_data_llm.get('emotion')}
                }
                print(f"  New character '{char_data_llm.get('name')}' (ID: {char_id}) added to database.")
            elif char_id in character_database:
                existing_char = character_database[char_id]
                existing_char["name"] = char_data_llm.get("name", existing_char.get("name"))
                new_aliases = set(existing_char.get("aliases", []))
                new_aliases.update(char_data_llm.get("aliases", []))
                existing_char["aliases"] = list(new_aliases)
                for key in ["appearance", "outfit"]:
                    if char_data_llm.get(key): existing_char[key] = char_data_llm.get(key)
                
                existing_char["last_seen_cut"] = cut_id
                if "all_actions" not in existing_char: existing_char["all_actions"] = {}
                if "all_emotions" not in existing_char: existing_char["all_emotions"] = {}
                existing_char["all_actions"][cut_id] = char_data_llm.get('action')
                existing_char["all_emotions"][cut_id] = char_data_llm.get('emotion')
                print(f"  Character '{existing_char.get('name')}' (ID: {char_id}) updated in database.")
            else:
                 print(f"  Error: Character {char_id} marked as existing but not found in DB. Skipping update for this entry.")
            
            final_char_representation_for_cut = {
                "id": char_id,
                "name": character_database.get(char_id, {}).get('name', char_data_llm.get('name')),
                "appearance": character_database.get(char_id, {}).get('appearance', char_data_llm.get('appearance')),
                "outfit": character_database.get(char_id, {}).get('outfit', char_data_llm.get('outfit')),
                "expression": char_data_llm.get('expression'),
                "emotion": char_data_llm.get('emotion'),
                "action": char_data_llm.get('action')
            }
            active_characters_in_cut.append(final_char_representation_for_cut)

        composition = self.configure_composition(cut_text, scene_context)
        background = self.configure_background(cut_text, scene_context)
        
        cut_id_prefix_for_dialogue = cut_id.replace("cut_", "") 
        dialogues_raw = self.separate_dialogues(cut_text, cut_id_prefix_for_dialogue)
        
        updated_dialogues = []
        for dlg_info in dialogues_raw:
            speaker_name_guess = dlg_info.get('speaker_name_guess')
            speaker_id_assigned = None
            if speaker_name_guess:
                for active_char in active_characters_in_cut: 
                    char_db_entry = character_database.get(active_char.get('id')) 
                    if char_db_entry: 
                        if char_db_entry.get('name') == speaker_name_guess or \
                        speaker_name_guess in char_db_entry.get('aliases', []):
                            speaker_id_assigned = active_char.get('id')
                            break
                if not speaker_id_assigned: 
                    for db_char_id, db_char_data in character_database.items():
                        if db_char_data.get('name') == speaker_name_guess or \
                           speaker_name_guess in db_char_data.get('aliases', []):
                            speaker_id_assigned = db_char_id
                            break
            dlg_info['speaker_id'] = speaker_id_assigned 
            updated_dialogues.append(dlg_info)
            
        speech_bubbles = self.guide_speech_bubble_placement(updated_dialogues, active_characters_in_cut, composition)

        return {
            "cut_id": cut_id,
            "original_text_for_cut": cut_text,
            "characters": active_characters_in_cut,
            "composition": composition,
            "background": background,
            "dialogues": updated_dialogues,
            "speech_bubble_guidance": speech_bubbles,
        }