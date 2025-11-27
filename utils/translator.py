"""
EEVE-Korean translation engine with optimized prompts
"""
from typing import List, Optional
from llama_cpp import Llama
from config import LANGUAGE_MAP, TEMPERATURE, MAX_TOKENS


def translate_segment(
    text: str,
    source_lang: str,
    target_lang: str,
    model: Llama,
    context_history: Optional[List[str]] = None
) -> str:
    """
    Translate a single subtitle segment.

    Args:
        text: Text to translate
        source_lang: Source language code (ja, en, etc.)
        target_lang: Target language code (ko)
        model: Loaded Llama model
        context_history: Optional list of previous segments for context

    Returns:
        Translated text in target language
    """
    source_lang_name = LANGUAGE_MAP.get(source_lang, source_lang)
    target_lang_name = LANGUAGE_MAP.get(target_lang, target_lang)

    # Build context section
    context_section = ""
    if context_history and len(context_history) > 0:
        context_text = "\n".join(context_history[-3:])  # Last 3 segments
        context_section = f"""
[이전 대화 맥락]
{context_text}
"""

    # EEVE-Korean optimized prompt
    prompt = f"""### System:
당신은 전문 자막 번역가입니다. {source_lang_name}를 자연스러운 {target_lang_name}로 번역합니다.

**중요 규칙**:
1. 순수한 한글로만 출력 (한자, 일본어, 중국어 문자 절대 사용 금지)
2. 자연스러운 구어체 사용
3. 존댓말/반말 일관성 유지
4. 번역문만 출력 (설명이나 부가 정보 금지)

### User:{context_section}
다음 {source_lang_name} 자막을 {target_lang_name}로 번역하세요:

{text}

### Assistant:
"""

    try:
        response = model(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stop=["###", "\n\n\n", "User:", "System:"],
            echo=False
        )

        translation = response['choices'][0]['text'].strip()

        # Clean up translation
        translation = translation.replace("번역:", "").strip()
        translation = translation.replace("[번역]", "").strip()

        # Validation: Check for unwanted characters
        if not translation:
            return f"[Translation Error: Empty response]"

        return translation

    except Exception as e:
        return f"[Translation Error: {str(e)}]"


def translate_batch(
    segments: List[str],
    source_lang: str,
    target_lang: str,
    model: Llama,
    context_history: Optional[List[str]] = None
) -> List[str]:
    """
    Translate a batch of segments together for better context.

    Args:
        segments: List of texts to translate
        source_lang: Source language code
        target_lang: Target language code
        model: Loaded Llama model
        context_history: Optional previous segments for context

    Returns:
        List of translated texts
    """
    source_lang_name = LANGUAGE_MAP.get(source_lang, source_lang)
    target_lang_name = LANGUAGE_MAP.get(target_lang, target_lang)

    # Build numbered segment list
    segment_text = "\n".join([f"[{i+1}] {seg}" for i, seg in enumerate(segments)])

    # Build context
    context_section = ""
    if context_history and len(context_history) > 0:
        context_text = "\n".join(context_history[-3:])
        context_section = f"""
[이전 대화 맥락]
{context_text}
"""

    # Batch translation prompt
    prompt = f"""### System:
당신은 전문 자막 번역가입니다. {source_lang_name}를 자연스러운 {target_lang_name}로 번역합니다.

**중요 규칙**:
1. 순수한 한글로만 출력 (한자 ❌, 일본어 문자 ❌, 중국어 문자 ❌)
2. 자연스러운 구어체 사용
3. 대화의 흐름과 맥락 유지
4. 출력 형식: [번호] 번역문

### User:{context_section}
다음 {source_lang_name} 자막들을 {target_lang_name}로 번역하세요:

{segment_text}

### Assistant:
"""

    try:
        response = model(
            prompt,
            max_tokens=MAX_TOKENS * len(segments),
            temperature=TEMPERATURE,
            stop=["###", "\n\n\n\n", "User:", "System:"],
            echo=False
        )

        translation_text = response['choices'][0]['text'].strip()

        # Parse numbered translations
        translations = []
        lines = [line.strip() for line in translation_text.split('\n') if line.strip()]

        for i, segment in enumerate(segments):
            translation = None

            # Try to find translation by number
            for line in lines:
                if line.startswith(f"[{i+1}]"):
                    translation = line.split(']', 1)[-1].strip()
                    break

            # Fallback: use line by index
            if translation is None and i < len(lines):
                line = lines[i]
                if line.startswith('['):
                    translation = line.split(']', 1)[-1].strip()
                else:
                    translation = line

            translations.append(translation if translation else segment)

        return translations

    except Exception as e:
        # Fallback to individual translation
        return [f"[Batch Error: {str(e)}]" for _ in segments]
