"""
EEVE-Korean translation engine with optimized prompts
"""
from typing import List, Optional, Tuple
from llama_cpp import Llama
from config import LANGUAGE_MAP, TEMPERATURE, MAX_TOKENS, BEST_OF_N, SAMPLING_TEMPERATURE
import re


def _score_translation(translation: str, original: str) -> float:
    """
    Score a translation candidate based on quality heuristics.

    Args:
        translation: Translated text
        original: Original text

    Returns:
        Quality score (higher is better)
    """
    score = 0.0

    # Penalty for empty translations
    if not translation or len(translation.strip()) == 0:
        return -1000.0

    # Penalty for error messages
    if translation.startswith("[") and "Error" in translation:
        return -1000.0

    # Penalty for untranslated text (same as original)
    if translation.strip() == original.strip():
        score -= 50.0

    # Penalty for non-Korean characters (Chinese/Japanese)
    non_korean_chars = len(re.findall(r'[一-龯ぁ-んァ-ン]', translation))
    score -= non_korean_chars * 10.0

    # Bonus for reasonable length (not too short, not too long)
    length_ratio = len(translation) / max(len(original), 1)
    if 0.5 <= length_ratio <= 2.0:
        score += 10.0
    else:
        score -= abs(length_ratio - 1.0) * 5.0

    # Bonus for complete sentences (ends with proper punctuation)
    if translation.rstrip()[-1:] in '.!?。！？':
        score += 5.0

    return score


def _generate_candidates(
    prompt: str,
    model: Llama,
    n_candidates: int,
    temperature: float
) -> List[Tuple[str, float]]:
    """
    Generate multiple translation candidates.

    Args:
        prompt: Translation prompt
        model: Loaded Llama model
        n_candidates: Number of candidates to generate
        temperature: Sampling temperature

    Returns:
        List of (translation, score) tuples
    """
    candidates = []

    for i in range(n_candidates):
        try:
            response = model(
                prompt,
                max_tokens=MAX_TOKENS,
                temperature=temperature if i > 0 else TEMPERATURE,  # First one uses base temp
                stop=["###", "\n\n\n", "User:", "System:"],
                echo=False,
                logprobs=1  # Request log probabilities for scoring
            )

            translation = response['choices'][0]['text'].strip()

            # Clean up translation
            translation = translation.replace("번역:", "").strip()
            translation = translation.replace("[번역]", "").strip()

            # Calculate score from logprobs if available
            logprob_score = 0.0
            if 'logprobs' in response['choices'][0] and response['choices'][0]['logprobs']:
                # Use average log probability as part of score
                logprobs_data = response['choices'][0]['logprobs']
                if logprobs_data and 'token_logprobs' in logprobs_data:
                    token_logprobs = [lp for lp in logprobs_data['token_logprobs'] if lp is not None]
                    if token_logprobs:
                        logprob_score = sum(token_logprobs) / len(token_logprobs)

            candidates.append((translation, logprob_score))

        except Exception as e:
            # If generation fails, add error candidate with low score
            candidates.append((f"[Generation Error: {str(e)}]", -1000.0))

    return candidates


def _select_best_candidate(
    candidates: List[Tuple[str, float]],
    original_text: str
) -> str:
    """
    Select the best translation from candidates.

    Args:
        candidates: List of (translation, logprob_score) tuples
        original_text: Original text for quality scoring

    Returns:
        Best translation
    """
    if not candidates:
        return "[Translation Error: No candidates generated]"

    # Score each candidate with combined metrics
    scored_candidates = []
    for translation, logprob_score in candidates:
        quality_score = _score_translation(translation, original_text)
        # Combine logprob score (normalized) with quality score
        combined_score = (logprob_score * 10.0) + quality_score
        scored_candidates.append((translation, combined_score))

    # Sort by score (descending)
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    # Return best candidate
    return scored_candidates[0][0]


def _parse_batch_translations(translation_text: str, segments: List[str]) -> List[str]:
    """
    Parse numbered batch translations into a list.

    Args:
        translation_text: Raw translation output with numbered segments
        segments: Original segments for fallback

    Returns:
        List of parsed translations
    """
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


def translate_segment(
    text: str,
    source_lang: str,
    target_lang: str,
    model: Llama,
    context_before: Optional[List[str]] = None,
    context_after: Optional[List[str]] = None,
    best_of_n: int = None
) -> str:
    """
    Translate a single subtitle segment with bidirectional context.

    Args:
        text: Text to translate
        source_lang: Source language code (ja, en, etc.)
        target_lang: Target language code (ko)
        model: Loaded Llama model
        context_before: Optional list of previous segments for context
        context_after: Optional list of following segments for context
        best_of_n: Number of candidates to generate (None = use config default)

    Returns:
        Translated text in target language
    """
    if best_of_n is None:
        best_of_n = BEST_OF_N
    source_lang_name = LANGUAGE_MAP.get(source_lang, source_lang)
    target_lang_name = LANGUAGE_MAP.get(target_lang, target_lang)

    # Build context section
    context_section = ""
    if context_before and len(context_before) > 0:
        context_text = "\n".join(context_before)
        context_section += f"""
[이전 대화 맥락]
{context_text}
"""

    if context_after and len(context_after) > 0:
        context_text = "\n".join(context_after)
        context_section += f"""
[이후 대화 맥락]
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
        # Best-of-N sampling
        if best_of_n > 1:
            candidates = _generate_candidates(prompt, model, best_of_n, SAMPLING_TEMPERATURE)
            translation = _select_best_candidate(candidates, text)
        else:
            # Single generation (original behavior)
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
    context_before: Optional[List[str]] = None,
    context_after: Optional[List[str]] = None,
    best_of_n: int = None
) -> List[str]:
    """
    Translate a batch of segments together with bidirectional context.

    Args:
        segments: List of texts to translate
        source_lang: Source language code
        target_lang: Target language code
        model: Loaded Llama model
        context_before: Optional previous segments for context
        context_after: Optional following segments for context
        best_of_n: Number of candidates to generate (None = use config default)

    Returns:
        List of translated texts
    """
    if best_of_n is None:
        best_of_n = BEST_OF_N
    source_lang_name = LANGUAGE_MAP.get(source_lang, source_lang)
    target_lang_name = LANGUAGE_MAP.get(target_lang, target_lang)

    # Build numbered segment list
    segment_text = "\n".join([f"[{i+1}] {seg}" for i, seg in enumerate(segments)])

    # Build context
    context_section = ""
    if context_before and len(context_before) > 0:
        context_text = "\n".join(context_before)
        context_section += f"""
[이전 대화 맥락]
{context_text}
"""

    if context_after and len(context_after) > 0:
        context_text = "\n".join(context_after)
        context_section += f"""
[이후 대화 맥락]
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
        # Best-of-N sampling for batch
        if best_of_n > 1:
            batch_candidates = []

            for i in range(best_of_n):
                response = model(
                    prompt,
                    max_tokens=MAX_TOKENS * len(segments),
                    temperature=SAMPLING_TEMPERATURE if i > 0 else TEMPERATURE,
                    stop=["###", "\n\n\n\n", "User:", "System:"],
                    echo=False,
                    logprobs=1
                )

                translation_text = response['choices'][0]['text'].strip()
                translations = _parse_batch_translations(translation_text, segments)

                # Calculate average score for this batch
                avg_score = 0.0
                for j, (original, translated) in enumerate(zip(segments, translations)):
                    quality_score = _score_translation(translated, original)
                    avg_score += quality_score

                # Get logprob score if available
                logprob_score = 0.0
                if 'logprobs' in response['choices'][0] and response['choices'][0]['logprobs']:
                    logprobs_data = response['choices'][0]['logprobs']
                    if logprobs_data and 'token_logprobs' in logprobs_data:
                        token_logprobs = [lp for lp in logprobs_data['token_logprobs'] if lp is not None]
                        if token_logprobs:
                            logprob_score = sum(token_logprobs) / len(token_logprobs)

                avg_score = (avg_score / len(segments)) + (logprob_score * 10.0)
                batch_candidates.append((translations, avg_score))

            # Select best batch
            batch_candidates.sort(key=lambda x: x[1], reverse=True)
            return batch_candidates[0][0]

        else:
            # Single generation (original behavior)
            response = model(
                prompt,
                max_tokens=MAX_TOKENS * len(segments),
                temperature=TEMPERATURE,
                stop=["###", "\n\n\n\n", "User:", "System:"],
                echo=False
            )

            translation_text = response['choices'][0]['text'].strip()
            return _parse_batch_translations(translation_text, segments)

    except Exception as e:
        # Fallback to individual translation
        return [f"[Batch Error: {str(e)}]" for _ in segments]
