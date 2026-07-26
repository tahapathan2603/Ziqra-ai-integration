"""Speaking pace measurement and classification."""

from typing import Dict, List


def calculate_wpm(word_count: int, duration: float) -> float:
    """Words per minute: wpm = total_words / total_minutes."""
    if duration <= 0:
        return 0.0
    return word_count / (duration / 60.0)


def calculate_sentences_per_minute(sentence_count: int, duration: float) -> float:
    """Sentences per minute, given total speech duration in seconds."""
    if duration <= 0:
        return 0.0
    return sentence_count / (duration / 60.0)


def classify_speaking_speed(wpm: float) -> str:
    """<90 -> very_slow, 90-120 -> slow, 120-160 -> ideal, 160-180 -> fast, 180+ -> too_fast."""
    if wpm < 90:
        return "very_slow"
    if wpm < 120:
        return "slow"
    if wpm < 160:
        return "ideal"
    if wpm < 180:
        return "fast"
    return "too_fast"


def analyze_speaking_speed(words: List[Dict], sentences: List[Dict], duration: float) -> Dict:
    """
    Run the full speaking speed analysis pipeline.

    Returns:
        {
            "words_per_minute": int,
            "sentences_per_minute": int,
            "classification": str
        }
    """
    wpm = calculate_wpm(len(words), duration)
    spm = calculate_sentences_per_minute(len(sentences), duration)
    return {
        "words_per_minute": round(wpm),
        "sentences_per_minute": round(spm),
        "classification": classify_speaking_speed(wpm),
    }
