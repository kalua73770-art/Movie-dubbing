"""
Hindi translation and duration matching.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import re
from loguru import logger

from hindi_dubbing.data.models import DubSegment, ProjectData, SegmentStatus
from hindi_dubbing.config import get_config


class TranslationEngine:
    """Hindi translation with duration matching."""
    
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        source_lang: str = "eng_Latn",
        target_lang: str = "hin_Deva",
        device: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.device = device
        self.kwargs = kwargs
        self._translator = None
        self._tokenizer = None
    
    @property
    def translator(self):
        """Lazy load the translation model."""
        if self._translator is None:
            logger.info(f"Loading translation model: {self.model_name}")
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._translator = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            
            if self.device:
                self._translator.to(self.device)
            elif torch.cuda.is_available():
                self._translator.to("cuda")
            
            logger.success("Translation model loaded")
        return self._translator
    
    @property
    def tokenizer(self):
        if self._translator is None:
            _ = self.translator
        return self._tokenizer
    
    def translate(self, text: str) -> str:
        """Translate English text to Hindi."""
        if not text.strip():
            return ""
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        if hasattr(self.translator, "device"):
            inputs = {k: v.to(self.translator.device) for k, v in inputs.items()}
        
        # Set target language
        forced_bos_token_id = self.tokenizer.lang_code_to_id.get(self.target_lang)
        
        import torch
        with torch.no_grad():
            outputs = self.translator.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=5,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                **self.kwargs,
            )
        
        translated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return translated.strip()
    
    def translate_batch(self, texts: list[str]) -> list[str]:
        """Translate multiple texts at once."""
        if not texts:
            return []
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        if hasattr(self.translator, "device"):
            inputs = {k: v.to(self.translator.device) for k, v in inputs.items()}
        
        forced_bos_token_id = self.tokenizer.lang_code_to_id.get(self.target_lang)
        
        import torch
        with torch.no_grad():
            outputs = self.translator.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=5,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                **self.kwargs,
            )
        
        translated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [t.strip() for t in translated]


class DurationMatcher:
    """Match Hindi translation duration to original dialogue duration."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_config()
        self.translation_config = self.config.translation
        self.max_ratio = self.translation_config.max_duration_ratio
        self.min_ratio = self.translation_config.min_duration_ratio
        self.max_attempts = self.translation_config.rewrite_attempts
    
    def estimate_speech_duration(self, text: str, language: str = "hi") -> float:
        """
        Estimate speech duration for text.
        Rough approximation based on characters/words per second.
        """
        if language == "hi":
            # Hindi: ~15-20 characters per second, ~5-6 words per second
            char_rate = 18
            word_rate = 5.5
        else:
            # English: ~13-15 characters per second, ~4-5 words per second
            char_rate = 14
            word_rate = 4.5
        
        char_count = len(text)
        word_count = len(text.split())
        
        # Use whichever gives longer estimate (more conservative)
        char_duration = char_count / char_rate
        word_duration = word_count / word_rate
        
        return max(char_duration, word_duration)
    
    def match_duration(
        self,
        segment: DubSegment,
        original_duration: float,
    ) -> str:
        """
        Adapt Hindi text to match target duration.
        
        Args:
            segment: DubSegment with hindi_text
            original_duration: Target duration in seconds
        
        Returns:
            Duration-adapted Hindi text
        """
        hindi_text = segment.hindi_text
        if not hindi_text:
            return ""
        
        estimated = self.estimate_speech_duration(hindi_text, "hi")
        ratio = estimated / original_duration if original_duration > 0 else 1.0
        
        segment.duration_ratio = ratio
        
        # If within acceptable range, return as-is
        if self.min_ratio <= ratio <= self.max_ratio:
            return hindi_text
        
        # Try to adapt
        adapted = self._adapt_text(hindi_text, original_duration, ratio)
        return adapted
    
    def _adapt_text(self, text: str, target_duration: float, current_ratio: float) -> str:
        """Adapt text to fit target duration."""
        if current_ratio > self.max_ratio:
            # Too long - shorten
            return self._shorten_text(text, target_duration)
        elif current_ratio < self.min_ratio:
            # Too short - expand naturally
            return self._expand_text(text, target_duration)
        return text
    
    def _shorten_text(self, text: str, target_duration: float) -> str:
        """Shorten Hindi text while preserving meaning."""
        # Strategy 1: Remove filler words
        filler_words = [
            "तो", "भी", "ही", "काफी", "बहुत", "जरा", "थोड़ा", "कुछ",
            "वैसे", "असल में", "दरअसल", "मतलब", "यानी", "जैसे",
        ]
        
        words = text.split()
        shortened = [w for w in words if w not in filler_words]
        
        if len(shortened) < len(words):
            result = " ".join(shortened)
            if self.estimate_speech_duration(result, "hi") <= target_duration * self.max_ratio:
                return result
        
        # Strategy 2: Use shorter synonyms
        synonyms = {
            "क्या कर रहे हो": "क्या कर रहे",
            "कहाँ जा रहे हो": "कहाँ जा रहे",
            "कैसे हो": "कैसे हो",
            "मैं समझ गया": "समझ गया",
            "कोई बात नहीं": "कोई बात नहीं",
            "चलो ठीक है": "चलो",
            "अच्छा ठीक है": "अच्छा",
        }
        
        result = text
        for long_form, short_form in synonyms.items():
            if long_form in result:
                result = result.replace(long_form, short_form)
                if self.estimate_speech_duration(result, "hi") <= target_duration * self.max_ratio:
                    return result
        
        # Strategy 3: Truncate at natural boundary
        sentences = re.split(r'[.!?।]', text)
        for i in range(len(sentences), 0, -1):
            candidate = ". ".join(sentences[:i]) + "।"
            if self.estimate_speech_duration(candidate, "hi") <= target_duration * self.max_ratio:
                return candidate
        
        # Strategy 4: Just truncate words
        words = text.split()
        for i in range(len(words), 0, -1):
            candidate = " ".join(words[:i])
            if self.estimate_speech_duration(candidate, "hi") <= target_duration * self.max_ratio:
                return candidate + "।"
        
        return text  # Fallback
    
    def _expand_text(self, text: str, target_duration: float) -> str:
        """Expand Hindi text naturally."""
        # Add natural fillers
        expansions = [
            "तो ",
            "असल में ",
            "वैसे ",
            "देखो ",
            "सुनो ",
            "हाँ ",
            "जी ",
        ]
        
        # Try adding fillers at natural positions
        for filler in expansions:
            candidate = filler + text
            if self.estimate_speech_duration(candidate, "hi") >= target_duration * self.min_ratio:
                return candidate
        
        # Add at end
        for filler in ["। हाँ", "। ठीक है", "। समझ गए", "। अच्छा"]:
            candidate = text + filler
            if self.estimate_speech_duration(candidate, "hi") >= target_duration * self.min_ratio:
                return candidate
        
        return text


def process_translation(project: ProjectData, config: Optional[dict] = None) -> ProjectData:
    """
    Process translation for all segments in project.
    
    Args:
        project: ProjectData with transcribed segments
        config: Optional configuration
    
    Returns:
        Updated ProjectData with translations
    """
    cfg = config or get_config()
    
    engine = TranslationEngine(
        model_name=cfg.translation.model,
        source_lang=cfg.translation.source_lang,
        target_lang=cfg.translation.target_lang,
    )
    
    matcher = DurationMatcher(cfg)
    
    logger.info(f"Translating {len(project.segments)} segments to Hindi")
    
    # Collect texts to translate
    texts_to_translate = []
    segment_indices = []
    
    for i, segment in enumerate(project.segments):
        if segment.original_text and segment.status == SegmentStatus.TRANSCRIBED:
            texts_to_translate.append(segment.original_text)
            segment_indices.append(i)
    
    if not texts_to_translate:
        logger.warning("No segments to translate")
        return project
    
    # Translate in batches
    batch_size = 8
    translations = []
    
    for i in range(0, len(texts_to_translate), batch_size):
        batch = texts_to_translate[i:i+batch_size]
        batch_translations = engine.translate_batch(batch)
        translations.extend(batch_translations)
    
    # Apply translations and duration matching
    for idx, translation in zip(segment_indices, translations):
        segment = project.segments[idx]
        segment.hindi_text = translation
        segment.hindi_text_adapted = matcher.match_duration(segment, segment.time_range.duration)
        segment.status = SegmentStatus.TRANSLATED
    
    logger.success("Translation complete")
    return project