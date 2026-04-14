import logging
import re
from time import perf_counter
from typing import Any

from google import genai

from audio_pipeline.models import PipelineSegment
from audio_pipeline.stages.text_metrics import (
    deterministic_light_enhance,
    has_word_sequence_drift,
)

logger = logging.getLogger(__name__)

SEGMENT_RE = re.compile(r"<SEG\s+id=['\"]?(\d+)['\"]?>(.*?)</SEG>", re.IGNORECASE | re.DOTALL)


def semantic_drift_detected(source: str, candidate: str) -> bool:
    return has_word_sequence_drift(source, candidate)


def _build_segment_payload(texts: list[str]) -> str:
    rows = []
    for index, text in enumerate(texts):
        cleaned = text.replace("\n", " ").strip()
        rows.append(f'<SEG id="{index}">{cleaned}</SEG>')
    return "\n".join(rows)


def _parse_segment_payload(payload: str, expected_count: int) -> list[str]:
    parsed: dict[int, str] = {}
    for match in SEGMENT_RE.finditer(payload):
        index = int(match.group(1))
        text = match.group(2).strip()
        parsed[index] = text

    if len(parsed) < expected_count:
        return []

    output: list[str] = []
    for index in range(expected_count):
        if index not in parsed:
            return []
        output.append(parsed[index])
    return output


def _placeholder_counts(text: str) -> dict[str, int]:
    counts = {
        "NAME": text.count("[ИМЯ]"),
        "PHONE": text.count("[ТЕЛЕФОН]"),
        "ID": text.count("[ID]"),
        "ADDRESS": text.count("[АДРЕС]"),
        "EMAIL": text.count("[EMAIL]"),
    }
    return {key: value for key, value in counts.items() if value > 0}


def _contains_pii_candidate(text: str) -> bool:
    if re.search(r"\b\+?\d[\d\s\-()]{6,}\b", text):
        return True
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        return True
    if re.search(r"\b\d{8,}\b", text):
        return True
    return False


class VertexTextProcessor:
    def __init__(
        self,
        api_key: str,
        project: str,
        location: str,
        model_name: str,
        enabled: bool,
        strict: bool = True,
    ) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self.strict = strict
        self.client = None

        if not enabled:
            logger.warning("Vertex processing is disabled by configuration")
            return

        # Prefer direct Google GenAI API key when provided.
        if api_key:
            try:
                self.client = genai.Client(api_key=api_key)
                logger.info("Google GenAI client initialized via API key")
                return
            except Exception as exc:  # noqa: BLE001
                if self.strict:
                    raise RuntimeError("Failed to initialize Google GenAI client with API key") from exc
                logger.error("Failed to initialize Google GenAI client with API key: %s", exc, exc_info=True)
                self.enabled = False
                return

        if not project or not location:
            message = (
                "Text processing is enabled but no auth settings provided. "
                "Set GOOGLE_API_KEY (or GEMINI_API_KEY), or GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION"
            )
            if self.strict:
                raise ValueError(message)
            logger.warning("%s. Falling back to pass-through text", message)
            self.enabled = False
            return

        try:
            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
            logger.info("Vertex AI client initialized for project=%s location=%s", project, location)
        except Exception as exc:  # noqa: BLE001
            if self.strict:
                raise RuntimeError("Failed to initialize Vertex AI client") from exc
            logger.error("Failed to initialize Vertex AI client: %s", exc, exc_info=True)
            self.enabled = False

    def _generate(self, prompt: str) -> str:
        if not self.enabled or self.client is None:
            return ""

        request: dict[str, Any] = {
            "model": self.model_name,
            "contents": prompt,
            "config": {
                "temperature": 0.0,
                "top_p": 1.0,
            },
        }

        try:
            response = self.client.models.generate_content(**request)
        except TypeError:
            request.pop("config", None)
            response = self.client.models.generate_content(**request)

        if getattr(response, "text", None):
            return str(response.text).strip()

        chunks: list[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if text:
                    chunks.append(str(text))

        return " ".join(chunks).strip()

    def anonymize_batch(self, texts: list[str]) -> tuple[list[str], dict[str, Any]]:
        if not texts:
            return [], {
                "llm_calls_total": 0,
                "llm_latency_total_ms": 0.0,
                "chars_in_call": 0,
                "segments_in_call": 0,
                "windows_failed": 0,
                "fallback_segments": 0,
                "entity_mask_counts": {},
                "pii_candidate_segments": 0,
                "masked_candidate_segments": 0,
            }

        if not self.enabled or self.client is None:
            return list(texts), {
                "llm_calls_total": 0,
                "llm_latency_total_ms": 0.0,
                "chars_in_call": sum(len(item) for item in texts),
                "segments_in_call": len(texts),
                "windows_failed": 0,
                "fallback_segments": 0,
                "entity_mask_counts": {},
                "pii_candidate_segments": 0,
                "masked_candidate_segments": 0,
            }

        payload = _build_segment_payload(texts)
        prompt = (
            "Ты анонимизируешь расшифровку звонка.\n"
            "Верни ТОЛЬКО сегменты в формате <SEG id=\"N\">...</SEG>.\n"
            "Для PII используй только плейсхолдеры: [ИМЯ], [ТЕЛЕФОН], [ID], [АДРЕС], [EMAIL].\n"
            "Нельзя менять порядок сегментов, смысл или не-PII текст.\n"
            "Нельзя добавлять комментарии.\n\n"
            f"Сегменты:\n{payload}"
        )

        call_start = perf_counter()
        windows_failed = 0
        fallback_segments = 0
        entity_counts: dict[str, int] = {}
        try:
            raw_response = self._generate(prompt)
            outputs = _parse_segment_payload(raw_response, expected_count=len(texts))
        except Exception as exc:  # noqa: BLE001
            if self.strict:
                raise
            logger.error("Vertex anonymization batch failed: %s", exc, exc_info=True)
            outputs = []

        if not outputs:
            windows_failed = 1
            outputs = list(texts)
            fallback_segments = len(texts)
        else:
            for index, value in enumerate(outputs):
                if not value.strip():
                    outputs[index] = texts[index]
                    fallback_segments += 1

        for item in outputs:
            for key, value in _placeholder_counts(item).items():
                entity_counts[key] = entity_counts.get(key, 0) + value

        pii_candidate_segments = sum(1 for item in texts if _contains_pii_candidate(item))
        masked_candidate_segments = sum(
            1
            for source, output in zip(texts, outputs)
            if _contains_pii_candidate(source) and bool(_placeholder_counts(output))
        )

        return outputs, {
            "llm_calls_total": 1,
            "llm_latency_total_ms": (perf_counter() - call_start) * 1000.0,
            "chars_in_call": sum(len(item) for item in texts),
            "segments_in_call": len(texts),
            "windows_failed": windows_failed,
            "fallback_segments": fallback_segments,
            "entity_mask_counts": entity_counts,
            "pii_candidate_segments": pii_candidate_segments,
            "masked_candidate_segments": masked_candidate_segments,
        }

    def enhance_batch(self, texts: list[str]) -> tuple[list[str], dict[str, Any]]:
        if not texts:
            return [], {
                "llm_calls_total": 0,
                "llm_latency_total_ms": 0.0,
                "chars_in_call": 0,
                "segments_in_call": 0,
                "windows_failed": 0,
                "fallback_segments": 0,
            }

        if not self.enabled or self.client is None:
            return list(texts), {
                "llm_calls_total": 0,
                "llm_latency_total_ms": 0.0,
                "chars_in_call": sum(len(item) for item in texts),
                "segments_in_call": len(texts),
                "windows_failed": 0,
                "fallback_segments": 0,
            }

        payload = _build_segment_payload(texts)
        prompt = (
            "Ты приводишь расшифровку к читаемому виду.\n"
            "Разрешено ТОЛЬКО: пунктуация, регистр, пробелы.\n"
            "Запрещено: менять/добавлять/удалять слова, менять смысл, переставлять слова.\n"
            "Верни ТОЛЬКО сегменты в формате <SEG id=\"N\">...</SEG>.\n"
            "Без комментариев.\n\n"
            f"Сегменты:\n{payload}"
        )

        call_start = perf_counter()
        windows_failed = 0
        fallback_segments = 0
        try:
            raw_response = self._generate(prompt)
            outputs = _parse_segment_payload(raw_response, expected_count=len(texts))
        except Exception as exc:  # noqa: BLE001
            if self.strict:
                raise
            logger.error("Vertex enhancement batch failed: %s", exc, exc_info=True)
            outputs = []

        if not outputs:
            windows_failed = 1
            outputs = list(texts)
            fallback_segments = len(texts)
        else:
            for index, value in enumerate(outputs):
                if not value.strip():
                    outputs[index] = texts[index]
                    fallback_segments += 1

        return outputs, {
            "llm_calls_total": 1,
            "llm_latency_total_ms": (perf_counter() - call_start) * 1000.0,
            "chars_in_call": sum(len(item) for item in texts),
            "segments_in_call": len(texts),
            "windows_failed": windows_failed,
            "fallback_segments": fallback_segments,
        }


def anonymize_segments(
    segments: list[PipelineSegment],
    semantic_windows: list[dict[str, Any]],
    processor: Any,
) -> tuple[list[PipelineSegment], dict[str, Any]]:
    anonymized: list[PipelineSegment] = []
    for segment in segments:
        anonymized.append(
            PipelineSegment(
                speaker=segment.speaker,
                start=segment.start,
                end=segment.end,
                raw_text=segment.raw_text,
                anonymized_text=segment.raw_text,
                enhanced_text="",
                chunk_path=segment.chunk_path,
                asr_status=segment.asr_status,
            )
        )

    total_calls = 0
    total_latency_ms = 0.0
    total_chars = 0
    total_segments_in_calls = 0
    total_windows_failed = 0
    total_fallback_segments = 0
    pii_candidate_segments = 0
    masked_candidate_segments = 0
    entity_mask_counts: dict[str, int] = {}

    for window in semantic_windows:
        positions = [
            int(position)
            for position in window.get("segment_positions", [])
            if 0 <= int(position) < len(anonymized)
        ]
        if not positions:
            continue

        texts = [anonymized[position].raw_text for position in positions]
        outputs, report = processor.anonymize_batch(texts)

        total_calls += int(report.get("llm_calls_total", 0))
        total_latency_ms += float(report.get("llm_latency_total_ms", 0.0))
        total_chars += int(report.get("chars_in_call", 0))
        total_segments_in_calls += int(report.get("segments_in_call", 0))
        total_windows_failed += int(report.get("windows_failed", 0))
        total_fallback_segments += int(report.get("fallback_segments", 0))
        pii_candidate_segments += int(report.get("pii_candidate_segments", 0))
        masked_candidate_segments += int(report.get("masked_candidate_segments", 0))

        for key, value in dict(report.get("entity_mask_counts", {})).items():
            entity_mask_counts[str(key)] = entity_mask_counts.get(str(key), 0) + int(value)

        for local_index, position in enumerate(positions):
            anonymized[position].anonymized_text = outputs[local_index] if local_index < len(outputs) else texts[local_index]

    report: dict[str, Any] = {
        "llm_calls_total": total_calls,
        "llm_latency_total_ms": total_latency_ms,
        "avg_chars_per_llm_call": (total_chars / total_calls) if total_calls else 0.0,
        "avg_segments_per_llm_call": (total_segments_in_calls / total_calls) if total_calls else 0.0,
        "windows_total": len(semantic_windows),
        "windows_failed": total_windows_failed,
        "fallback_segments": total_fallback_segments,
        "entity_mask_counts": entity_mask_counts,
        "pii_candidate_segments": pii_candidate_segments,
        "masked_candidate_segments": masked_candidate_segments,
        "anonymization_recall_proxy": (
            masked_candidate_segments / max(1, pii_candidate_segments)
        ),
    }

    return anonymized, report


def enhance_segments(
    segments: list[PipelineSegment],
    semantic_windows: list[dict[str, Any]],
    processor: Any,
    low_confidence_positions: set[int] | None = None,
    skip_low_confidence: bool = True,
) -> tuple[list[PipelineSegment], dict[str, Any]]:
    safe_low_confidence_positions = set(low_confidence_positions or set())

    enhanced: list[PipelineSegment] = []
    for segment in segments:
        source_text = segment.anonymized_text or segment.raw_text
        enhanced.append(
            PipelineSegment(
                speaker=segment.speaker,
                start=segment.start,
                end=segment.end,
                raw_text=segment.raw_text,
                anonymized_text=segment.anonymized_text,
                enhanced_text=source_text,
                chunk_path=segment.chunk_path,
                asr_status=segment.asr_status,
            )
        )

    total_calls = 0
    total_latency_ms = 0.0
    total_chars = 0
    total_segments_in_calls = 0
    total_windows_failed = 0
    total_fallback_segments = 0
    semantic_drift_flags_total = 0
    low_confidence_segments_skipped = 0
    deterministic_only_segments = 0

    for window in semantic_windows:
        positions = [
            int(position)
            for position in window.get("segment_positions", [])
            if 0 <= int(position) < len(enhanced)
        ]
        if not positions:
            continue

        positions_for_llm: list[int] = []
        texts_for_llm: list[str] = []
        for position in positions:
            source_text = enhanced[position].anonymized_text or enhanced[position].raw_text
            if skip_low_confidence and position in safe_low_confidence_positions:
                enhanced[position].enhanced_text = deterministic_light_enhance(source_text)
                low_confidence_segments_skipped += 1
            else:
                positions_for_llm.append(position)
                texts_for_llm.append(source_text)

        if not positions_for_llm:
            deterministic_only_segments += len(positions)
            continue

        outputs, report = processor.enhance_batch(texts_for_llm)

        total_calls += int(report.get("llm_calls_total", 0))
        total_latency_ms += float(report.get("llm_latency_total_ms", 0.0))
        total_chars += int(report.get("chars_in_call", 0))
        total_segments_in_calls += int(report.get("segments_in_call", 0))
        total_windows_failed += int(report.get("windows_failed", 0))
        total_fallback_segments += int(report.get("fallback_segments", 0))

        for local_index, position in enumerate(positions_for_llm):
            source_text = texts_for_llm[local_index]
            candidate = outputs[local_index] if local_index < len(outputs) else source_text
            if semantic_drift_detected(source_text, candidate):
                semantic_drift_flags_total += 1
                candidate = source_text
            enhanced[position].enhanced_text = candidate

    report: dict[str, Any] = {
        "llm_calls_total": total_calls,
        "llm_latency_total_ms": total_latency_ms,
        "avg_chars_per_llm_call": (total_chars / total_calls) if total_calls else 0.0,
        "avg_segments_per_llm_call": (total_segments_in_calls / total_calls) if total_calls else 0.0,
        "windows_total": len(semantic_windows),
        "windows_failed": total_windows_failed,
        "fallback_segments": total_fallback_segments,
        "semantic_drift_flags_total": semantic_drift_flags_total,
        "low_confidence_segments_skipped": low_confidence_segments_skipped,
        "deterministic_only_segments": deterministic_only_segments,
    }

    return enhanced, report


def enhance_segments_deterministic(
    segments: list[PipelineSegment],
) -> tuple[list[PipelineSegment], dict[str, Any]]:
    enhanced: list[PipelineSegment] = []
    changed_segments = 0

    for segment in segments:
        source_text = segment.anonymized_text or segment.raw_text
        enhanced_text = deterministic_light_enhance(source_text)
        if enhanced_text != source_text:
            changed_segments += 1
        enhanced.append(
            PipelineSegment(
                speaker=segment.speaker,
                start=segment.start,
                end=segment.end,
                raw_text=segment.raw_text,
                anonymized_text=segment.anonymized_text,
                enhanced_text=enhanced_text,
                chunk_path=segment.chunk_path,
                asr_status=segment.asr_status,
            )
        )

    report: dict[str, Any] = {
        "llm_calls_total": 0,
        "llm_latency_total_ms": 0.0,
        "avg_chars_per_llm_call": 0.0,
        "avg_segments_per_llm_call": 0.0,
        "windows_total": 0,
        "windows_failed": 0,
        "fallback_segments": 0,
        "semantic_drift_flags_total": 0,
        "low_confidence_segments_skipped": 0,
        "deterministic_only_segments": len(segments),
        "deterministic_changed_segments": changed_segments,
    }
    return enhanced, report
