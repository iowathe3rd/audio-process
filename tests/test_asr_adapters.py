from __future__ import annotations
import unittest
from pathlib import Path

from app.stages.asr_adapters import FallbackASRAdapter, build_single_asr_adapter, ASRAdapter


class _OkAdapter(ASRAdapter):
    def __init__(self, provider_name: str, model_name: str, text: str) -> None:
        self._provider_name = provider_name
        self._model_name = model_name
        self._text = text

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def transcribe_batch(self, audio_paths: list[Path], batch_size: int, pretokenize: bool) -> list[str]:
        return [self._text for _ in audio_paths]


class _FailAdapter(ASRAdapter):
    def __init__(self, provider_name: str, model_name: str) -> None:
        self._provider_name = provider_name
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def transcribe_batch(self, audio_paths: list[Path], batch_size: int, pretokenize: bool) -> list[str]:
        raise RuntimeError("primary failed")


class AsrAdapterTests(unittest.TestCase):
    def test_fallback_switches_provider_on_failure(self) -> None:
        primary = _FailAdapter(provider_name="chirp", model_name="chirp_2")
        fallback = _OkAdapter(provider_name="nemo", model_name="nvidia/model", text="ok")
        adapter = FallbackASRAdapter(primary=primary, fallback=fallback, requested_provider_name="chirp")

        result = adapter.transcribe_batch([Path("a.wav"), Path("b.wav")], batch_size=2, pretokenize=False)

        self.assertEqual(result, ["ok", "ok"])
        self.assertTrue(adapter.fallback_used)
        self.assertEqual(adapter.provider_name, "nemo")
        self.assertIn("primary failed", adapter.fallback_reason)

    def test_no_fallback_propagates_error(self) -> None:
        primary = _FailAdapter(provider_name="chirp", model_name="chirp_2")
        adapter = FallbackASRAdapter(primary=primary, fallback=None, requested_provider_name="chirp")

        with self.assertRaises(RuntimeError):
            adapter.transcribe_batch([Path("a.wav")], batch_size=1, pretokenize=False)

    def test_build_single_adapter_rejects_unknown_provider(self) -> None:
        with self.assertRaises(ValueError):
            build_single_asr_adapter(
                provider_name="unknown",
                model_name="x",
                device="cpu",
                google_api_key="",
                chirp_language_code="ru-RU",
                chirp_project="",
                chirp_location="global",
                chirp_recognizer="",
            )


if __name__ == "__main__":
    unittest.main()
