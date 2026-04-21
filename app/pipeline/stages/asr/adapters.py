import importlib.util
import logging
from pathlib import Path

from app.domain.contracts import ASRAdapter

logger = logging.getLogger(__name__)


class NeMoASRAdapter(ASRAdapter):
    def __init__(self, model_name: str, device: str) -> None:
        from app.pipeline.stages.asr.nemo import NeMoTranscriber

        self._provider_name = "nemo"
        self._model_name = str(model_name)
        self._transcriber = NeMoTranscriber(model_name=model_name, device=device)

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def transcribe_batch(
        self,
        audio_paths: list[Path],
        batch_size: int,
        pretokenize: bool,
    ) -> list[str]:
        return self._transcriber.transcribe_batch(
            audio_paths=audio_paths,
            batch_size=batch_size,
            pretokenize=pretokenize,
        )


class FallbackASRAdapter(ASRAdapter):
    def __init__(
        self,
        primary: ASRAdapter,
        fallback: ASRAdapter | None,
        requested_provider_name: str,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._active: ASRAdapter = primary
        self.requested_provider_name = str(requested_provider_name)
        self.fallback_used = False
        self.fallback_reason = ""

    @property
    def provider_name(self) -> str:
        return self._active.provider_name

    @property
    def model_name(self) -> str:
        return self._active.model_name

    def transcribe_batch(
        self,
        audio_paths: list[Path],
        batch_size: int,
        pretokenize: bool,
    ) -> list[str]:
        if self._active is not self._primary:
            return self._active.transcribe_batch(audio_paths, batch_size, pretokenize)

        try:
            return self._primary.transcribe_batch(audio_paths, batch_size, pretokenize)
        except Exception as exc:  # noqa: BLE001
            if self._fallback is None:
                raise

            self.fallback_used = True
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            self._active = self._fallback
            logger.warning(
                "ASR provider '%s' failed, switching to fallback '%s': %s",
                self._primary.provider_name,
                self._fallback.provider_name,
                self.fallback_reason,
            )
            return self._active.transcribe_batch(audio_paths, batch_size, pretokenize)


def build_single_asr_adapter(
    provider_name: str,
    model_name: str,
    device: str,
    google_api_key: str = "",
    **_provider_options: str,
) -> ASRAdapter:
    """Instantiate a single ASR provider adapter."""
    if provider_name == "nemo":
        if importlib.util.find_spec("nemo") is None:
            raise RuntimeError(
                "NeMo ASR provider is not installed. "
                "Install audio-process[asr-nemo] or choose another ASR provider."
            )
        return NeMoASRAdapter(model_name=model_name, device=device)
    raise ValueError(f"Unknown ASR provider: {provider_name}")


def make_asr_adapter(
    provider_name: str,
    model_name: str,
    device: str,
    fallback_provider_name: str = "none",
    fallback_model_name: str = "",
    google_api_key: str = "",
) -> ASRAdapter:
    """Create a primary ASR adapter with optional fallback."""
    primary = build_single_asr_adapter(
        provider_name=provider_name,
        model_name=model_name,
        device=device,
        google_api_key=google_api_key,
    )

    fallback: ASRAdapter | None = None
    if fallback_provider_name != "none":
        fallback = build_single_asr_adapter(
            provider_name=fallback_provider_name,
            model_name=fallback_model_name,
            device=device,
            google_api_key=google_api_key,
        )

    return FallbackASRAdapter(
        primary=primary,
        fallback=fallback,
        requested_provider_name=provider_name,
    )
