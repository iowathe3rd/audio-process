import importlib
import logging
from pathlib import Path

from app.domain.contracts import ASRAdapter

logger = logging.getLogger(__name__)


class NeMoASRAdapter(ASRAdapter):
    def __init__(self, model_name: str, device: str) -> None:
        from app.stages.asr_nemo import NeMoTranscriber

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


class GoogleChirpASRAdapter(ASRAdapter):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        language_code: str,
        project: str,
        location: str,
        recognizer: str,
    ) -> None:
        self._provider_name = "chirp"
        self._model_name = str(model_name)
        self._language_code = str(language_code or "").strip()
        self._recognizer = self._resolve_recognizer(
            recognizer=str(recognizer or "").strip(),
            project=str(project or "").strip(),
            location=str(location or "").strip(),
        )

        try:
            speech_module = importlib.import_module("google.cloud.speech_v2")
            self._speech_types = importlib.import_module("google.cloud.speech_v2.types.cloud_speech")
            client_options_module = importlib.import_module("google.api_core.client_options")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Google Chirp adapter requires google-cloud-speech. "
                "Install dependency: uv add google-cloud-speech"
            ) from exc

        SpeechClient = getattr(speech_module, "SpeechClient")
        ClientOptions = getattr(client_options_module, "ClientOptions")

        api_key_value = str(api_key or "").strip()
        if api_key_value:
            self._client = SpeechClient(client_options=ClientOptions(api_key=api_key_value))
        else:
            self._client = SpeechClient()

    @staticmethod
    def _resolve_recognizer(recognizer: str, project: str, location: str) -> str:
        if recognizer:
            return recognizer
        if project:
            safe_location = location or "global"
            return f"projects/{project}/locations/{safe_location}/recognizers/_"
        return ""

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
        del batch_size
        del pretokenize

        if not self._language_code:
            raise RuntimeError("Chirp language code is required. Set --chirp-language-code.")

        if not self._recognizer:
            raise RuntimeError(
                "Chirp recognizer is not configured. "
                "Set --chirp-recognizer or provide --chirp-project (and optionally --chirp-location)."
            )

        texts: list[str] = []
        for path in audio_paths:
            audio_bytes = Path(path).read_bytes()
            config = self._speech_types.RecognitionConfig(
                auto_decoding_config=self._speech_types.AutoDetectDecodingConfig(),
                language_codes=[self._language_code],
                model=self._model_name,
            )
            request = self._speech_types.RecognizeRequest(
                recognizer=self._recognizer,
                config=config,
                content=audio_bytes,
            )
            response = self._client.recognize(request=request)

            parts: list[str] = []
            for result in getattr(response, "results", []) or []:
                alternatives = getattr(result, "alternatives", []) or []
                if not alternatives:
                    continue
                text = str(getattr(alternatives[0], "transcript", "")).strip()
                if text:
                    parts.append(text)

            texts.append(" ".join(parts).strip())

        if len(texts) < len(audio_paths):
            texts.extend([""] * (len(audio_paths) - len(texts)))
        return texts[: len(audio_paths)]


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
    chirp_language_code: str = "",
    chirp_project: str = "",
    chirp_location: str = "global",
    chirp_recognizer: str = "",
) -> ASRAdapter:
    """Instantiate a single ASR provider adapter."""
    if provider_name == "nemo":
        return NeMoASRAdapter(model_name=model_name, device=device)
    if provider_name == "chirp":
        return GoogleChirpASRAdapter(
            model_name=model_name,
            api_key=google_api_key,
            language_code=chirp_language_code,
            project=chirp_project,
            location=chirp_location,
            recognizer=chirp_recognizer,
        )
    raise ValueError(f"Unknown ASR provider: {provider_name}")


def make_asr_adapter(
    provider_name: str,
    model_name: str,
    device: str,
    fallback_provider_name: str = "none",
    fallback_model_name: str = "",
    google_api_key: str = "",
    chirp_language_code: str = "",
    chirp_project: str = "",
    chirp_location: str = "global",
    chirp_recognizer: str = "",
) -> ASRAdapter:
    """Create a primary ASR adapter with optional fallback."""
    primary = build_single_asr_adapter(
        provider_name=provider_name,
        model_name=model_name,
        device=device,
        google_api_key=google_api_key,
        chirp_language_code=chirp_language_code,
        chirp_project=chirp_project,
        chirp_location=chirp_location,
        chirp_recognizer=chirp_recognizer,
    )

    fallback: ASRAdapter | None = None
    if fallback_provider_name != "none":
        fallback = build_single_asr_adapter(
            provider_name=fallback_provider_name,
            model_name=fallback_model_name,
            device=device,
            google_api_key=google_api_key,
            chirp_language_code=chirp_language_code,
            chirp_project=chirp_project,
            chirp_location=chirp_location,
            chirp_recognizer=chirp_recognizer,
        )

    return FallbackASRAdapter(
        primary=primary,
        fallback=fallback,
        requested_provider_name=provider_name,
    )
