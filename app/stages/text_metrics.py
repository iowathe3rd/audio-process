import re

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ0-9]+")
KK_SPECIFIC_RE = re.compile(r"[ӘәІіҢңҒғҮүҰұҚқӨөҺһ]")
RU_CYR_RE = re.compile(r"[А-Яа-яЁё]")
LAT_RE = re.compile(r"[A-Za-z]")
SPACE_RE = re.compile(r"\s+")


def normalize_words(text: str) -> list[str]:
    return [item.lower() for item in WORD_RE.findall(text)]


def has_word_sequence_drift(source: str, candidate: str) -> bool:
    return normalize_words(source) != normalize_words(candidate)


def token_language(token: str) -> str:
    has_kk = bool(KK_SPECIFIC_RE.search(token))
    has_ru_cyr = bool(RU_CYR_RE.search(token))
    has_lat = bool(LAT_RE.search(token))

    if has_kk:
        return "kk"
    if has_ru_cyr and has_lat:
        return "mixed"
    if has_ru_cyr:
        return "ru"
    if has_lat:
        return "lat"
    return "other"


def language_switching_ratio(texts: list[str]) -> float:
    tokens: list[str] = []
    for text in texts:
        tokens.extend(WORD_RE.findall(text))

    labels = [token_language(token) for token in tokens]
    labels = [label for label in labels if label in {"ru", "kk", "lat", "mixed"}]
    if len(labels) < 2:
        return 0.0

    switches = sum(1 for index in range(1, len(labels)) if labels[index] != labels[index - 1])
    return switches / max(1, len(labels) - 1)


def punctuation_density(texts: list[str]) -> float:
    joined = " ".join(item for item in texts if item)
    if not joined:
        return 0.0
    punct_count = sum(1 for char in joined if char in ".,!?;:-()[]{}\"'")
    return punct_count / max(1, len(joined))


def deterministic_light_enhance(text: str) -> str:
    cleaned = SPACE_RE.sub(" ", text.strip())
    if not cleaned:
        return ""

    first = cleaned[0]
    if first.isalpha():
        cleaned = first.upper() + cleaned[1:]

    return cleaned
