"""Turkish transcript normalization (numbers to words, script checks) and Turkish-aware WER text.

Turkish writes numbers as separate words ("iki bin on"), lower-cases dotted İ to i and dotless I to ı,
and reads percentages as "yüzde N". Podcast transcripts here contain digits in about 10% of rows,
so instead of rejecting them the digits are spelled out; a row is rejected only if something
unreadable remains (digits, non-Latin script).
"""

import re
import unicodedata

ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
SCALES = ["", "bin", "milyon", "milyar", "trilyon", "katrilyon"]
ORDINAL = {
    "bir": "birinci", "iki": "ikinci", "üç": "üçüncü", "dört": "dördüncü", "beş": "beşinci",
    "altı": "altıncı", "yedi": "yedinci", "sekiz": "sekizinci", "dokuz": "dokuzuncu",
    "on": "onuncu", "yirmi": "yirminci", "otuz": "otuzuncu", "kırk": "kırkıncı", "elli": "ellinci",
    "altmış": "altmışıncı", "yetmiş": "yetmişinci", "seksen": "sekseninci", "doksan": "doksanıncı",
    "yüz": "yüzüncü", "bin": "bininci", "milyon": "milyonuncu", "milyar": "milyarıncı",
    "trilyon": "trilyonuncu", "katrilyon": "katrilyonuncu", "sıfır": "sıfırıncı",
}
LATIN_EXTRA = set("çğıöşüâîûÇĞİÖŞÜÂÎÛ")
PUNCTUATION = set(".,;:!?'\"-()")


def tr_lower(text):
    """Turkish case folding: İ -> i and I -> ı before the generic lower-casing."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def _under_thousand(n):
    words = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        words.append("yüz" if hundreds == 1 else f"{ONES[hundreds]} yüz")
    tens, ones = divmod(rest, 10)
    if tens:
        words.append(TENS[tens])
    if ones:
        words.append(ONES[ones])
    return " ".join(words)


def number_words(n):
    """Integer -> Turkish words with spaces ("2010" -> "iki bin on")."""
    if n < 0:
        return "eksi " + number_words(-n)
    if n == 0:
        return "sıfır"
    parts = []
    scale = 0
    while n:
        n, group = divmod(n, 1000)
        if group:
            if scale == 1 and group == 1:
                parts.append("bin")  # "bin", never "bir bin"
            else:
                parts.append(f"{_under_thousand(group)} {SCALES[scale]}".strip())
        scale += 1
    return " ".join(reversed(parts))


def ordinal_words(n):
    words = number_words(n).split()
    words[-1] = ORDINAL[words[-1]]
    return " ".join(words)


def _digits_words(s):
    """Digit strings read digit by digit (leading zeros, long codes)."""
    return " ".join(number_words(int(c)) for c in s)


def _int_words(s):
    if len(s) > 1 and s[0] == "0" or len(s) > 15:
        return _digits_words(s)
    return number_words(int(s))


LETTER = "a-zA-ZçğıöşüâîûÇĞİÖŞÜÂÎÛ"


def normalize_numbers(text):
    """Spell out the numeric expressions Turkish podcast transcripts contain."""
    # 50% / %50 / % 50 -> yüzde 50
    text = re.sub(r"%\s?(\d+(?:[.,]\d+)?)", r"yüzde \1", text)
    text = re.sub(r"(\d+(?:[.,]\d+)?)\s?%", r"yüzde \1", text)
    # 12:30 -> 12 30
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", r"\1 \2", text)
    # 1.000.000 -> 1000000 (dot as thousands separator when exactly three digits follow)
    text = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", text)
    # decimals: 3,5 -> 3 virgül 5 ; 17.5 -> 17 nokta 5
    text = re.sub(r"(\d+),(\d+)", r"\1 virgül \2", text)
    text = re.sub(r"(\d+)\.(\d+)", r"\1 nokta \2", text)
    # ranges 3-4 -> 3 4
    text = re.sub(r"(\d+)\s?[-–]\s?(\d+)", r"\1 \2", text)
    # ordinals "12. nesil" -> "on ikinci nesil" (only when a lower-case word follows)
    text = re.sub(r"\b(\d+)\.(?=\s+[a-zçğıöşü])", lambda m: ordinal_words(int(m.group(1))), text)
    # split letters glued to digits: 350D -> 350 D ; USB3 -> USB 3
    text = re.sub(rf"(?<=\d)(?=[{LETTER}])", " ", text)
    text = re.sub(rf"(?<=[{LETTER}])(?=\d)", " ", text)
    # apostrophe suffixes 2010'da -> iki bin onda ; 3'üncü -> üçüncü
    def suffixed(m):
        number, suffix = m.group(1), m.group(2)
        if re.fullmatch(r"[iıuü]?nc[iıuü]", suffix):
            return ordinal_words(int(number))
        return _int_words(number) + suffix

    text = re.sub(rf"(\d+)['’]([{LETTER}]+)", suffixed, text)
    # remaining integers
    text = re.sub(r"\d+", lambda m: _int_words(m.group()), text)
    return re.sub(r"\s+", " ", text).strip()


def check_script(text):
    """Reject transcripts that contain characters a Turkish byte model should never see."""
    for c in text:
        if c.isspace() or c in PUNCTUATION or c in LATIN_EXTRA or ("a" <= c <= "z") or ("A" <= c <= "Z"):
            continue
        if c.isdigit():
            raise ValueError(f"Digit survived normalization: {c!r}")
        raise ValueError(f"Unsupported character for Turkish transcript: {c!r} (U+{ord(c):04X})")


def normalize_turkish(text):
    """Turkish transcript -> model text: NFKC, numbers as words, script check, whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[“”„]", '"', text)
    text = text.replace("‘", "'").replace("’", "'").replace("–", "-").replace("—", "-").replace("…", "...")
    text = normalize_numbers(text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # "kelime ." -> "kelime."
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("Empty transcript")
    check_script(text)
    return text


def metric_text_turkish(text):
    """WER/CER normalization for Turkish: numbers spelled out, Turkish lower-case, letters only."""
    text = unicodedata.normalize("NFKC", text)
    text = normalize_numbers(text)
    text = tr_lower(text)
    text = "".join(c if c.isalnum() or c.isspace() else ("" if c in "'’" else " ") for c in text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(text.split())
