"""Turkish text frontend for synthesis: free user text -> text that `turkish-v1` accepts, and sentence chunks.

`turkish.normalize_turkish` (version `turkish-v1`) is the training normalization: it spells out numbers and rejects
every character the byte model never saw. Text typed into a demo is messier: currency and unit symbols, dates and
clock times written with dots, abbreviations, acronyms, e-mail addresses, emoji and foreign letters. `prepare_text`
rewrites these into spoken Turkish words first and drops what cannot be spoken, reporting every change, so that
synthesis never fails on a stray symbol. Plain Turkish sentences pass through unchanged.

`split_sentences` cuts long text into sentence-based chunks short enough for the model (training utterances were at
most ~20 s including the voice prompt).
"""

import re
import unicodedata
from dataclasses import dataclass, field

from .turkish import LATIN_EXTRA, PUNCTUATION, ordinal_words, tr_lower

LETTERS = "a-zA-ZçğıöşüâîûÇĞİÖŞÜÂÎÛ"
UPPER = "A-ZÇĞİÖŞÜÂÎÛ"
VOWELS = set("aeıioöuüâîûAEIİOÖUÜÂÎÛ")
MONTHS = ["ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık"]
# Letter names used when an acronym is spelled out ("ABD" -> "a be de", "CHP" -> "ce ha pe", "PKK" -> "pe ka ka").
LETTER_NAMES = {
    "A": "a", "B": "be", "C": "ce", "Ç": "çe", "D": "de", "E": "e", "F": "fe", "G": "ge", "Ğ": "yumuşak ge",
    "H": "ha", "I": "ı", "İ": "i", "J": "je", "K": "ka", "L": "le", "M": "me", "N": "ne", "O": "o", "Ö": "ö",
    "P": "pe", "R": "re", "S": "se", "Ş": "şe", "T": "te", "U": "u", "Ü": "ü", "V": "ve", "Y": "ye", "Z": "ze",
    "Q": "kü", "W": "ve", "X": "iks", "Â": "a", "Î": "i", "Û": "u",
}
# Acronyms with a conventional spoken form (read as a word, or spelled irregularly).
SPOKEN_ACRONYMS = {
    "NATO": "nato", "NASA": "nasa", "UEFA": "uefa", "FIFA": "fifa", "UNESCO": "unesko", "UNICEF": "unisef",
    "ODTÜ": "odtü", "TÜBİTAK": "tübitak", "ASELSAN": "aselsan", "HAVELSAN": "havelsan", "ROKETSAN": "roketsan",
    "TOKİ": "toki", "TÜİK": "tüik", "BOTAŞ": "botaş", "TÜSİAD": "tüsiad", "MÜSİAD": "müsiad", "DİSK": "disk",
    "OPEC": "opek", "COVID": "kovid", "LED": "led", "RAM": "ram", "ROM": "rom", "SIM": "sim", "PIN": "pin",
    "GIF": "gif", "JPEG": "jipeg", "LAN": "lan", "WAN": "van", "PDF": "pe de fe", "USB": "u se be",
    "NBA": "en bi ey", "BBC": "bi bi si", "CNN": "si en en", "FBI": "ef bi ay", "CIA": "si ay ey",
    "AI": "ey ay", "OK": "okey", "TV": "te ve", "DJ": "di cey", "PC": "pe ce", "CEO": "si i o",
}
ABBREVIATIONS = {
    "Dr.": "doktor", "Prof.": "profesör", "Doç.": "doçent", "Yrd.": "yardımcı", "Av.": "avukat", "Uzm.": "uzman",
    "Op.": "operatör", "Arş.": "araştırma", "Gör.": "görevlisi", "Öğr.": "öğretim", "Sn.": "sayın", "Hz.": "hazreti",
    "Müh.": "mühendis", "Mah.": "mahallesi", "Cad.": "caddesi", "Sok.": "sokağı", "Apt.": "apartmanı",
    "Blv.": "bulvarı", "Tel.": "telefon", "No.": "numara",
    "vb.": "ve benzeri", "vs.": "vesaire", "vd.": "ve diğerleri", "örn.": "örneğin", "bkz.": "bakınız",
    "yy.": "yüzyıl", "yak.": "yaklaşık", "ort.": "ortalama", "maks.": "maksimum", "mah.": "mahallesi",
    "cad.": "caddesi", "sok.": "sokağı", "apt.": "apartmanı", "tel.": "telefon", "no.": "numara",
}
UNITS = {
    "km/sa": "kilometre", "km/s": "kilometre", "km/h": "kilometre",
    "km²": "kilometrekare", "km2": "kilometrekare", "m²": "metrekare", "m2": "metrekare", "m³": "metreküp",
    "m3": "metreküp", "km": "kilometre", "cm": "santimetre", "mm": "milimetre", "m": "metre", "kg": "kilogram",
    "gr": "gram", "g": "gram", "mg": "miligram", "lt": "litre", "ml": "mililitre", "L": "litre", "sn": "saniye",
    "dk": "dakika", "ms": "milisaniye", "GB": "gigabayt", "MB": "megabayt", "KB": "kilobayt", "TB": "terabayt",
    "mAh": "miliamper saat", "kWh": "kilovat saat", "kW": "kilovat", "W": "vat", "V": "volt", "Hz": "hertz",
    "kHz": "kilohertz", "MHz": "megahertz", "GHz": "gigahertz", "ha": "hektar", "fps": "kare",
}
CURRENCY_SYMBOLS = {"₺": "lira", "$": "dolar", "€": "avro", "£": "sterlin", "¥": "yen", "₽": "ruble", "₿": "bitcoin"}
CURRENCY_CODES = {"TL": "lira", "TRY": "lira", "USD": "dolar", "EUR": "avro", "GBP": "sterlin"}
SYMBOL_WORDS = {"&": " ve ", "+": " artı ", "=": " eşittir ", "×": " çarpı ", "÷": " bölü ", "±": " artı eksi ",
                "<": " küçüktür ", ">": " büyüktür ", "≈": " yaklaşık ", "~": " yaklaşık ", "%": " yüzde ",
                "°": " derece ", "№": " numara ", "§": " madde "}
QUOTES = {"«": '"', "»": '"', "‹": "'", "›": "'", "„": '"', "‟": '"', "“": '"', "”": '"', "‘": "'", "’": "'",
          "‚": "'", "`": "'", "´": "'", "′": "'", "″": '"', "–": "-", "—": "-", "―": "-", "‐": "-", "‑": "-", "⁄": "/",
          "−": "-", "…": "...", "·": ",", "•": ",", "|": ",", "[": "(", "]": ")", "{": "(", "}": ")"}
# Non-Turkish Latin letters: closest Turkish spelling.
FOREIGN = {"č": "ç", "ć": "ç", "Č": "Ç", "Ć": "Ç", "š": "ş", "ś": "ş", "Š": "Ş", "Ś": "Ş", "ž": "j", "Ž": "J",
           "ß": "ss", "æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe", "ø": "ö", "Ø": "Ö", "ł": "l", "Ł": "L",
           "đ": "d", "Đ": "D", "ð": "d", "þ": "t", "ə": "e", "Ə": "E"}
# `%` stays: turkish-v1 reads it next to a number (%50 -> yüzde elli); a lone % is rewritten before the final filter.
ALLOWED = set(PUNCTUATION) | LATIN_EXTRA | set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 %")


def tr_upper(text):
    """Turkish upper-casing: i -> İ and ı -> I before the generic upper-casing."""
    return text.replace("i", "İ").replace("ı", "I").upper()


def tr_title(word):
    return tr_upper(word[:1]) + tr_lower(word[1:])


@dataclass
class PreparedText:
    text: str
    changes: list = field(default_factory=list)  # human-readable notes, one per rewrite or removal


def _suffix_vowel(word):
    """Last vowel of a word decides Turkish vowel harmony: back (a ı o u) or front (e i ö ü)."""
    for c in reversed(tr_lower(word)):
        if c in "aıou":
            return "a"
        if c in "eiöü":
            return "e"
    return "e"


def locative(word):
    """Locative case of a number word: iki -> ikide, üç -> üçte, dört -> dörtte, altı -> altıda."""
    consonant = "t" if tr_lower(word)[-1] in "çfhkpsşt" else "d"
    return f"{word}{consonant}{_suffix_vowel(word)}"


def _roman(value):
    numerals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total, previous = 0, 0
    for c in reversed(value):
        n = numerals[c]
        total, previous = (total - n, previous) if n < previous else (total + n, n)
    return total


def _int_to_roman(n):
    out = ""
    for value, symbol in ((100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out, n = out + symbol, n - value
    return out


def spell(acronym):
    return " ".join(LETTER_NAMES.get(c, c.lower()) for c in acronym)


def _pronounceable(token):
    """Rough Turkish phonotactics: a vowel present, no three consonants in a row, no final consonant pair."""
    if not any(c in VOWELS for c in token):
        return False
    pattern = "".join("v" if c in VOWELS else "c" for c in token)
    return "ccc" not in pattern and not pattern.endswith("cc") and not pattern.startswith("cc")


class _Rewriter:
    def __init__(self):
        self.changes = []

    def sub(self, pattern, replacement, text, note, flags=0):
        def apply(match):
            new = replacement(match) if callable(replacement) else match.expand(replacement)
            if new != match.group(0):
                self.changes.append(f"{note}: {match.group(0).strip()!r} → {new.strip()!r}")
            return new

        return re.sub(pattern, apply, text, flags=flags)


def _number_words(n):
    from .turkish import number_words

    return number_words(n)


def prepare_text(text):
    """Rewrite free Turkish text into speakable words; never raises on symbols. Returns PreparedText."""
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    r = _Rewriter()
    text = unicodedata.normalize("NFKC", text)
    text = "".join(QUOTES.get(c, c) for c in text)
    # Shouted text (mostly capitals) is read as ordinary text, not as a list of acronyms.
    words = [w for w in re.findall(rf"[{LETTERS}]{{4,}}", text) if w not in SPOKEN_ACRONYMS]
    if len(words) >= 3 and sum(w == tr_upper(w) for w in words) / len(words) >= 0.8:
        lowered = tr_lower(text)
        text = re.sub(r"(^|[.!?]\s+)([a-zçğıöşü])", lambda m: m.group(1) + tr_upper(m.group(2)), lowered)
        r.changes.append("büyük harfli metin normal yazıya çevrildi")
    # Web addresses and e-mail.
    text = r.sub(r"\bhttps?://", "", text, "adres")
    text = r.sub(r"\bwww\.", "", text, "adres")
    text = r.sub(rf"\b([{LETTERS}0-9_.+-]+)@([{LETTERS}0-9-]+(?:\.[{LETTERS}0-9-]+)+)",
                 lambda m: m.group(1).replace(".", " nokta ").replace("_", " ") + " et "
                 + m.group(2).replace(".", " nokta "), text, "e-posta")
    text = r.sub(rf"(?<![{LETTERS}0-9])[@#]([{LETTERS}0-9_]+)", lambda m: m.group(1).replace("_", " "), text, "etiket")
    text = r.sub(rf"\b([{LETTERS}0-9-]+)\.(com|net|org|ai|io|co|gov|edu|tr|de|uk)(\.tr)?(/\S*)?",
                 lambda m: m.group(1) + " nokta " + m.group(2) + (" nokta tr" if m.group(3) else ""), text, "adres")
    # Dates: 23.09.2026, 23/09/2026, 23-09-2026 -> 23 eylül 2026 (the year keeps its suffix: 2026'da).
    def date(m):
        day, month = int(m.group(1)), int(m.group(2))
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return m.group(0)
        return f"{day} {MONTHS[month - 1]} {m.group(3)}"

    text = r.sub(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b", date, text, "tarih")
    # Clock times written with a dot or with a leading zero: 09.30'da -> 9 30'da, 14:00 -> 14, 14:05 -> 14 sıfır 5.
    def clock(m):
        hour, minute = int(m.group(2)), int(m.group(3))
        if hour > 24 or minute > 59:
            return m.group(0)
        spoken = str(hour) if minute == 0 else (f"{hour} sıfır {minute}" if minute < 10 else f"{hour} {minute}")
        return m.group(1) + spoken

    text = r.sub(r"((?:[Ss]aat\s+)|(?<![\d.,]))(\d{1,2})[:](\d{2})\b(?![.,:]\d)", clock, text, "saat")
    text = r.sub(r"((?:[Ss]aat\s+))(\d{1,2})[.](\d{2})\b(?![.,:]\d)", clock, text, "saat")
    text = r.sub(r"()(?<![\d.,])(0\d|1\d|2[0-3])[.](\d{2})(?=['’][" + LETTERS + r"])", clock, text, "saat")
    # Currency: symbols before or after the amount, and currency codes after it.
    for symbol, word in CURRENCY_SYMBOLS.items():
        s = re.escape(symbol)
        text = r.sub(rf"{s}\s?(\d+(?:[.,]\d+)*)", rf"\1 {word}", text, "para birimi")
        text = r.sub(rf"(\d+(?:[.,]\d+)*)\s?{s}", rf"\1 {word}", text, "para birimi")
        text = r.sub(s, f" {word} ", text, "para birimi")
    for code, word in CURRENCY_CODES.items():
        text = r.sub(rf"(\d+(?:[.,]\d+)*)\s?{code}\b", rf"\1 {word}", text, "para birimi")
    # Temperatures and angles.
    text = r.sub(r"(\d)\s?°\s?C\b", r"\1 derece", text, "sıcaklık")
    text = r.sub(r"(\d)\s?°\s?F\b", r"\1 fahrenhayt", text, "sıcaklık")
    # Speeds and units right after a number (suffixes such as km'lik stay attached: kilometrelik).
    text = r.sub(r"(\d+(?:[.,]\d+)*)\s?(?:km/sa|km/s|km/h)\b", r"saatte \1 kilometre", text, "birim")
    units = sorted(UNITS, key=len, reverse=True)
    unit_pattern = "|".join(re.escape(u) for u in units)
    text = r.sub(rf"(\d)\s?({unit_pattern})(?:['’]([{LETTERS}]+))?(?![{LETTERS}0-9²³])",
                 lambda m: f"{m.group(1)} {UNITS[m.group(2)]}{m.group(3) or ''}", text, "birim")
    # Fractions 1/2 -> ikide bir; other slashes between numbers -> bölü; between words -> space.
    def fraction(m):
        numerator, denominator = int(m.group(1)), int(m.group(2))
        if 0 < numerator < denominator <= 100:
            return f"{locative(_number_words(denominator))} {_number_words(numerator)}"
        return f"{m.group(1)} bölü {m.group(2)}"

    text = r.sub(r"\b(\d{1,3})/(\d{1,3})\b", fraction, text, "kesir")
    text = r.sub(r"(\d)\s*/\s*(\d)", r"\1 bölü \2", text, "bölü")
    text = r.sub(rf"([{LETTERS}])/([{LETTERS}])", r"\1 \2", text, "eğik çizgi")
    # Negative numbers and approximate values.
    text = r.sub(r"(?<![\w\d])-(?=\d)", "eksi ", text, "eksi")
    # Phone-style groups with a leading zero read as "sıfır beş yüz otuz iki".
    text = r.sub(r"(?<![\d.,])0(\d{3})(?=\s\d{3}\b)", r"sıfır \1", text, "telefon")
    # Roman ordinals: II. Dünya Savaşı -> ikinci Dünya Savaşı, XIX. yüzyıl -> on dokuzuncu yüzyıl.
    def roman(m):
        token = m.group(1)
        value = _roman(token)
        if not 0 < value <= 100 or _int_to_roman(value) != token or token in {"L", "C"}:
            return m.group(0)
        return ordinal_words(value) + (m.group(2) or "")

    text = r.sub(r"\b([IVXLC]{1,7})\.(?=\s+[" + LETTERS + r"])()", roman, text, "Roma rakamı")
    text = r.sub(r"\b([IVXLC]{1,7})\.?( yüzyıl)", roman, text, "Roma rakamı")
    # Abbreviations with a period; keep a sentence end when the abbreviation closes the text.
    for abbreviation, word in sorted(ABBREVIATIONS.items(), key=lambda item: -len(item[0])):
        a = re.escape(abbreviation)
        word = tr_title(word) if abbreviation[0].isupper() else word  # "Dr. Ahmet" -> "Doktor Ahmet"
        text = r.sub(rf"(?<![{LETTERS}]){a}(?=\s*$)", word + ".", text, "kısaltma")
        if abbreviation[0].islower():  # "armut vs. Sonra" keeps its sentence end; "Dr. Ahmet" does not
            text = r.sub(rf"(?<![{LETTERS}]){a}(?=\s+[{UPPER}])", word + ".", text, "kısaltma")
        text = r.sub(rf"(?<![{LETTERS}]){a}", word, text, "kısaltma")
    # Letters glued to digits (5G, 3D, COVID-19): split, and spell a single capital letter.
    text = r.sub(rf"(?<=[{LETTERS}])-(?=\d)", " ", text, "tire")
    text = r.sub(rf"(\d)([{UPPER}])(?![{LETTERS}])", lambda m: f"{m.group(1)} {LETTER_NAMES.get(m.group(2), m.group(2))}", text, "harf")
    # Acronyms: spelled out unless pronounceable or known as words.
    def acronym(m):
        token = m.group(1)
        suffix = m.group(2) or ""
        if token in SPOKEN_ACRONYMS:
            spoken = SPOKEN_ACRONYMS[token]
        elif len(token) >= 4 and _pronounceable(token):
            spoken = tr_lower(token)
        else:
            spoken = spell(token)
        return spoken + (suffix[1:] if suffix else "")

    text = r.sub(rf"(?<![{LETTERS}])([{UPPER}]{{2,7}})(['’][{LETTERS}]+)?(?![{LETTERS}])", acronym, text, "kısaltma")
    # Longer words in capitals are ordinary words written in capitals: İSTANBUL'da -> İstanbul'da.
    text = r.sub(rf"(?<![{LETTERS}])([{UPPER}]{{8,}})(?![{LETTERS}])", lambda m: tr_title(m.group(1)), text, "büyük harf")
    # Remaining symbols with a spoken form.
    for symbol, word in SYMBOL_WORDS.items():
        # A percent sign next to a number is spelled by turkish-v1 itself (%50, 50 %).
        pattern = r"(?<!\d)(?<!\d )%(?! ?\d)" if symbol == "%" else re.escape(symbol)
        text = r.sub(pattern, word, text, "sembol")
    # Foreign letters: map or strip diacritics; drop everything that cannot be spoken (emoji, other scripts).
    out = []
    for c in text:
        if c in ALLOWED or c.isspace():
            out.append(c)
            continue
        if c in FOREIGN:
            out.append(FOREIGN[c])
            r.changes.append(f"yabancı harf: {c!r} → {FOREIGN[c]!r}")
            continue
        base = "".join(b for b in unicodedata.normalize("NFKD", c) if unicodedata.category(b) != "Mn")
        if base and all(b in ALLOWED for b in base):
            out.append(base)
            r.changes.append(f"yabancı harf: {c!r} → {base!r}")
            continue
        out.append(" ")
        name = unicodedata.name(c, f"U+{ord(c):04X}")
        r.changes.append(f"okunamayan karakter çıkarıldı: {c!r} ({name.lower()})")
    text = "".join(out)
    # Tidy punctuation and whitespace: collapse repeats, no space before punctuation.
    text = re.sub(r"([!?])[!?]+", r"\1", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([.,;:!?])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return PreparedText(text, r.changes)


def speakable(text):
    """prepare_text + turkish-v1 normalization; returns (normalized text, changes). Raises only on empty text."""
    from .text import normalize

    prepared = prepare_text(text)
    candidate = prepared.text
    for _ in range(8):
        try:
            return normalize(candidate, "turkish-v1"), prepared.changes
        except ValueError as error:
            match = re.search(r"'(.)'", str(error))
            if not match or "Empty" in str(error):
                raise ValueError("Metin boş ya da okunabilir karakter içermiyor") from error
            prepared.changes.append(f"okunamayan karakter çıkarıldı: {match.group(1)!r}")
            candidate = candidate.replace(match.group(1), " ")
    raise ValueError("Metin normalleştirilemedi")


def split_sentences(text, max_chars=180, min_chars=40):
    """Sentence chunks of at most `max_chars` characters (hard limit applies to single long sentences too).

    Sentences end at . ! ? … or a line break. Short neighbours are merged up to `max_chars` so that very short chunks
    (which the model reads less reliably) are rare; long sentences are cut at ; : , and finally at a space. Returns a
    list of (chunk, pause) where pause is "sentence" or "clause" (the boundary after the chunk).
    """
    text = text.strip()
    if not text:
        return []
    pieces = []
    for line in text.split("\n"):
        for sentence in re.split(r"(?<=[.!?])[\"')]*\s+", line.strip()):
            if sentence.strip():
                pieces.extend(_split_long(sentence.strip(), max_chars))
    chunks = []
    for piece, kind in pieces:
        if chunks and chunks[-1][1] == "sentence" and len(chunks[-1][0]) < min_chars \
                and len(chunks[-1][0]) + 1 + len(piece) <= max_chars:
            chunks[-1] = (chunks[-1][0] + " " + piece, kind)
        elif chunks and kind == "sentence" and len(piece) < min_chars and chunks[-1][1] == "sentence" \
                and len(chunks[-1][0]) + 1 + len(piece) <= max_chars:
            chunks[-1] = (chunks[-1][0] + " " + piece, kind)
        else:
            chunks.append((piece, kind))
    return chunks


def _split_long(sentence, max_chars):
    if len(sentence) <= max_chars:
        return [(sentence, "sentence")]
    for separator in (r"(?<=[;:])\s+", r"(?<=,)\s+", r"\s+"):
        parts = [p for p in re.split(separator, sentence) if p]
        if len(parts) < 2:
            continue
        out, current = [], ""
        for part in parts:
            if current and len(current) + 1 + len(part) > max_chars:
                out.append(current)
                current = part
            else:
                current = f"{current} {part}".strip()
        if current:
            out.append(current)
        if all(len(p) <= max_chars for p in out) or separator == r"\s+":
            result = []
            for p in out:
                if len(p) > max_chars:  # a single word longer than the limit
                    result.extend((p[i : i + max_chars], "clause") for i in range(0, len(p), max_chars))
                else:
                    result.append((p, "clause"))
            result[-1] = (result[-1][0], "sentence")
            return result
        # Some part is still too long: split those parts further with the next separator.
        result = []
        for p in out:
            result.extend(_split_long(p, max_chars) if len(p) > max_chars else [(p, "clause")])
        result[-1] = (result[-1][0], "sentence")
        return result
    return [(sentence, "sentence")]
