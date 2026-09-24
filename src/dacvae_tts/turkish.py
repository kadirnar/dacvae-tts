"""Turkish transcript normalization (numbers to words, script checks) and Turkish-aware WER text.

Turkish writes numbers as separate words ("iki bin on"), lower-cases dotted İ to i and dotless I to ı,
and reads percentages as "yüzde N". Podcast transcripts here contain digits in about 10% of rows,
so instead of rejecting them the digits are spelled out; a row is rejected only if something
unreadable remains (digits, non-Latin script).

Versions are frozen once caches or scores exist: `turkish-v1` (training) and the `turkish-v1` metric text keep their
exact output, known mistakes included ("4'e" -> "dörte"). Fixes go into new versions, such as the `turkish-v2`
training text (`normalize_turkish_v2`).
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
VOWELS = set("aeıioöuüâîû")
# Abbreviations with a fixed spoken form. The synthesis frontend expands all of them; the training normalization
# `turkish-v2` and the `turkish-v2` metric text only expand SAFE_ABBREVIATIONS (below).
ABBREVIATIONS = {
    "Dr.": "doktor", "Prof.": "profesör", "Doç.": "doçent", "Yrd.": "yardımcı", "Av.": "avukat", "Uzm.": "uzman",
    "Op.": "operatör", "Arş.": "araştırma", "Gör.": "görevlisi", "Öğr.": "öğretim", "Sn.": "sayın", "Hz.": "hazreti",
    "Müh.": "mühendis", "Mah.": "mahallesi", "Cad.": "caddesi", "Sok.": "sokağı", "Apt.": "apartmanı",
    "Blv.": "bulvarı", "Tel.": "telefon", "No.": "numara",
    "vb.": "ve benzeri", "vs.": "vesaire", "vd.": "ve diğerleri", "örn.": "örneğin", "bkz.": "bakınız",
    "yy.": "yüzyıl", "yak.": "yaklaşık", "ort.": "ortalama", "maks.": "maksimum", "mah.": "mahallesi",
    "cad.": "caddesi", "sok.": "sokağı", "apt.": "apartmanı", "tel.": "telefon", "no.": "numara",
    "T.C.": "te ce", "A.Ş.": "anonim şirketi", "Ltd. Şti.": "limited şirketi", "Ltd.Şti.": "limited şirketi",
    "Ltd.": "limited", "Şti.": "şirketi", "Öğr. Gör.": "öğretim görevlisi", "Arş. Gör.": "araştırma görevlisi",
}
# Expanded inside training transcripts and metric text: abbreviations that cannot be an ordinary word followed by a
# full stop. Left out on purpose: "Av." (av, hunt), "Gör." / "Arş." alone (gör, arş), "Hz." (hertz after a
# number), "No." (English), "Op.", "Tel." / "tel." (tel, wire), "sok." / "yak." (imperatives) and the address words.
# Frozen with turkish-v2: tests pin every expansion.
SAFE_ABBREVIATIONS = {
    key: ABBREVIATIONS[key]
    for key in ("T.C.", "A.Ş.", "Ltd. Şti.", "Ltd.Şti.", "Ltd.", "Şti.", "Öğr. Gör.", "Arş. Gör.", "Dr.", "Prof.",
                "Doç.", "Yrd.", "Uzm.", "Öğr.", "Müh.", "Sn.", "vb.", "vs.", "vd.", "örn.", "bkz.", "yy.")
}


def tr_lower(text):
    """Turkish case folding: İ -> i and I -> ı before the generic lower-casing."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def tr_upper(text):
    """Turkish upper-casing: i -> İ and ı -> I before the generic upper-casing."""
    return text.replace("i", "İ").replace("ı", "I").upper()


def tr_title(word):
    return tr_upper(word[:1]) + tr_lower(word[1:])


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
UPPER = "A-ZÇĞİÖŞÜÂÎÛ"


def abbreviation_rules(table):
    """(pattern, replacement) pairs expanding the abbreviations of `table`, longest first.

    A capitalized abbreviation with a one-word expansion is written capitalized ("Dr. Ahmet" -> "Doktor Ahmet");
    longer expansions stay lower-case ("A.Ş." -> "anonim şirketi"). The full stop of an abbreviation that ends the
    text, or of a lower-case one before a capitalized word, also ends the sentence ("armut vs. Sonra" -> "armut
    vesaire. Sonra").
    """
    rules = []
    for abbreviation, word in sorted(table.items(), key=lambda item: -len(item[0])):
        a = re.escape(abbreviation)
        word = tr_title(word) if abbreviation[0].isupper() and " " not in word else word
        rules.append((rf"(?<![{LETTER}]){a}(?=\s*$)", word + "."))
        if abbreviation[0].islower():
            rules.append((rf"(?<![{LETTER}]){a}(?=\s+[{UPPER}])", word + "."))
        rules.append((rf"(?<![{LETTER}]){a}", word))
    return rules


SAFE_ABBREVIATION_RULES = [(re.compile(pattern), word) for pattern, word in abbreviation_rules(SAFE_ABBREVIATIONS)]


def expand_safe_abbreviations(text):
    """T.C. -> te ce, A.Ş. -> anonim şirketi, Ltd. Şti. -> limited şirketi, Dr. -> Doktor, vb. -> ve benzeri."""
    if "." not in text:  # every abbreviation ends with a full stop; most transcripts skip all the patterns
        return text
    for pattern, word in SAFE_ABBREVIATION_RULES:
        text = pattern.sub(word, text)
    return text


def normalize_numbers(text, soften=False):
    """Spell out the numeric expressions Turkish podcast transcripts contain.

    `soften` (turkish-v2) voices the final t of "dört" before a vowel-initial suffix, as Turkish does: 4'e -> dörde,
    14'ün -> on dördün, %4'ü -> yüzde dördü, 2024'e -> iki bin yirmi dörde (without it: dörte, on dörtün). No other
    number word changes (üçe, kırka, sekize are right as written) and ordinals are dördüncü either way; the suffix
    vowels were written for "dört" already, so its harmony stays correct.
    """
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
        words = _int_words(number)
        if soften and words.endswith("dört") and tr_lower(suffix[0]) in VOWELS:
            words = words[:-1] + "d"
        return words + suffix

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


def normalize_turkish(text, version="turkish-v1"):
    """Turkish transcript -> model text: NFKC, numbers as words, script check, whitespace.

    `turkish-v2` also expands SAFE_ABBREVIATIONS (T.C. -> te ce, A.Ş. -> anonim şirketi) and softens "dört" before
    a vowel suffix (4'e -> dörde); everything else is byte-identical to `turkish-v1`.
    """
    if version not in {"turkish-v1", "turkish-v2"}:
        raise ValueError(f"Unknown Turkish normalization version: {version}")
    v2 = version == "turkish-v2"
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[“”„]", '"', text)
    text = text.replace("‘", "'").replace("’", "'").replace("–", "-").replace("—", "-").replace("…", "...")
    if v2:
        text = expand_safe_abbreviations(text)
    text = normalize_numbers(text, soften=v2)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # "kelime ." -> "kelime."
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("Empty transcript")
    check_script(text)
    return text


def normalize_turkish_v2(text):
    return normalize_turkish(text, "turkish-v2")


def metric_text_turkish(text):
    """WER/CER normalization for Turkish: numbers spelled out, Turkish lower-case, letters only."""
    text = unicodedata.normalize("NFKC", text)
    text = normalize_numbers(text)
    text = tr_lower(text)
    text = "".join(c if c.isalnum() or c.isspace() else ("" if c in "'’" else " ") for c in text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(text.split())
