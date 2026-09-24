"""Turkish transcript normalization (numbers to words, script checks) and Turkish-aware WER text.

Turkish writes numbers as separate words ("iki bin on"), lower-cases dotted İ to i and dotless I to ı,
and reads percentages as "yüzde N". Podcast transcripts here contain digits in about 10% of rows,
so instead of rejecting them the digits are spelled out; a row is rejected only if something
unreadable remains (digits, non-Latin script).

Versions are frozen once caches or scores exist: `turkish-v1` (training) and the `turkish-v1` metric text keep their
exact output, known mistakes included ("4'e" -> "dörte"). Fixes go into new versions: the `turkish-v2` training
text (`normalize_turkish_v2`) and the `turkish-v2` metric text (`metric_text_turkish_v2`).
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

    Abbreviations are often written without a space: the parts of a multi-word one may be joined ("Öğr.Gör." ->
    "öğretim görevlisi") and a capital letter right after the full stop starts the next word, which gets a space
    ("Prof.Dr. Ahmet" -> "Profesör Doktor Ahmet", not "ProfesörDr."). A lower-case letter there is a suffix, which
    Turkish attaches to such abbreviations without an apostrophe ("16. yy.da" -> "yüzyılda"), so it stays attached.
    """
    rules = []
    for abbreviation, word in sorted(table.items(), key=lambda item: -len(item[0])):
        a = r"\s*".join(re.escape(part) for part in abbreviation.split(" "))
        word = tr_title(word) if abbreviation[0].isupper() and " " not in word else word
        rules.append((rf"(?<![{LETTER}]){a}(?=\s*$)", word + "."))
        if abbreviation[0].islower():
            rules.append((rf"(?<![{LETTER}]){a}(?=\s+[{UPPER}])", word + "."))
            rules.append((rf"(?<![{LETTER}]){a}(?=[{UPPER}])", word + ". "))
        rules.append((rf"(?<![{LETTER}]){a}(?=[{UPPER}])", word + " "))
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
    """WER/CER normalization for Turkish: numbers spelled out, Turkish lower-case, letters only.

    Metric version `turkish-v1`, frozen so that published scores stay comparable; metric_text_turkish_v2 fixes
    its mistakes.
    """
    text = unicodedata.normalize("NFKC", text)
    text = normalize_numbers(text)
    text = tr_lower(text)
    text = "".join(c if c.isalnum() or c.isspace() else ("" if c in "'’" else " ") for c in text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(text.split())


# Metric text `turkish-v2`. Frozen like every metric version: editing a table or rule below changes turkish-v2 WER/CER,
# so improvements go into a new version.
METRIC_MARKS = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "ʼ": "'", "′": "'", "‑": "-", "‐": "-"})
# Written after a number: 5 km, 5km'de, 25°C, 3 TL. Currency symbols go before or after the amount.
METRIC_UNITS = {
    "km/sa": "kilometre", "km/s": "kilometre", "km²": "kilometrekare", "m²": "metrekare", "m³": "metreküp",
    "km": "kilometre", "cm": "santimetre", "mm": "milimetre", "m": "metre", "kg": "kilogram", "mg": "miligram",
    "gr": "gram", "g": "gram", "lt": "litre", "ml": "mililitre", "sn": "saniye", "dk": "dakika",
    "KB": "kilobayt", "MB": "megabayt", "GB": "gigabayt", "TB": "terabayt", "TL": "lira", "°C": "derece", "°": "derece",
}
METRIC_CURRENCY = {"₺": "lira", "$": "dolar", "€": "avro", "£": "sterlin"}
# Whole-word spelling variants (after lower-casing): an ASR hypothesis and a reference may spell the same spoken word
# differently. "ok" is not here: it is the Turkish word for arrow.
METRIC_VARIANTS = {
    "euro": "avro", "öro": "avro", "yuro": "avro", "tl": "lira", "kovid": "covid", "okay": "okey",
    "herşey": "her şey", "birşey": "bir şey", "hiçbirşey": "hiçbir şey", "yanlız": "yalnız", "şöför": "şoför",
    "orjinal": "orijinal", "süpriz": "sürpriz", "inşaallah": "inşallah", "maşaallah": "maşallah",
}
# All-caps Latin acronyms whose I is a dotted i: "COVID" must fold like "Covid" (covid), not to "covıd". An all-caps
# token with Q, W or X cannot be Turkish and folds the same way; every other all-caps word keeps Turkish casing
# (IRAK -> ırak, ISPARTA -> ısparta).
LATIN_I_ACRONYMS = {
    "AI", "API", "BMI", "CIA", "COVID", "FBI", "FIFA", "GIF", "IBAN", "IBM", "ID", "IKEA", "IMF", "IOS", "IP", "ISIS",
    "IT", "LINUX", "MINI", "MIT", "PIN", "SIM", "UNICEF", "WIFI",
}
# Roman numerals written with I, V and X only (1-39): C, L, D and M also spell acronyms and initials (CD, MI, DC).
ROMAN = r"(?=[IVX])X{0,3}(?:IX|IV|V?I{0,3})"
_METRIC_RULES = {
    "hyphen": re.compile(rf"(?<=[{LETTER}])-(?=[{LETTER}])"),
    # XIX. yüzyıl, XIX yüzyıl, V. yüzyıl; "II. Dünya", "XVI.yüzyıl"; a single letter only before a capitalized word
    # ("I. Dünya", "V. Murat", but "Bay X. geldi" stays); standalone numerals of two or more letters become digits
    # that the number rules read with their suffix (Faz II -> faz iki, II'de -> ikide, IV'üncü -> dördüncü).
    "roman_century": re.compile(rf"(?<![\w.'-])({ROMAN})\.?(?=\s*yüzyıl)"),
    "roman_ordinal": re.compile(rf"(?<![\w.'-])((?=[IVX]{{2}}){ROMAN}\.(?=\s*[{LETTER}])|[IVX]\.(?=\s*[{UPPER}]))"),
    "roman_cardinal": re.compile(rf"(?<![\w.'-])((?=[IVX]{{2}}){ROMAN})(?![\w-])"),
    "latin_i": re.compile(rf"(?<![{LETTER}])(?=[A-Z]*I)[A-Z]{{2,}}(?![{LETTER}])"),
    "camel_i": re.compile(r"(?<=[a-zçğıöşü])I"),
    "clock_saat": re.compile(r"(\b(?i:saat)\s+)(\d{1,2})[.:](\d{2})\b(?![.,:]\d)"),
    "clock_suffix": re.compile(rf"()(?<![\d.,:])(\d{{1,2}})[.:](\d{{2}})(?=['][{LETTER}])"),
    "clock_colon": re.compile(r"()(?<![\d.,:])(\d{1,2}):(\d{2})\b(?![.,:]\d)"),
    "glued_ordinal": re.compile(r"(?<![\w.,])(\d+)\.(?=[a-zçğıöşü])"),
    # A suffix on TL was written for "te le"; on "lira" it takes back vowels: 5 TL'ye -> 5 liraya, TL'den -> liradan.
    "lira_suffix": re.compile(rf"(\d)\s?TL'([{LETTER}]+)"),
    "unit": re.compile(
        rf"(\d)\s?({'|'.join(re.escape(u) for u in sorted(METRIC_UNITS, key=len, reverse=True))})(?![{LETTER}0-9²³/])"
    ),
}
_BACK_VOWELS = str.maketrans("eiü", "aıı")


def roman_value(token):
    values = {"I": 1, "V": 5, "X": 10}
    return sum(
        -values[c] if i + 1 < len(token) and values[token[i + 1]] > values[c] else values[c]
        for i, c in enumerate(token)
    )


def _clock(match):
    """saat 3.30 -> saat 3 30, 3.30'a -> 3 30'a, 14.00'te -> 14'te, 09:05 -> 9 05 (read "dokuz sıfır beş")."""
    hour, minute = int(match.group(2)), int(match.group(3))
    if hour > 24 or minute > 59:
        return match.group(0)
    return f"{match.group(1)}{hour}" + ("" if minute == 0 else f" {match.group(3)}")


def _fold_marks(text):
    """Drop combining marks and fold accented letters (â -> a, î -> i, û -> u, ô -> o, é -> e); keep ç ğ ö ş ü."""
    return "".join(
        c if c < "\x80" or c in "çğöşü"
        else "".join(b for b in unicodedata.normalize("NFD", c) if unicodedata.category(b) != "Mn")
        for c in text
    )


def metric_text_turkish_v2(text):
    """WER/CER normalization `turkish-v2`: the `turkish-v1` metric text without its known mistakes (issue #5).

    Order: apostrophe/hyphen variants, NFKC, then rewrites that need the original case and punctuation, Turkish
    lower-casing, combining-mark removal and the character filter:
    - intra-word hyphens join (e-posta -> eposta, Wi-Fi -> wifi) instead of splitting a word in two;
    - SAFE_ABBREVIATIONS expand (T.C. -> te ce, A.Ş. -> anonim şirketi, Dr. -> doktor, vb. -> ve benzeri);
    - Roman numerals (I-XXXIX): "II. Dünya", "XVI.yüzyıl", "XIX yüzyıl", "I. Dünya" -> ordinals; other standalone
      numerals of two or more letters -> cardinals (Faz II -> faz iki); v1 produced "ıı";
    - all-caps Latin acronyms (LATIN_I_ACRONYMS, or any with Q/W/X) and a capital I inside a lower-case word
      (LinkedIn) keep a dotted i: COVID -> covid, not covıd;
    - clock times: saat 3.30 -> saat üç otuz, 3.30'a -> üç otuza, 14:00 -> on dört (v1: "üç nokta otuza");
    - ordinals without a space: 21.yüzyıl -> yirmi birinci yüzyıl;
    - currency symbols and units after a number: €50 -> elli avro, 5km'de -> beş kilometrede, 3 TL'ye -> üç liraya;
    - numbers as in turkish-v1 with "dört" softening (4'e -> dörde);
    - combining marks are removed after lower-casing, so "i̇stanbul" (a non-Turkish lower() of İstanbul) is
      "istanbul" rather than "i stanbul"; â/î/û/ô and foreign accents fold (kâr -> kar);
    - apostrophes are deleted (İsveç'ten -> isveçten), other punctuation separates words;
    - METRIC_VARIANTS unify spellings (euro -> avro, herşey -> her şey).
    Plain sentences (letters, apostrophes and sentence punctuation, no abbreviation, all-caps word or variant
    spelling) come out exactly as with `turkish-v1`; this holds for all 495 Freya-TR-Eval sentences. The default
    for Turkish stays `turkish-v1`; pass `turkish-v2` explicitly (`--metric-normalization turkish-v2`).
    """
    rules = _METRIC_RULES
    text = unicodedata.normalize("NFKC", text.translate(METRIC_MARKS))
    text = rules["hyphen"].sub("", text)
    text = expand_safe_abbreviations(text)
    text = rules["roman_century"].sub(lambda m: ordinal_words(roman_value(m.group(1))) + " ", text)
    text = rules["roman_ordinal"].sub(lambda m: ordinal_words(roman_value(m.group(1)[:-1])) + " ", text)
    text = rules["roman_cardinal"].sub(lambda m: str(roman_value(m.group(1))), text)
    text = rules["latin_i"].sub(
        lambda m: m.group().lower() if m.group() in LATIN_I_ACRONYMS or re.search("[QWX]", m.group()) else m.group(),
        text,
    )
    text = rules["camel_i"].sub("i", text)
    for rule in ("clock_saat", "clock_suffix", "clock_colon"):
        text = rules[rule].sub(_clock, text)
    text = rules["glued_ordinal"].sub(r"\1. ", text)
    for symbol, word in METRIC_CURRENCY.items():
        s = re.escape(symbol)
        text = re.sub(rf"{s}\s?(\d+(?:[.,]\d+)*)", rf"\1 {word}", text)
        text = re.sub(rf"(\d)\s?{s}", rf"\1 {word}", text)
    text = rules["lira_suffix"].sub(lambda m: f"{m.group(1)} lira" + m.group(2).translate(_BACK_VOWELS), text)
    text = rules["unit"].sub(lambda m: f"{m.group(1)} {METRIC_UNITS[m.group(2)]}", text)
    text = normalize_numbers(text, soften=True)
    text = _fold_marks(tr_lower(text))
    text = "".join(c if c.isalnum() or c.isspace() else ("" if c == "'" else " ") for c in text)
    return " ".join(" ".join(METRIC_VARIANTS.get(word, word) for word in text.split()).split())
