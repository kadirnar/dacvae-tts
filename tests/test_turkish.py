import json
from pathlib import Path

import pytest

from dacvae_tts.metrics import error_counts, metric_text
from dacvae_tts.text import normalize, tokenize
from dacvae_tts.turkish import (
    SAFE_ABBREVIATIONS,
    metric_text_turkish,
    metric_text_turkish_v2,
    normalize_turkish,
    normalize_turkish_v2,
    number_words,
    ordinal_words,
    tr_lower,
)


def test_number_words():
    assert number_words(0) == "sıfır"
    assert number_words(11) == "on bir"
    assert number_words(100) == "yüz"
    assert number_words(1000) == "bin"
    assert number_words(2010) == "iki bin on"
    assert number_words(1000000) == "bir milyon"
    assert number_words(12345) == "on iki bin üç yüz kırk beş"
    assert ordinal_words(12) == "on ikinci"
    assert ordinal_words(4) == "dördüncü"


def test_normalize_turkish_numbers():
    assert normalize_turkish("%99'u senin olsun") == "yüzde doksan dokuzu senin olsun"
    assert normalize_turkish("1920'lerde 12. nesil") == "bin dokuz yüz yirmilerde on ikinci nesil"
    assert normalize_turkish("Saat 12:30'da, 3,5 lira, 17.5 sürüm") == "Saat on iki otuzda, üç virgül beş lira, on yedi nokta beş sürüm"
    assert normalize_turkish("T24'te 350D ile") == "T yirmi dörtte üç yüz elli D ile"
    assert normalize_turkish("1.000.000 lira 3'üncü") == "bir milyon lira üçüncü"
    assert normalize_turkish("3-4 milyar") == "üç dört milyar"


def test_normalize_turkish_rejects_foreign_script():
    with pytest.raises(ValueError):
        normalize_turkish("kelime ابت")
    with pytest.raises(ValueError):
        normalize_turkish("   ")


def test_turkish_case_folding_and_metric():
    assert tr_lower("İSTANBUL IĞDIR") == "istanbul ığdır"
    assert metric_text_turkish("İsveç'ten 86 kişi!") == "isveçten seksen altı kişi"
    assert metric_text("İsveç'ten 86 kişi!", "turkish-v1") == "isveçten seksen altı kişi"
    counts = error_counts("Sadece 86 kişi İsveç'ten aldı.", "sadece seksen altı kişi isveçten aldı", "turkish-v1")
    assert counts["wer"] == 0 and counts["cer"] == 0


def test_text_versions_and_tokens():
    assert normalize("2 kişi", "turkish-v1") == "iki kişi"
    tokens, segments = tokenize("", "Şu 2 kişi", version="turkish-v1", layout="joined")
    assert tokens.tolist()[1:-1] == [b + 4 for b in "Şu iki kişi".encode("utf-8")]
    assert segments.sum() == len(tokens)


# turkish-v1 is frozen: existing caches and checkpoints hold exactly these outputs, known mistakes included (issue #5).
V1_PINS = {
    "4'e": "dörte",
    "4'ü": "dörtü",
    "%4'ü": "yüzde dörtü",
    "2024'e": "iki bin yirmi dörte",
    "14'ün": "on dörtün",
    "T.C. vatandaşı": "T.C. vatandaşı",
    "Koç Holding A.Ş.": "Koç Holding A.Ş.",
    "Yılmaz Ltd. Şti.": "Yılmaz Ltd. Şti.",
    "Dr. Ahmet": "Dr. Ahmet",
    "3.30'a": "üç nokta otuza",
    "21.yüzyıl": "yirmi bir.yüzyıl",
    "e-posta": "e-posta",
}
# Plain sentences in the style of Freya-TR-Eval (letters and punctuation only: no digits, no circumflexes).
FREYA_LIKE = [
    "Yemekten sonra bulaşıkları makineye dizme işini bugün ben hallederim.",
    "Çocuğun kıvırcık saçları ve kocaman kara gözleri vardı.",
    "Irmak kenarında oturup İstanbul'un ışıklarını seyrettik, değil mi?",
    "Öğretmenimiz ödevleri cuma gününe kadar teslim etmemizi istedi!",
    "Ağaçların arasından süzülen güneş ışığı, şehrin gürültüsünü unutturdu.",
    "Iğdır'dan gelen misafirler, Ankara'da iki gece kalacaklarmış.",
    "Okçular ok atarken sen yine de dikkatli ol.",
]


@pytest.mark.parametrize("raw, v1", V1_PINS.items())
def test_turkish_v1_outputs_are_pinned(raw, v1):
    assert normalize_turkish(raw) == v1
    assert normalize(raw, "turkish-v1") == v1


@pytest.mark.parametrize(
    "raw, v2",
    [
        # Issue #5 table: "dört" softens before a vowel suffix.
        ("4'e", "dörde"),
        ("4'ü", "dördü"),
        ("%4'ü", "yüzde dördü"),
        ("2024'e", "iki bin yirmi dörde"),
        ("14'ün", "on dördün"),
        ("4'ünde", "dördünde"),
        ("4'er", "dörder"),
        ("3,4'ü", "üç virgül dördü"),
        ("1.004'e", "bin dörde"),
        ("04'e", "sıfır dörde"),
        ("Saat 14:04'e", "Saat on dört sıfır dörde"),
        # ... not before a consonant, and no other number word softens; ordinals were right already.
        ("4'te", "dörtte"),
        ("4'lü", "dörtlü"),
        ("4'ten", "dörtten"),
        ("4'üncü", "dördüncü"),
        ("4. sınıf", "dördüncü sınıf"),
        ("3'e", "üçe"),
        ("40'a", "kırka"),
        ("8'e", "sekize"),
        # Issue #5 table: abbreviations are expanded.
        ("T.C. vatandaşıyım.", "te ce vatandaşıyım."),
        ("Koç Holding A.Ş.", "Koç Holding anonim şirketi."),
        ("Yılmaz Ltd. Şti. kuruldu", "Yılmaz limited şirketi kuruldu"),
        ("Yılmaz Ltd.Şti.", "Yılmaz limited şirketi."),
        ("Prof. Dr. Ayşe geldi", "Profesör Doktor Ayşe geldi"),
        ("Arş. Gör. Ali", "araştırma görevlisi Ali"),
        ("elma, armut vb. Sonra geldi", "elma, armut ve benzeri. Sonra geldi"),
        ("16. yy.", "on altıncı yüzyıl."),
        # Abbreviations that could be an ordinary word before a full stop stay as they are.
        ("Ateşi yak.", "Ateşi yak."),
        ("Bir tel.", "Bir tel."),
        ("Gel, gör.", "Gel, gör."),
        ("Av. sezonu", "Av. sezonu"),
    ],
)
def test_turkish_v2_softening_and_abbreviations(raw, v2):
    assert normalize_turkish_v2(raw) == v2
    assert normalize(raw, "turkish-v2") == v2


def test_turkish_v2_is_v1_elsewhere_and_pins_its_abbreviations():
    for sentence in FREYA_LIKE + ["1920'lerde 12. nesil", "Saat 12:30'da, 3,5 lira", "%99'u senin olsun"]:
        assert normalize_turkish_v2(sentence) == normalize_turkish(sentence)
    # Frozen with turkish-v2: an edit here changes every cache prepared with it.
    assert SAFE_ABBREVIATIONS == {
        "T.C.": "te ce", "A.Ş.": "anonim şirketi", "Ltd. Şti.": "limited şirketi", "Ltd.Şti.": "limited şirketi",
        "Ltd.": "limited", "Şti.": "şirketi", "Öğr. Gör.": "öğretim görevlisi", "Arş. Gör.": "araştırma görevlisi",
        "Dr.": "doktor", "Prof.": "profesör", "Doç.": "doçent", "Yrd.": "yardımcı", "Uzm.": "uzman",
        "Öğr.": "öğretim", "Müh.": "mühendis", "Sn.": "sayın", "vb.": "ve benzeri", "vs.": "vesaire",
        "vd.": "ve diğerleri", "örn.": "örneğin", "bkz.": "bakınız", "yy.": "yüzyıl",
    }
    tokens, _ = tokenize("", "4'e", version="turkish-v2", layout="joined")
    assert tokens.tolist()[1:-1] == [b + 4 for b in "dörde".encode("utf-8")]
    with pytest.raises(ValueError):
        normalize_turkish("x", "turkish-v3")
    with pytest.raises(ValueError):
        normalize("kelime ابت", "turkish-v2")


# The turkish-v1 metric text is frozen too (published WER/CER), including the mistakes issue #5 lists.
METRIC_V1_PINS = {
    "i̇stanbul": "i stanbul",
    "kâr": "kâr",
    "II. Dünya": "ıı dünya",
    "21.yüzyıl": "yirmi bir yüzyıl",
    "3.30'a": "üç nokta otuza",
    "50 euro": "elli euro",
    "5 km": "beş km",
    "e-posta": "e posta",
    "COVID-19": "covıd on dokuz",
    "4'e": "dörte",
}


@pytest.mark.parametrize("raw, v1", METRIC_V1_PINS.items())
def test_metric_v1_outputs_are_pinned(raw, v1):
    assert metric_text_turkish(raw) == v1
    assert metric_text(raw, "turkish-v1") == v1


@pytest.mark.parametrize(
    "raw, v2",
    [
        # (1) combining marks are removed after lower-casing, not turned into word breaks
        ("i̇stanbul'a gittik", "istanbula gittik"),
        ("İSTANBUL", "istanbul"),
        # (2) circumflexes fold
        ("Kâr payı", "kar payı"),
        ("hâlâ", "hala"),
        ("millî takım", "milli takım"),
        ("mahkûm", "mahkum"),
        # (3) Roman numerals
        ("II. Dünya Savaşı", "ikinci dünya savaşı"),
        ("I. Dünya Savaşı", "birinci dünya savaşı"),
        ("XVI. yüzyıl", "on altıncı yüzyıl"),
        ("XVI.yüzyıl", "on altıncı yüzyıl"),
        ("XIX yüzyılda", "on dokuzuncu yüzyılda"),
        ("V. Murat", "beşinci murat"),
        ("Faz II başladı", "faz iki başladı"),
        ("Faz II'de", "faz ikide"),
        ("Bay X. geldi", "bay x geldi"),
        ("XL beden", "xl beden"),
        ("CD aldım", "cd aldım"),
        # (4) ordinals without a space
        ("21.yüzyıl", "yirmi birinci yüzyıl"),
        ("3.sınıf", "üçüncü sınıf"),
        # (5) clock times
        ("3.30'a kadar", "üç otuza kadar"),
        ("saat 3.30", "saat üç otuz"),
        ("Saat 14.00'te", "saat on dörtte"),
        ("14:00", "on dört"),
        ("saat 09.05", "saat dokuz sıfır beş"),
        ("3.5 puan", "üç nokta beş puan"),
        # (6) spelling variants
        ("50 euro", "elli avro"),
        ("€50", "elli avro"),
        ("50€", "elli avro"),
        ("herşey yolunda", "her şey yolunda"),
        ("kovid", "covid"),
        # (7) units and abbreviations
        ("5 TL", "beş lira"),
        ("5 TL'ye", "beş liraya"),
        ("5km'de", "beş kilometrede"),
        ("2 kg", "iki kilogram"),
        ("25°C", "yirmi beş derece"),
        ("T.C.", "te ce"),
        ("Koç A.Ş.", "koç anonim şirketi"),
        ("Yılmaz Ltd. Şti.", "yılmaz limited şirketi"),
        ("Dr. Ahmet", "doktor ahmet"),
        ("elma vb.", "elma ve benzeri"),
        # (8) intra-word hyphens join
        ("e-posta", "eposta"),
        ("Wi-Fi", "wifi"),
        # (9) Latin acronyms keep a dotted i; Turkish capitals keep ı
        ("COVID-19", "covid on dokuz"),
        ("UNICEF", "unicef"),
        ("WIFI", "wifi"),
        ("LinkedIn", "linkedin"),
        ("IRAK", "ırak"),
        ("ISPARTA'da", "ıspartada"),
        # numbers, apostrophes, words that must not change
        ("4'e", "dörde"),
        ("%4'ü", "yüzde dördü"),
        ("İsveç'ten 86 kişi!", "isveçten seksen altı kişi"),
        ("İsveç´ten", "isveçten"),
        ("‘Merhaba’ dedi", "merhaba dedi"),
        ("ok attı", "ok attı"),
    ],
)
def test_metric_v2(raw, v2):
    assert metric_text_turkish_v2(raw) == v2
    assert metric_text(raw, "turkish-v2") == v2


@pytest.mark.parametrize(
    "reference, hypothesis",
    [
        ("II. Dünya Savaşı 3.30'da bitti.", "İkinci Dünya Savaşı üç otuzda bitti."),
        ("COVID-19 salgını", "Covid-19 salgını"),
        ("Kâr 50 euro", "kar elli avro"),
        ("e-posta adresi", "eposta adresi"),
        ("21.yüzyıl", "21. yüzyıl"),
        ("i̇stanbul", "İstanbul"),
        ("T.C. vatandaşı", "te ce vatandaşı"),
        ("5 km yürüdük", "beş kilometre yürüdük"),
        ("Saat 4'e kadar", "saat dörde kadar"),
    ],
)
def test_metric_v2_scores_equivalent_spellings_as_equal(reference, hypothesis):
    assert error_counts(reference, hypothesis, "turkish-v2")["wer"] == 0
    assert error_counts(reference, hypothesis, "turkish-v1")["wer"] > 0


def test_metric_v2_equals_v1_on_plain_sentences():
    for sentence in FREYA_LIKE:
        assert metric_text_turkish_v2(sentence) == metric_text_turkish(sentence)
    freya = Path("/workspace/data/eval/freya_tr_eval.jsonl")
    if freya.exists():  # the full Freya-TR-Eval set: no digits, circumflexes, hyphens or acronyms
        rows = [json.loads(line)["text"] for line in freya.read_text().splitlines() if line.strip()]
        assert all(metric_text_turkish_v2(r) == metric_text_turkish(r) for r in rows)
