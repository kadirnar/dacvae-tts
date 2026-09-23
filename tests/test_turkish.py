import pytest

from dacvae_tts.metrics import error_counts, metric_text
from dacvae_tts.text import normalize, tokenize
from dacvae_tts.turkish import metric_text_turkish, normalize_turkish, number_words, ordinal_words, tr_lower


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
