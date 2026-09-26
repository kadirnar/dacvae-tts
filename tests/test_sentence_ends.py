"""Sentence punctuation for synthesis: the tr-combined models read a chunk without a final period as noise."""

import pytest

from dacvae_tts.frontend import finish_sentence, restore_sentence_ends

PARAGRAPH = (
    "opus beş otomatik davranış denetimimizde bugüne kadarki tüm modeller arasında en iyi puanları elde etmektedir son "
    "modellere kıyasla geri döndürülmesi zor eylemlerde bulunma veya kendisine verilen sınırların dışına çıkma olasılığı "
    "çok daha düşüktür ve prompt enjeksiyonuna karşı opus beşten daha dirençlidir ayrıca uyum testlerimizi daha uzun "
    "görevleri imkansız görevleri ve gerçek olaylardan modellenen senaryoları kapsayacak şekilde genişlettik ancak yine "
    "de sınırları bulunmaktadır değerlerlendirmemizin tüm ayrıntıları opus beş sistem kartında mevcuttur"
)


def test_an_unpunctuated_paragraph_gets_its_sentence_ends():
    restored = restore_sentence_ends(PARAGRAPH)
    assert restored.count(".") == 4
    for end in ("etmektedir. Son", "dirençlidir. Ayrıca", "genişlettik. Ancak", "bulunmaktadır. Değerlerlendirmemizin"):
        assert end in restored
    assert "düşüktür ve prompt" in restored  # "ve" continues the sentence


def test_punctuated_and_short_text_is_unchanged():
    punctuated = "Bugün hava çok güzeldir. Parka gidiyoruz, orada arkadaşlarımızla buluşacağız ve top oynayacağız."
    assert restore_sentence_ends(punctuated) == punctuated
    assert restore_sentence_ends("Bu kısa bir metindir ve noktası yok") == "Bu kısa bir metindir ve noktası yok"


def test_future_and_progressive_predicates_and_turkish_capitals():
    text = ("bugün hava çok güzel olduğu için parka gidiyoruz iki saat sonra arkadaşlarımızla buluşacağız sonra birlikte top "
            "oynayacağız ikindi vakti eve döneceğiz")
    assert restore_sentence_ends(text) == (
        "bugün hava çok güzel olduğu için parka gidiyoruz. İki saat sonra arkadaşlarımızla buluşacağız. Sonra birlikte "
        "top oynayacağız. İkindi vakti eve döneceğiz")


@pytest.mark.parametrize("text, expected", [
    ("merhaba dünya", "merhaba dünya."),
    ("Merhaba dünya.", "Merhaba dünya."),
    ("Nasılsın?", "Nasılsın?"),
    ("bu bir test,", "bu bir test."),
    ("sonra da —", "sonra da."),
    ('Dedi ki "geliyorum"', 'Dedi ki "geliyorum."'),
    ("", ""),
])
def test_finish_sentence(text, expected):
    assert finish_sentence(text) == expected


def test_long_sentences_split_evenly_without_a_one_word_tail():
    from dacvae_tts.frontend import split_sentences

    sentence = ("Son modellere kıyasla geri döndürülmesi zor eylemlerde bulunma veya kendisine verilen sınırların dışına "
                "çıkma olasılığı çok daha düşüktür ve prompt enjeksiyonuna karşı opus beşten daha dirençlidir.")
    for limit in (185, 120, 90, 60):
        chunks = [chunk for chunk, _ in split_sentences(sentence, max_chars=limit, min_chars=0)]
        assert all(len(c) <= limit for c in chunks) and " ".join(chunks) == sentence
        assert min(len(c) for c in chunks) >= 30, chunks  # no "dirençlidir." left on its own
    # Conjunctions are preferred cut points when the sentence has no comma.
    assert [c for c, _ in split_sentences(sentence, max_chars=185, min_chars=0)][1].startswith("veya ")
