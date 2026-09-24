import json
from pathlib import Path

import pytest

from dacvae_tts.frontend import locative, prepare_text, speakable, spell, split_sentences


@pytest.mark.parametrize(
    "raw, spoken",
    [
        ("Toplantı 23.09.2026 tarihinde saat 14:30'da.", "Toplantı yirmi üç eylül iki bin yirmi altı tarihinde saat on dört otuzda."),
        ("Saat 09.30'da gel, ara 14:05'te, bitiş 18:00.", "Saat dokuz otuzda gel, ara on dört sıfır beşte, bitiş on sekiz."),
        ("Fiyat 1.250,50 TL, $199 ya da 180 €.", "Fiyat bin iki yüz elli virgül elli lira, yüz doksan dokuz dolar ya da yüz seksen avro."),
        ("Dolar 34,5 ₺ oldu. Fiyatı $199.", "Dolar otuz dört virgül beş lira oldu. Fiyatı yüz doksan dokuz dolar."),
        ("Hava 25°C, gece -3 derece.", "Hava yirmi beş derece, gece eksi üç derece."),
        ("Dr. Ahmet ve Prof. Ayşe geldi.", "Doktor Ahmet ve Profesör Ayşe geldi."),
        ("T.C. Sağlık Bakanlığı ve Koç Holding A.Ş. açıkladı.", "te ce Sağlık Bakanlığı ve Koç Holding anonim şirketi açıkladı."),
        ("Yılmaz Ltd. Şti. kuruldu.", "Yılmaz limited şirketi kuruldu."),
        ("elma, armut vs. Sonra geldi.", "elma, armut vesaire. Sonra geldi."),
        ("ABD'de ve TBMM'nin açıklaması, NATO'nun tepkisi.", "a be dede ve te be me menin açıklaması, natonun tepkisi."),
        ("Ali & Veli, 2+2=4.", "Ali ve Veli, iki artı iki eşittir dört."),
        ("Nüfusun 3/4 kadarı.", "Nüfusun dörtte üç kadarı."),
        ("5 km yürüdüm, 2 kg'lık paket, 120 km/sa hız.", "beş kilometre yürüdüm, iki kilogramlık paket, saatte yüz yirmi kilometre hız."),
        ("II. Dünya Savaşı ve XIX. yüzyıl.", "ikinci Dünya Savaşı ve on dokuzuncu yüzyıl."),
        ("BU ÇOK ÖNEMLİ BİR DUYURUDUR, LÜTFEN DİKKAT EDİN!", "Bu çok önemli bir duyurudur, lütfen dikkat edin!"),
        ("İSTANBUL'da 5G ve 4K.", "İstanbul'da beş ge ve dört ka."),
        ("%50 indirim, 20% kâr, sadece % oranı.", "yüzde elli indirim, yüzde yirmi kâr, sadece yüzde oranı."),
        ("Harika! 😊 #Türkiye @kadirnar", "Harika! Türkiye kadirnar"),
        ("«Merhaba» dedi, Café'de José ile.", '"Merhaba" dedi, Cafe\'de Jose ile.'),
        ("Ne dersin?! Harika!!! Bekle.....", "Ne dersin? Harika! Bekle..."),
        ("Bilgi: info@vyvo.ai", "Bilgi: info et vyvo nokta ai"),
    ],
)
def test_speakable_rewrites_symbols_dates_and_acronyms(raw, spoken):
    text, changes = speakable(raw)
    assert text == spoken
    assert all(isinstance(change, str) for change in changes)


def test_plain_turkish_is_unchanged():
    for sentence in ("Yarın öğleden sonra sağanak bekleniyormuş, şemsiyeni unutma.", "O ne yapıyor?", "Peki ya sonra?"):
        assert prepare_text(sentence).text == sentence
        assert prepare_text(sentence).changes == []
    freya = Path("/workspace/data/eval/freya_tr_eval.jsonl")
    if freya.exists():
        rows = [json.loads(line) for line in freya.read_text().splitlines() if line.strip()]
        assert all(prepare_text(r["text"]).text == r["text"] for r in rows)


def test_unspeakable_characters_are_removed_not_fatal():
    text, changes = speakable("Merhaba ابت dünya ☃")
    assert text == "Merhaba dünya"
    assert any("okunamayan" in c for c in changes)
    with pytest.raises(ValueError):
        speakable("😊😊")


def test_helpers():
    assert spell("CHP") == "ce ha pe"
    assert [locative(w) for w in ("iki", "üç", "dört", "altı", "on", "yüz")] == ["ikide", "üçte", "dörtte", "altıda", "onda", "yüzde"]


def test_split_sentences_merges_short_and_cuts_long():
    text = ("Kısa. Bir cümle daha. " + "Uzun bir cümle, virgüllerle bölünebilir, " * 8 + "ve biter.\nYeni satır.")
    chunks = split_sentences(text, max_chars=120, min_chars=30)
    assert all(len(c) <= 120 for c, _ in chunks)
    assert chunks[0][0].startswith("Kısa. Bir cümle daha.")
    assert chunks[-1][1] == "sentence"
    assert " ".join(c for c, _ in chunks).split() == text.split()
    assert split_sentences("   ") == []
    assert split_sentences("Tek cümle.") == [("Tek cümle.", "sentence")]


def test_suffix_harmony_list_ordinals_and_letter_names():
    from dacvae_tts.frontend import harmonize

    assert [harmonize("lira", "dir"), harmonize("dolar", "yi"), harmonize("avro", "dan"), harmonize("lira", "si")] == \
        ["dır", "ı", "dan", "sı"]
    text, _ = speakable("Kira 15.000 TL'dir, 10 USD'yi bozdurdum.")
    assert text == "Kira on beş bin liradır, on doları bozdurdum."
    assert speakable("1. Madde: Kira. 2. Madde: Depozito.")[0] == "birinci Madde: Kira. ikinci Madde: Depozito."
    assert speakable("Saat 7.30'da çıktım.")[0] == "Saat yedi otuzda çıktım."
    assert speakable("Q3 geliri, A4 kağıt.")[0] == "kü üç geliri, a dört kağıt."
    assert speakable("#YapayZeka günü")[0] == "Yapay Zeka günü"
