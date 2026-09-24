# DACVAE-TTS Türkçe yol haritası (24 Eylül 2026)

Gerekçeler ve kanıtlar: [arastirma-raporu-2026-09-24.md](arastirma-raporu-2026-09-24.md). Takip issue'su:
[#17](https://github.com/kadirnar/dacvae-tts/issues/17). Her iş bir GitHub issue'su ve
aynı adlı bir branch'tir; kod değişiklikleri **konfigürasyonla açılır, varsayılan kapalıdır** (mevcut checkpoint'ler ve
konfigler aynen çalışır). Issue'lar deney kanıtı gelene kadar açık kalır: kod branch'te hazır, A/B GPU'da koşulur.

**Öncelik ilkesi:** önce ölçüm (yanlış metrikle verilen kararlar geri alınamaz), sonra kaliteyi değiştirmeyen hız
kazanımları (her sonraki deneyi ucuzlatır), sonra A/B'ler, en son veri büyütme ve post-training.

## Faz 0 — Ölçüm ve temizlik

| # | İş | Branch | Hedef metrik |
|---|---|---|---|
| [#2](https://github.com/kadirnar/dacvae-tts/issues/2) | Doküman temizliği, araştırma raporu, yol haritası | `docs/temizlik-arastirma-raporu` | — |
| [#3](https://github.com/kadirnar/dacvae-tts/issues/3) | Değerlendirme protokolü v2: SIM-o, UTMOS, deterministik Whisper, kırpılma/bant genişliği | `eval/protokol-v2` | SIM, kalite, WER güvenilirliği |
| [#4](https://github.com/kadirnar/dacvae-tts/issues/4) | Konuşmacı-kümelenmiş bootstrap ve eşli karşılaştırma | `eval/istatistik` | Karar güvenilirliği |
| [#5](https://github.com/kadirnar/dacvae-tts/issues/5) | Türkçe normalizasyon v2 (ünsüz yumuşaması, kısaltmalar, metrik hataları) | `text/turkish-v2` | WER/CER |
| [#6](https://github.com/kadirnar/dacvae-tts/issues/6) | Konuşmacı sızıntısı ve program düzeyinde ayrım | `data/konusmaci-sizinti` | SIM/WER güvenilirliği |

## Faz 1 — Hız ve ucuz kazanımlar

| # | İş | Branch | Hedef metrik |
|---|---|---|---|
| [#7](https://github.com/kadirnar/dacvae-tts/issues/7) | Eğitim hızı: senkronizasyonlar, blok bazlı compile, dolgu, seçici checkpointing | `perf/egitim-hizi` | Eğitim süresi ×1,8–2,5 |
| [#8](https://github.com/kadirnar/dacvae-tts/issues/8) | Latent tekrar/atlama negatifleri (ek ileri geçişsiz) | `objective/latent-negatifler` | Adım süresi −%20, WER/CER |
| [#12](https://github.com/kadirnar/dacvae-tts/issues/12) | Süre: artikülasyon kuralı, süre-çeşitli yeniden sıralama | `duration/artikulasyon-kurali` | WER/CER (kısa cümleler) |
| [#13](https://github.com/kadirnar/dacvae-tts/issues/13) | Örnekleyici/çıktı kalitesi: pencere başına guidance, tanh öncesi kazanç, bileşik seçici | `inference/ornekleyici-kalite` | DNSMOS/UTMOS, kırpılma |

## Faz 2 — Mimari ve hedef A/B'leri

| # | İş | Branch | Hedef metrik |
|---|---|---|---|
| [#9](https://github.com/kadirnar/dacvae-tts/issues/9) | DiT blok seçenekleri: uzun skip, value residual, conv-FFN, attention kapısı, SwiGLU, final adaLN | `arch/dit-blok-secenekleri` | WER/CER, kalite |
| [#10](https://github.com/kadirnar/dacvae-tts/issues/10) | Speech-REPA + TLA-SA yardımcı kayıpları | `objective/ogretmen-hizalama` | WER, SIM, yakınsama hızı |
| [#11](https://github.com/kadirnar/dacvae-tts/issues/11) | Eğitim çiftleri: farklı cümle prompt'u, kısa hedef, kuyruk sessizliği, karakter CTC | `data/egitim-ciftleri` | WER/CER, SIM, süre dayanıklılığı |
| [#14](https://github.com/kadirnar/dacvae-tts/issues/14) | Çizelge/düzenlileştirme: WSD, uniform-t soğuma, dropout, çift EMA, model-guidance | `train/cizelge-duzenlilestirme` | Aşırı uyum, kalite, CFG'siz çıkarım |

## Faz 3 — Veri ve post-training

| # | İş | Branch | Hedef metrik |
|---|---|---|---|
| [#15](https://github.com/kadirnar/dacvae-tts/issues/15) | Türkçe veri genişletme hattı (YODAS tr000, Common Voice tr, ISSAI TSC) | `data/turkce-veri-hatti` | SIM, genelleme |
| [#16](https://github.com/kadirnar/dacvae-tts/issues/16) | Bileşik ödüllü Flow-GRPO | `posttrain/grpo` | DNSMOS/UTMOS, WER, SIM |

## Birleştirme sırası

Branch'ler `main`'den ayrı ayrı açıldı ve birbirinden bağımsız incelenebilir. Birden fazlası `config.py`, `model.py`,
`training.py`, `data.py` gibi ortak dosyalara **eklemeler** yapıyor; önerilen birleştirme sırası küçük çakışmaları en aza
indirir: #2 → #5 → #3 → #4 → #6 → #7 → #8 → #13 → #12 → #9 → #11 → #10 → #14 → #15 → #16. Her birleştirmeden sonra kalan
branch'ler `main` üzerine rebase edilir; çakışmalar çoğunlukla yan yana eklenen konfig alanlarıdır.

## A/B protokolü (tüm deneyler)

- Baz: `configs/nano_tr_w512.yaml`, temiz cache, 20k güncelleme (~3,2 saat, bir 4090; #7 sonrası ~1,5–2 saat).
- **Baz iki tohumla** koşulur (gürültü tabanı); her kolda tek değişken.
- Değerlendirme: protokol v2 (#3) — Freya-TR-Eval (register/uzunluk kovaları + 8 kHz sütunu), MiniMax-Türkçe, iç podcast
  seti; CER birincil, WER, S/D/I, SIM-o + ikinci SV modeli, DNSMOS/UTMOSv2, kırpılma, adım süresi.
- Karar: konuşmacı-kümelenmiş eşli bootstrap (#4); GA 0'ı içeriyorsa "berabere". CER kazancı tohum yayılımından büyük
  olan değişiklikler birleştirilip 60k'lık tam koşuda doğrulanır.

## Hedefler

| Metrik | Şimdi (C-60k) | Faz 1–2 hedefi |
|---|---|---|
| Freya CER / WER (tek örnek, kural süre) | %2,5 / %4,3 | ≤ %1,8 / ≤ %3,3 |
| SIM-o (WavLM-large ECAPA) | ölçülecek (#3) | gerçek-ses tavanının ≥ %90'ı |
| DNSMOS OVRL / UTMOSv2 | 2,89 / ölçülecek | ≥ 3,05 / baz + 0,2 |
| 60k güncelleme süresi (tek 4090) | 9,5 saat | ≤ 5 saat |
| Parametre (çıkarım) | 66,5M | ≤ 68M |

## Gelecek iş (issue açılmadı)

CFG-kaynaşık MeanFlow/IntMeanFlow damıtmasıyla 4 adımlı çıkarım; donmuş CAM++ embedding → adaLN ve uzun (≥20 s) çok
klipli referanslar için ayrı referans kodlayıcı; yalnız prompt'a tempo pertürbasyonu (yeniden kodlama gerektirir); Sidon
restorasyonlu tam bant veri; 16 × 448 derinlik/genişlik A/B'si; karakter sözlüklü metin girdisi; latent-uzayı konuşmacı
doğrulayıcısı ile SV kaybı.
