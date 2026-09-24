# Kod incelemesi, eğitimsiz kanıtlar ve yeni mimari/veri hattı seçenekleri (24 Eylül 2026, ikinci tur)

Bu belge `iyilestirme/inceleme-mimari-veri-hatti` dalını anlatıyor. Dal `main` (c1aeaf7) üzerine 42 commit ekliyor.
İçerik dört başlık altında:

1. Deponun tamamının kod incelemesi ve doğrulanan hataların düzeltmeleri.
2. GPU ve eğitim olmadan elde edilen kanıtlar: istatistik, normalizasyon, konuşmacı sızıntısı, yayımlanmış sonuçların
   yeniden analizi.
3. Yol haritasının "gelecek iş" listesinden dört yeni mimari/veri hattı seçeneği.
4. GPU'da koşulacak A/B'ler.

Kurallar:

- Yerelde model eğitimi, sentez/puanlama koşusu ya da veri seti hazırlığı yapılmadı.
- Her değişiklik eğitim içermeyen birim testleriyle doğrulandı (son koşu: 520 test geçti, `ruff` temiz).
- Bütün yeni seçenekler varsayılan olarak kapalı. Kapalıyken:
  - veri akışı bit düzeyinde aynı kalıyor (kayıtlı özet testleri),
  - model aynı kalıyor (parmak izi testleri),
  - eski checkpoint'ler aynen yükleniyor.

Önceki belgeler: [arastirma-raporu-2026-09-24.md](arastirma-raporu-2026-09-24.md) ve
[yol-haritasi.md](yol-haritasi.md).

---

## 1. Kod incelemesi: bulunan ve düzeltilen hatalar

Beş paralel inceleyici şu alanlara baktı: model/hedef fonksiyonu, eğitim/veri, çıkarım/süre, metin/değerlendirme,
GRPO/veri hattı. Her bulgu kodla ya da küçük bir betikle doğrulandı. Her düzeltmenin kendi commit'i ve regresyon
testi var.

| Alan | Hata | Etki |
|---|---|---|
| Çıkarım | `quality.py` float64'ü cihaz üzerinde hesaplıyordu | Apple Silicon'da (MPS) **her sentez** çöküyordu |
| Eğitim | `compile` geri dönüşü, DDP'de gradyan senkronunu kapatıyordu | Çok GPU'lu `compile: blocks` koşusunda ranklar ayrı modeller eğitiyordu |
| Eğitim | EMA ısınması ek EMA izlerini birbirinin aynısı yapıyordu; `--init-from` sıcak EMA'yı ~10 adımda siliyordu | Model-guidance ince ayarında ikinci EMA anlamsızdı; `train.ema_warmup: false` eklendi |
| Eğitim | Metin-hinge RNG'si checkpoint'e yazılmıyordu | Resume bit düzeyinde tekrar edilemiyordu |
| Eğitim | WSD soğuma cache'inin bölümü denetlenmiyordu | Val/test sesleri soğumada eğitime girebiliyordu |
| Eğitim | TLA-SA mantık başları Muon'daydı | Sıfır-init başlar ilk adımdan tam boy güncelleme alıyordu |
| Post-training | `distill`, EDM modellerinde ham çıkışı hıza regress ediyordu | **Tüm config'lerde distill yanlış fonksiyonu öğretiyordu** |
| Post-training | Veri kümesi `layout` almıyordu | Joined modeller (tüm Türkçe config'ler) eğitimde görmedikleri SEP/segment düzeniyle post-train ediliyordu |
| GRPO | Standartlaştırılmış avantaj, taban altı gürültüyü birim varyansa büyütüyordu | Yargıç gürültüsüne tam boy politika gradyanı veriliyordu |
| Veri | `make_drop_list.py` puanı eksik satırları tutuyordu | CER 0,6'lık bir satır, DNSMOS hatası yüzünden eğitimde kalabiliyordu |
| Veri | Lisans/kaynak bilgisi `prepare`'de kayboluyordu | Metadata'ya kaynak başına lisans özeti eklendi |
| Örnekleyici | `speaker_guidance`, metin ölçeği 1'de sessizce kapanıyordu | Demo kaydırıcısının en alt değerinde konuşmacı guidance'ı çalışmıyordu |
| Örnekleyici | `speaker_guidance` ile APG/rescale seçenekleri yok sayılıyordu | Artık hata veriyor |
| Çıkarım | Model-guidance checkpoint'lerinde CFG yeniden uygulanıyordu (`recommended_guidance` okunmuyordu) | 5,0 guidance fiilen ~16,7 oluyordu |
| Çıkarım | `synthesize_many`, süre başlı modellerde başı yok sayıyordu; tuple referansta ASR çöküyordu; Türkçe checkpoint'te ASR İngilizce'ydi | Üçü de düzeltildi |
| Metin | turkish-v2: `Prof.Dr.` yapışıyordu; `2'incisi` → `ikiincisi` oluyordu; saat/tarih/eksi yoktu; `5'de` → `beşde`; `9.-10.` → `dokuz.eksi onuncu` | Hepsi v2'de düzeltildi (v1 değişmedi) |
| Metrik | turkish-v2 tireli kelimeleri birleştiriyordu (`yazlık-kışlık`) | Doğru 16 transkripte 2'şer hata yazıyordu (§2.2) |
| Değerlendirme | Puanlayıcı kimliği satırlara yazılmıyordu; `--metric-normalization` yoktu; determinizm eksikti; prompt WAV'ları dosya adına güveniyordu; seed-tts noktalama farkı vardı; Freya metriği yoktu | Hepsi düzeltildi; `--freya-metric` eklendi |

Tam liste ve her birinin gerekçesi commit mesajlarında: `git log c1aeaf7..HEAD`.

---

## 2. Eğitimsiz kanıtlar

### 2.1 İstatistik (#4): eski güven aralıkları gereğinden dar

`scripts/simulate_interval_coverage.py`, bilinen bir gerçeğe karşı simülasyon yapıyor. Model, yayımlanmış Freya
sonuçlarına kalibre edildi: 495 cümlenin gerçek kelime sayıları, 10 konuşmacı, konuşmacı/cümle/üretim gürültüsü etkileri.
Her senaryo 500 kez tekrarlandı.

| Ölçüt (hedef %95 ya da %5) | Cümle bootstrap | Küme yüzdelik (eski varsayılan) | **Jackknife-t (yeni varsayılan)** |
|---|---|---|---|
| Tek sistem WER kapsaması | %64–86 | %87–89 | **%91–93** |
| Eşli fark kapsaması | %85–94 | %87–90 | **%93–96** |
| Gerçek fark 0 iken yanlış kazanç/kayıp | %6–7 (etkileşimde %16) | **%10–13** | **%4–6** |

Karar: `compare_evaluations`, `compare` ve `scripts/compare_evals.py` artık varsayılan olarak konuşmacı bazında
delete-one jackknife ve t(G−1) kullanıyor (MacKinnon, Nielsen & Webb, arXiv:2301.04527). `--interval percentile`
eski aralıkları bit düzeyinde aynen veriyor.

Yayımlanmış GPU sonuçları (run C, Freya s42 ve s1000) jackknife-t ile yeniden analiz edildi:

| Ayar | WER farkı [%95 GA] | Karar |
|---|---|---|
| best-of-3 | −2,25 [−3,18; −1,32] | kazanç |
| auto + best-of-3 (iki set birleşik, 990 cümle) | −2,61 [−3,85; −1,36] | kazanç |
| clamp (iki set birleşik) | −0,69 [−1,46; +0,08] | **berabere** (eski yöntem s1000'de "kazanç" diyordu) |
| predictor | −0,29 [−1,43; +0,84] | berabere; DNSMOS +0,07 kazanç |
| APG η 0,5 | +0,64 [+0,19; +1,09] | kayıp |
| CFG-rescale 0,7 | +0,84 [+0,26; +1,42] | kayıp |
| konuşmacı guidance 7 | +0,82 [+0,27; +1,36] | kayıp |
| süre ×1,3 | +5,22 [+1,40; +9,03] | kayıp |

Uyarı: best-of-N kazancında seçici (Whisper-turbo) ile yargıç (large-v3) aynı ailedendir. Bağımsız bir yargıçla
(ör. MMS-1b) doğrulanmadan bu sayı iyimser bir üst sınır olarak okunmalıdır.

### 2.2 Türkçe normalizasyon (#5)

**Eğitim metni.** turkish-v1 ve v2, `Vyvo/tr-dataset-12`'nin 42.591 transkriptine uygulandı. Rakam içeren 3.189 satırın
138'i değişiyor; eğitim etkisi küçük. Değişikliklerin hepsi düzeltme:

- `yetmiş dörtü` → `dördü`
- `altmışdan` → `altmıştan`
- `yetmişde` → `yetmişte`
- `sekiz nokta otuzda` → `sekiz otuzda`
- tarihler
- `vs.` → `vesaire`
- `A.Ş.` → `anonim şirketi`

Tek istisna olan `9.-10.` hatası da düzeltildi.

**Metrik.** Yayımlanmış 19 koşudaki (~9.400 cümle) Whisper hipotezleri v1 ve v2 ile yeniden puanlandı:

- v2, **24 cümlede daha az yanlış hata sayıyor, v1'den kötü olduğu cümle yok.**
- Korpus WER %4,464 → %4,431, CER %2,607 → %2,588.

### 2.3 Konuşmacı sızıntısı (#6): "görülmemiş" sesler büyük ölçüde görülmüş

**Meta veri analizi.** 22 held-out etiketin 21'inin programı, 13'ünün bölümü train'de var.

**Akustik analiz.** SIM-o modeliyle (WavLM-large ECAPA) konuşmacı merkezleri hesaplandı. Freya'nın 10 prompt
konuşmacısından **6'sının train'de aynı programdan 0,90–0,97 kosinüslü bir eşi var.** Karşılaştırma değerleri:

- aynı etiketin iki yarısı: medyan 0,936, 5. yüzdelik 0,876,
- başka programlardan etiketler: en çok 0,60.

Bir konuşmacı sınırda (0,81), yalnız 3'ü temiz. Yani yayımlanan "görülmemiş konuşmacı" SIM/WER sayıları kısmen
görülmüş seslerle ölçüldü.

**Araç.**

- `scripts/data/make_prompt_set.py`, Common Voice test konuşmacılarından sızıntısız bir prompt seti kuruyor. Bu
  konuşmacılar podcast verisinde hiç yok.
- `eval_sentences.py --prompt-set` bu seti kullanıyor. SIM-o'nun orijinal kaydı doğrudan prompt dosyası.
- 10 yerine örneğin 48 bağımsız ses, güven aralıklarını da daraltır.

### 2.4 Diğer ölçümler

**SIM ölçeği (#3).** Run C'nin 16 Freya çıktısında eski metrik `wavlm-base-plus-sv` 0,87–0,98 arasında doymuşken
SIM-o 0,37–0,83 arasında ayırt ediyor.

**Yerel yeniden üretim.** Freya'nın 24 prompt'u ham veriden birebir yeniden kuruldu (korelasyon 1,0000). Yerel temel
kolun sonuçları yayımlananla uyumlu:

| Ölçüt | Yerel | Yayımlanan |
|---|---|---|
| Eski SIM | 0,9466 | 0,9461 |
| DNSMOS | 2,896 | 2,889 |
| Kırpılan dosya | %96 | %96 |

**Süre kuralları (#12).** 22 held-out konuşmacıdan 1.722 gerçek cümle çiftinde gerçek süreyi tahmin hatası (log-MAE):

| Kural | log-MAE | Not |
|---|---|---|
| predictor | 0,126 | |
| auto | 0,132 | |
| clamp | 0,145 | |
| bayt kuralı | 0,154 | |
| hece | 0,159 | |
| **articulation** | 0,170 | %7 sistematik kısa |

Articulation kuralı gerçek süreyi en kötü tahmin eden kural. Sentezle WER ölçümü GPU'ya kaldı.

**Tanh öncesi kazanç (#13).** Aynı latentlerden, eşli karşılaştırma:

| Ölçüt | Fark |
|---|---|
| Kırpılan dosya oranı | %96 → %41 |
| DNSMOS | +0,055 [+0,034; +0,077] |
| WER/CER | berabere |
| UTMOS | −0,16 |

Kazanç sesi de kıstığı için (−13,7 → −15,6 LUFS) seviye-normalize karşılaştırma gerekiyordu. Bu adım, yerel koşular
kullanıcı isteğiyle durdurulduğu için yapılamadı. Seviye-normalize tekrar GPU/demo tarafında yapılmalı.

---

## 3. Yeni mimari ve veri hattı seçenekleri

Dördü de config ile açılıyor ve varsayılan olarak kapalı. Her biri için tek değişkenli bir A/B config'i var.
Tasarımlar, her biri için ayrı bir literatür ve kod taramasına dayanıyor.

| Seçenek | Config | Parametre | Dayanak |
|---|---|---|---|
| **Karakter birimleri** `model.text_units: chars` | `experiments/tr_w512_char_units.yaml` | 0 | Türkçe iki baytlı harfler (%12) LARoPE köşegeninde iki pay, iki CTC etiketi ve bayt kuralında fazla süre alıyor. FreyaTTS 92 sembollü karakter sözlüğü kullanıyor. Latin yazıda bayt ile karakteri karşılaştıran bir ablation yok. |
| **Prompt tempo pertürbasyonu** `train.tempo_prompt_prob` | `experiments/tr_w512_tempo_prompts.yaml` | 0 | VoiceStar: WER 6,42 → 5,66 (±%25, WSOLA). Model, prompt ile hedef arasındaki hız kırılmasını eğitimde görür. |
| **Dondurulmuş konuşmacı gömmesi** `model.speaker_condition_dim` | `experiments/tr_w512_speaker_condition.yaml` | +98 bin | Koel-TTS: SV vektörü 0,619, in-context 0,637. MiniMax: encoder + prompt 0,746, yalnız prompt 0,726; dondurulmuş SV WER'i artırdı. WER'e kapı olarak bakılmalı. |
| **Çok klipli konuşmacı bağlamı** `model.speaker_context: vector` | `experiments/tr_w512_speaker_context.yaml` | +2,24M (%3,4) | Irodori v4: tek klip 0,661 → 30 s 0,752 → 120 s 0,775. Kanıt yalnız büyük modellerden. Sıfır-init olduğu için run C'den sıcak başlatılabilir. |

**Karakter birimleri.**

- Kimlikler ISO-8859-9 (Latin-5) + 4. ASCII kimlikleri baytlarla aynı kalıyor; sözlük (260), embedding şekli ve
  boşluk/noktalama token'ları değişmiyor.
- turkish-v1/v2'nin geçirdiği 82 sembolün her biri tek ve ayrı bir birim.
- Hiçbir kimlik UTF-8 devam aralığına (0x80–0xBF) düşmüyor. Bu sayede karakter modeline bayt satırı verilirse model
  hata veriyor.
- Cache değişmiyor; dönüşüm yükleme anında yapılıyor.
- Çıkarımda prompt-hızı kuralı karakter başına kare tutuyor.

**Tempo.**

- `dacvae_tts.tempo`: WSOLA (40 ms pencere, 15 ms arama, float64, perde korunuyor) ve yan depo yazıcı/okuyucu/birleştirici.
- `scripts/build_tempo_variants.py`: satır × tempo (×0,8/0,9/1,0/1,111/1,25) latentlerini üretiyor. ×1,0 yeniden
  kodlanmış kontroldür.
- Veri kümesi: kesim aynı kalıyor; prompt `varyant[:round(kesim/t)]`, hedef değişmiyor.
- Statik ve epoch maliyet sınırları frame bütçesini koruyor.
- Gerilmiş prompt kareleri REPA'dan dışlanıyor.

**Konuşmacı gömmesi.**

- Sıfır-init ve biassız Linear, L2-normalize gömmeyi `voice`'a ekliyor. CFG null dalı, dropout ve prompt'suz dal gömmeyi
  kendiliğinden düşürüyor.
- Cümle-içi öğelerde gömme aynı etiketin **başka** bir cümlesinden alınıyor, böylece hedef sızmıyor.
- Checkpoint gömücüyü kaydediyor; çıkarımda aynı gömücü prompt'un codec yeniden yapılandırmasına uygulanıyor.
- TLA-SA ile aynı depo kullanılamıyor.

**Konuşmacı bağlamı.**

- 3 × 256, konumsuz küme transformer'ı, dikkat havuzlaması ve sıfır-init çıkış. Klip sırası sonucu değiştirmiyor.
- Öğelerin yarısına aynı etiketin başka cümlelerinden 3–30 s bağlam veriliyor.
- Çıkarımda bağlam, prompt ile `--context-audio` kliplerinden oluşuyor.

Ayrıca eğitimsiz yapılan iş:

- `eval_sentences.py --prompt-set` ve `scripts/data/make_prompt_set.py` (§2.3),
- jackknife-t aralıkları ve simülasyon betiği (§2.1),
- turkish-v2 düzeltmeleri (§2.2).

---

## 4. GPU'da yapılacaklar (öncelik sırasıyla)

1. **Ölçümü düzelt.** Freya'yı sızıntısız bir prompt setiyle yeniden ölç:
   - `make_prompt_set.py` ile CV-tr test'ten 48 konuşmacı,
   - `eval_sentences.py --prompt-set ... --protocol-v2 --sim-o --metric-normalization turkish-v2`,
   - kararlar jackknife-t ile (`compare_evals.py`).
2. **Best-of-N'i bağımsız doğrula.** Seçiciyi yargıçtan farklı aileden seç (ör. MMS-1b CER) ya da ikinci yargıç ekle.
   Demo varsayılanı (auto + best-of-3) buna bağlı.
3. **Çıkarım seçenekleri.** Tanh öncesi kazancı seviye-normalize olarak yeniden ölç. Articulation süre kuralını ve
   süre-çeşitli best-of-N'i (#12, #13) ölç.
4. **Yeni mimari A/B'leri.** 20k güncellemede, iki tohumlu baz koşuya karşı:
   - karakter birimleri,
   - tempo prompt'ları (önce depo üretimi),
   - konuşmacı gömmesi (önce `speakers --splits train,val`),
   - konuşmacı bağlamı (sıfırdan ya da run C'den `--init-from`).
5. **Eski issue'lar.** #7–#11 ve #14–#16'nın A/B'leri, [yol-haritasi.md](yol-haritasi.md)'deki protokolle.
