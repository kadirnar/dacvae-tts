# Türkçe DACVAE-TTS: araştırma özeti ve tarif kararları (22 Eylül 2026)

**Amaç:** `Vyvo/tr-dataset-12` (42.591 podcast segmenti, 93,2 saat, 44,1/48 kHz MP3, kalite puanı 50–100) ile
2×RTX 4090 üzerinde sıfırdan Türkçe zero-shot ses klonlama TTS eğitmek; ses kalitesi yüksek, WER düşük olmalı.
Eğitimden önce üç paralel literatür/kod taraması yapıldı (Echo-TTS; Irodori-TTS + Darya-TTS; 2024–2026 flow/diffusion
TTS literatürü). Ham raporlar Ek A–C'de; bu bölüm yalnız **bizim tarife etkisi olan** bulguları toplar.

## 1. Veri setinin gerçek yüzü (yerel ölçüm)

| Ölçüm | Değer |
|---|---|
| Satır / süre | 42.591 / 93,2 saat; ortalama 7,5 s, medyan 6,7 s, en uzun ≈ 19,6 s (15 s üstü %3) |
| Ses | MP3, mono, 44,1 kHz (bir kısmı 48 kHz); −50 dB bant genişliği çoğunlukla 12–16 kHz (README'deki "16 kHz" yanlış) |
| Kalite puanı | medyan 69; <55 %13, <60 %30, <70 %50 |
| Konuşmacı etiketi | bölüm-içi diarizasyon (`<episode>_speaker_k`); aynı kişi farklı bölümlerde farklı etiket alabilir |
| Metin | %10 satırda rakam var (yıl, yüzde, sürüm numarası, ek almış sayılar: "2010'da", "%99'u", "12. nesil"); 90 satırda Arapça/Kiril/özel karakter |

Karar: rakamlı satırlar atılmadı, **Türkçe sayı okunuşuna çevrildi** (`src/dacvae_tts/turkish.py`, `--text-normalization turkish-v1`):
yüzde, saat, binlik ayraç, ondalık ("virgül"/"nokta"), aralık, sıralı ("12." → "on ikinci"), kesme işaretli ekler
("2010'da" → "iki bin onda"), harfe yapışık sayılar ("350D" → "üç yüz elli D"). Kalan rakam veya Latin dışı karakter → satır
reddedilir (42.591 satırda 90 ret). FreyaTTS'nin Türkçe bulgusu bunu doğruluyor: rakam dizileri okunuşa çevrilmezse süre
tahmini ve WER bozuluyor.

WER ölçümü için `turkish-v1` metrik normalizasyonu: sayılar her iki tarafta okunuşa çevrilir, **İ→i, I→ı** (Python `lower()`
Türkçe için yanlış), kesme işareti silinir, noktalama atılır. ASR: faster-whisper **large-v3**, `language=tr`.

## 2. Üç kaynaktan tarife giren maddeler

| Bulgu | Kaynak | Bizim tarif |
|---|---|---|
| Byte-düzeyi metin dil bağımsız çalışır; noktalama stilini eğitim/inference'ta tutarlı tut | Echo, Irodori, DiTTo (ByT5) | UTF-8 byte + `turkish-v1` normalizasyonu her iki tarafta |
| Cross-attention + **LARoPE**; 25 fps latentte F5-tarzı doldurma-birleştirme riskli (byte/s ≈ frame/s) | Literatür (LARoPE, ZipVoice, Freya, Supertonic) | `positions: rope` (LARoPE cross-attn) korundu |
| **CTC yardımcı kaybı** küçük veride en güçlü WER kaldıracı (A-DMA: 0,6k saat, WER 2,68→1,97) | A-DMA, ARCHI-TTS | `ctc_layer: 8/12, ctc_weight: 0.1` korundu |
| Stratified logit-normal *t*, hız-MSE | Echo, Irodori, Darya, LAION | korundu (EDM ön-koşullandırma ile) |
| Context-sharing batch expansion (Ke=4) hizalamayı hızlandırır | Supertonic | `batch_expansion: 2` (bellek), tam koşuda 3–4 denenecek |
| Skip/repeat kontrastif negatifler (Seed WER 1,44→1,38) | RobustSpeechFlow | `contrastive_weight: 0.2` korundu |
| Bağımsız metin/konuşmacı guidance; **CFG yalnız gürültülü yarıda** (`cfg_min_t=0.5`) → ~2× NFE tasarrufu | Echo, Irodori, ZipVoice, "unified guidance" | Örnekleyiciye zamana bağlı CFG eklendi (§4) |
| Başlangıç gürültüsünü 0,8–0,9 ile ölçekleme, 30–60 adım, sway sampling | Echo, F5 | Sway (s=−1) mevcut; adım/guidance/ölçek taraması pilotta |
| Süre: prompt konuşma hızı kuralı GT süreye yakın; ayrı predictor gövde dondurulduktan **sonra** eğitilmeli | F5 Tablo 4, Irodori v4.1, DMOSpeech 2 | `duration: rule`; `duration_scale` taraması |
| Loudness −16 LUFS encode öncesi ve inference referansında | Irodori, DACVAE API | `--loudness -16` |
| Düşük-rank AdaLN + QK-norm + Muon (küçük ablation'larda Adam'dan iyi) | Echo, Irodori | korundu |
| Veri filtreleme: DNSMOS ≥ 2,8–3,0; ASR ile yeniden transkript, CER kuyruğunun %15'ini kes (%50 kesmek zarar) | Emilia, Raon-OpenTTS | Aşama 2: tüm korpus large-v3 ile transkript edilip CER kuyruğu kesilecek |
| 128-boyutlu latent sabit kapasitede TTS'i zorlar; 500M "tüketici ölçeği" tatlı nokta | LongCat, Irodori, Darya | 51M nano ile başla (93 saat → aşırı uyum riski), val/WER'e göre genişlik 512–640 varyantı |
| Türkçe için Whisper-large-v3 ile ölç, CER'i birincil raporla (eklemeli dil WER'i şişirir); Freya-TR-Eval: FreyaTTS 183M WER 8,0 / CER 3,0, XTTS-v2 11,1, F5-TTS 24,3 | FreyaTTS, OmniVoice | Hedef: 50M modelde CER %3–5 iyi sonuç sayılır |

## 3. Deney planı

1. **Pilot** (4 shard ≈ 21 saat, ~12k güncelleme, tek GPU): hattı uçtan uca doğrula (normalizasyon → cache → eğitim →
   large-v3 WER). Aynı anda ikinci GPU'da tüm korpusun large-v3 transkripti ve codec/ASR tavanı.
2. **Tam koşu** (17 shard, kalite ≥ 55 ≈ 80 saat, 2 GPU DDP, 60k güncelleme): `configs/nano_tr.yaml`; her 5k'da kalıcı
   snapshot + monitor (48 görülmemiş konuşmacı vakası). Val flow + metin kazancı ile aşırı uyum takibi.
3. **İyileştirme turu**: CER-filtreli / yüksek kaliteli alt küme ile ince ayar; guidance/adım/sway/ölçek taraması;
   gerekirse genişlik 512 varyantı.
4. Her eğitim sonunda sonuçlar (config, eğriler, monitor sesleri, checkpoint) `VoiceHub/*` HF veri seti olarak yayımlanır.

## 4. Kod değişiklikleri (bu tarih)

- `src/dacvae_tts/turkish.py`: Türkçe sayı okunuşu, betik denetimi, Türkçe küçük harf, WER metrik normalizasyonu.
- `text.py`/`metrics.py`/`cli.py`/`scripts/monitor.py`: `turkish-v1` sürümleri; `--language tr` için varsayılan metrik.
- `scripts/prepare_hf_shards.py`: `--local-dir`, `--text-normalization`, `--languages`; yerel shard'lar için
  indirme düzeniyle aynı satır kimlikleri (symlink) — aksi halde merge'de kimlik çakışması.
- `scripts/prepare_local_2gpu.sh`, `scripts/train_2gpu.sh`, `scripts/push_results.py`, `scripts/eval_ceiling.py`,
  `scripts/eval_sentences.py` (Freya-TR-Eval vb. sabit cümle setleri), `scripts/transcribe_corpus.py` (tüm korpus için
  Whisper-large-v3 CER + DNSMOS; `merge --drop-uids` ile filtre), `configs/nano_tr.yaml`, `tests/test_turkish.py`.
- Örnekleyici: `guidance_until` (CFG yalnız t < eşik, Echo/Irodori'nin `cfg_min_t`'si) ve `noise_scale` (başlangıç
  gürültüsü kısaltma) seçenekleri; `monitor.py` aynı checkpoint için birden çok örnekleyici ayarını puanlayabiliyor.
- Yerel gözlemler: `compile: model` bu makinede ilk adımda 15 dk+ derleme (dinamik şekil yeniden derlemeleri) → kapatıldı;
  eager + aktivasyon checkpointing, frame bütçesi 12000'de tepe 17,2 GB, 0,6 s/güncelleme (tek 4090). DNSMOS/decoder
  işçi havuzları CPU'yu doyurunca eğitim 3× yavaşladı → onnxruntime oturumları tek iş parçacığına sabitlendi.


---

## Ek A — Echo-TTS (Jordan Darefsky, Kasım/Aralık 2025) — ajan raporu özeti

Kaynaklar: blog https://jordandarefsky.com/blog/2025/echo/ ; repo https://github.com/jordandare/echo-tts (yalnız inference);
ağırlıklar https://huggingface.co/jordand/echo-tts-base ; eğitim kodu içeren LAION yeniden uygulamaları:
https://github.com/LAION-AI/jax-dacvae-echotts (DACVAE latentleri) ve https://github.com/LAION-AI/scaled-echo-tts.

- **Mimari:** 2,4B DiT (24 katman × 2048, 16 head), 14 katmanlı byte text encoder (vocab 256), klonlama için temiz referans
  latentleri üzerinde ayrı nedensel speaker encoder (×4 patch, girişte /6), referans transkripti gerekmiyor. Joint self+cross
  attention (self/text/speaker K,V tek softmax'ta), QK-norm, sigmoid kapılı attention, RoPE yalnız self-attention head'lerinin
  yarısında, cross anahtarlarında RoPE yok; düşük-rank (256) AdaLN + tanh kapı, yalnız zaman koşulu; SwiGLU; çıkış zero-init (LAION).
- **Latent:** Fish S1-DAC 44,1 kHz, 21,5 fps; 1024-d dequantize → PCA 80 boyut × skaler 1/18. Kanal-bazlı standardizasyon
  zarar verdi, tek global skaler işe yaradı. LAION: DACVAE 48 kHz/128 ch/25 fps ham latent, dolgu = encode edilmiş sessizlik.
- **Hedef:** rectified flow, v = gürültü − x0, stratified logit-normal *t*, düz MSE (dolgu dahil), bağımsız %10/%10 metin ve
  konuşmacı dropout. CTC, süre predictor'ı, yardımcı kayıp yok.
- **Süre:** sabit 640 frame (~30 s) tuval; sıfır dolgu hedefe dahil; inference'ta kuyruktaki düz bölge kırpılır
  (pencere 20, std<0,05, |ort|<0,1). Uzun metin 30 s'ye sıkıştırılır (hızlı konuşma).
- **Eğitim:** ~160k saat podcast, WhisperD transkriptleri ([S1]/[S2], "uh/um", "(laughs)"); Muon (küçük ablation'larda Adam'ı
  geçti), batch 768, 800k adım, WSD çizelgesi, bf16 hesap / fp32 master. Tepe LR/EMA belirtilmemiş. LAION DACVAE tarifi:
  AdamW 1e-4, betas (0,9, 0,99), wd 0,01 (bias/norm/gate/out_proj hariç), clip 1,0, %5 ısınma + cosine, global batch 256.
- **Örnekleme:** Euler 30–60 adım; bağımsız guidance metin 3 / konuşmacı 5–8, yalnız gürültülü yarıda (cfg_min_t=0,5) →
  ~2× NFE; başlangıç gürültüsü 0,8–0,9 ile ölçekleme; temporal score rescaling (k=1,2, σ=3); OOD metinde konuşmacı
  kaymasına karşı speaker K/V ölçekleme 1,1–1,5 (t≥0,9).
- **Kalite:** WER/SIM/MOS rapor edilmemiş. Hata modları: OOD metinde referansı yok sayma, 30 adımda artefakt (60 çözüyor),
  kısa prompt'ta boşluklar, uzun metnin 30 s'ye sıkışması.
- **Bize aktarılan:** byte metin dil bağımsız; `: ; —` → virgül gibi tutarlı noktalama; bağımsız metin/konuşmacı guidance ve
  yalnız yüksek gürültüde CFG; gürültü kısaltma; 30+ adım; düşük-rank AdaLN + QK-norm; Muon. Hesap gerçeği: Echo ~614M örnek
  gördü, biz birkaç milyon → "hizalama hilesi gerekmez" iddiası bizim ölçekte kanıtlanmış değil (CTC/LARoPE korunmalı).

## Ek B — Irodori-TTS (Aratako) ve Darya-TTS (Respair) — ajan raporu özeti

**Irodori-TTS** — https://github.com/Aratako/Irodori-TTS ; kartlar: Irodori-TTS-v4.1-Small, v4-Small, 500M-v3, 600M-v3-VoiceDesign,
500M-v2, 500M; codec: https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim . Japonca, Echo-tarzı joint-attention RF-DiT, MIT.
- 500M ailesi: DiT 12 × 1280, 20 head, SwiGLU 2,875×, adaln_rank 192; sıfırdan 10 × 512 text encoder (LLM BPE token'ları);
  8 × 768 referans encoder. v4 (766M): text encoder yerine ince ayarlı ModernBERT-ja-310m (LR 1e-5), speaker_patch 4,
  referans 1–120 s rastgele birleştirme (SIM: tek klip 0,661 → 30 s 0,752 → 120 s 0,775).
- Latent: v1 `facebook/dacvae-watermarked` (48 kHz, 25 fps, 128-d); v2+ Japonca WavLM-distile 32-boyutlu Semantic-DACVAE
  (yazar daha hızlı downstream eğitimi buna bağlıyor). Latent normalizasyonu yok; encode öncesi −16 dB loudness.
- Hedef: RF (t=0 veri), stratified logit-normal, MSE; v1/v2 sabit 750 frame sıfır dolgulu hedef, v3+ değişken uzunluk +
  utterance-ortalaması; bağımsız 0,1/0,1/0,1 dropout; süre Huber(log1p frame). CTC yok. MeanFlow distillation (4 adım).
- Süre: v3+ token-toplamlı DurationPredictor; **v4.1'de yalnız predictor, gövde dondurulmuş, 200k adımda yeniden eğitildi**
  (birlikte eğitim süreyi fazla tahmin ediyordu) → Kana-CER 7,43 → 7,29, standart CER 5,35 → 4,69.
- Tarif: batch 80/GPU (v4: 40×2), Muon 1e-4 (match_rms_adamw) + AdamW yardımcı, wd 0,01, WSD (1k ısınma), 30–50k adım,
  bf16/fp32 master, clip 1, EMA yok; uzunluk kovalı örnekleyici. Saat/GPU sayısı açıklanmamış.
- Örnekleme: Euler 40 adım; CFG metin 3 / caption 3 / konuşmacı 5, `independent`, **yalnız t∈[0,5, 1]**; sway ile 6 adım; gürültü
  kısaltma 0,8–0,9; speaker K/V ölçekleme. Kalite: JSUT Kana-CER 3,43 (v4.1), JVS CAM++ SIM 0,661–0,775.

**Darya-TTS** — https://github.com/Respaired/Darya_TTS ; https://huggingface.co/Respair/Darya_TTS ; codec https://huggingface.co/Respair/dune_codec .
Farsça+Tacikçe/Rusça/İngilizce, 1B RF Enc/Dec DiT, OpenRAIL++-M.
- Text encoder: sıfırdan ModernBERT-config 12 × 1024, özel BPE 3.333; DiT 20 × 1280, düşük-rank AdaLN 256, QK-norm, yarım RoPE,
  kapılı attention, joint attention. **Prompt aynı dizide infilling ile** (VoiceBox/E2/F5 tarzı): span mask U(0,7, 1,0), kayıp yalnız
  maskeli karelerde; yazar: "prompt benzerliği için elimizdeki en iyi şey". Opsiyonel TitaNet FiLM stil vektörü.
- Latent: NVIDIA NanoCodec encoder'ı (22,05 kHz, **12,5 fps**, FSQ 52-d ön-kuantizasyon latenti, normalize edilmemiş), Dune
  decoder 44,1 kHz; decode'da FSQ'ya geri kuantize (küçük regresyon hatalarına bağışık). 30 s = 376 frame.
- Hedef: RF (t=1 veri), stratified logit-normal, %10 metin+konuşmacı, %10 yalnız metin dropout. CTC yok; discriminator hiç açılmadı.
- Süre: ayrı SpeechLengthPredictor (metin encoder + nedensel decoder, 378 sınıflı); yazar: "iyi bir süre predictor'ı her şeyi etkiler".
- Tarif: ~51,5k saat, batch 112 × 4 birikim, bf16 + torchao float8 + compile, AdamW8bit 5e-5, wd 0, cosine, 100k adım, clip 1,
  EMA 0,9999; 80k adımda yayımlandı. Sıfırdan tüketici GPU için ~500M öneriyor.
- Örnekleme: Euler 32 / midpoint 16, CFG 2–3, APG opsiyonu, son %20 adımda hız yeniden kullanımı ("reducio"). WER/SIM/MOS yok.

**Aktarılanlar:** latent seçimi en büyük kaldıraç (25 Hz/32-d veya 12,5 Hz/52-d; 128-d DACVAE en zoru); ~500M tüketici tatlı
noktası; iki klonlama tarifi (ayrı referans encoder / span-mask infilling — bizimki ikincisine denk); süre predictor'ını gövde
donduktan sonra eğit; Muon 1e-4 + WSD veya AdamW8bit + EMA; Türkçe için karakter/byte yeterli; CFG yalnız gürültülü yarıda;
−16 dB loudness; 30 s üst sınır.

## Ek C — 2024–2026 NAR flow/diffusion TTS literatürü — ajan raporu özeti

- **FreyaTTS (Türkçe, 2026, https://arxiv.org/html/2607.09530):** 183M DiT (16 × 640) donmuş AudioVAE2 latenti (64-d, 25 Hz,
  48 kHz), 92 sembollü Türkçe karakter sözlüğü, 4 ConvNeXt text bloğu, cross-attention, log-süre başlığı; doğrusal flow matching,
  AdamW 1e-4, 150k adım, batch 64, **CFG yok**, 32 Euler. Freya-TR-Eval (495 cümle, Whisper-large-v3): WER 8,0 / CER 3,0;
  XTTS-v2 11,1; F5-TTS 24,3; Piper 4,4; MMS-TTS 6,8; MOS 3,68. Dersler: rakamlar okunuşa çevrilmeli; uzun cümlede kayma → cümle parçalama.
- **SupertonicTTS (44M, 945 saat):** 24-ch latent ~14 Hz, kanal-bazlı standardizasyon, karakter girdi, cross-attention, 0,5M süre
  predictor'ı, L1, uniform t, CFG dropout %5, NFE 32; **context-sharing batch expansion Ke=4** hizalamayı büyük batch'ten çok
  hızlandırıyor. LibriSpeech-PC WER 2,41.
- **LongCat-AudioDiT:** 64/128/256-d latent ablation'ı — sabit kapasitede yüksek latent boyutu TTS'e zarar; APG (η 0,5, β −0,3)
  CFG 4,0'ın yerine; prompt latentleri her adımda GT ile üzerine yazılır.
- **DiTTo-TTS:** ByT5 byte encoder; S 42M WER 3,07 / B 152M 2,74 / XL 740M 2,56; uzunluk predictor'ı sabit uzunluğa karşı 5,58 vs 8,89.
- **ZipVoice (123M):** average upsampling kaldırılınca WER 1,69 → 20,19; text encoder kaldırılınca → 2,04; zamana bağlı CFG
  (erken adımlarda yalnız metin düşür); 8 NFE'de 1,69 / SIM 0,610 / UTMOS 4,16.
- **F5-TTS:** uniform t, span mask %70–100, audio-cond drop 0,3 / ikisi 0,2, sway s=−1, CFG 2,0, 16–32 NFE; süre = karakter oranı
  kuralı (GT süre pek eklemiyor). 155M / 945 saat: F5 WER 4,17 vs E2 9,63 (E2 %7 felaket örnek). v1: text_mask_padding, tüm
  head'lerde RoPE. EMA erken ince ayar checkpoint'lerinde zararlı olabilir. F5R-TTS (GRPO) −%29,5 WER. Türkçe F5 ince ayarları
  (Karayakar, marduk-ra; CV17) WER raporlamıyor.
- **E2, Voicebox, NaturalSpeech 3, MaskGCT, Seed-TTS DiT, CosyVoice 2/3, IndexTTS-2, MegaTTS 3, Kokoro, Chatterbox (Türkçe
  içeriyor), Zonos, Dia, Mimi:** özet Ek C'nin ham raporunda; öne çıkan: MaskGCT'de G2P, Whisper-BPE'den iyi (SIM 0,728 vs 0,711);
  MegaTTS 3 seyrek hizalama (WER 1,82 vs hizalamasız 2,14); CosyVoice L1 + cosine t + CFG 0,7 / NFE 10.
- **Konu makaleleri:** RobustSpeechFlow (repeat/skip sert negatifler; Seed 1,44 → 1,38, CER 0,48 → 0,35); A-DMA (CTC katman ≈ 2/3
  derinlik, λ 0,1; + HuBERT kosinüs; 0,6k saatte WER 2,68 → 1,97, yakınsama 2×); ARCHI-TTS (289M, 12,5 Hz, CTC η 0,1, 8×5090'da 4 gün,
  LS-PC WER 1,98); LARoPE; APG; "unified guidance": metin hizalaması guidance'ı erken-orta adımlarda, konuşmacı benzerliği geç
  adımlarda ister; SR-FD (4 adımda WER 2,23 → 1,41); Target-KL VAE (düşük bit hızı → düşük WER, tekdüze prozodi).
- **Türkçe kaynaklar:** Whisper-large-v2 FLEURS-tr WER ≈ %8,4, v3 %10–20 daha az hata; topluluk turbo ince ayarları WER 15–19;
  Common Voice tr ≈ 130 saat (ort. 2,6 s), FLEURS-tr 12 saat, YODAS'ta Türkçe ilk 15'te; Emilia'da Türkçe yok; Freya-TR-Eval tek
  açık TTS benchmark'ı; normalizasyon: trnorm (https://github.com/ysdede/trnorm), num2words tr (bitişik yazıyor → kendi uygulamamız).
- **Öneriler (ajan):** cross-attention + LARoPE; CTC λ 0,1 orta-geç katmanda; ≥4 bloklu text encoder; Türkçe küçük harf; CFG
  dropout %10–20 metin / %10–30 ses; ölçek 2–3, zamana bağlı CFG, ≥4 ise APG; sway + 16–32 NFE; karakter oranı süre kuralı
  + hafif süre başlığı; Whisper-large-v3 ile yeniden transkript, CER ≤ ~%5 filtre, DNSMOS OVRL ≥ 3,0 (2,8), 3–20 s; batch expansion
  Ke=4; RobustSpeechFlow negatifleri; kanal-bazlı latent standardizasyonu (güvenli varsayılan); EMA 0,999–0,9999; CER'i birincil
  raporla; hedef: 50M modelde Türkçe CER %3–5.

## 5. Korpus yeniden transkripsiyonu (22 Eylül, 15:20)

Tüm 42.591 klip Whisper-large-v3 (HF, fp16, 32'lik batch) ile yeniden transkript edildi ve DNSMOS ile puanlandı
(`scripts/transcribe_corpus.py`, ~97 dk tek 4090'da; `outputs/corpus-scores/scores.jsonl`).

| Ölçüm | Değer |
|---|---|
| CER (Türkçe normalizasyon) | medyan 0,000; p75 0,034; p85 0,062; p90 0,087; p95 0,145; ortalama 0,046 |
| CER > 0,05 / > 0,10 / > 0,15 | %18,6 / %8,3 / %4,7 |
| DNSMOS OVRL | p5 2,73; p10 2,90; medyan 3,28; p75 3,41 |
| OVRL < 2,8 / < 3,0 | %6,6 / %15,8 |
| Kalite puanı ile korelasyon | quality↔OVRL 0,47; quality↔CER −0,09 (kalite puanı büyük ölçüde akustik, transkript hatasını yakalamıyor) |

Yüksek CER'li satırlar: yabancı dil (Almanca/İngilizce) parçalar, tek kelimelik segment–transkript uyumsuzlukları,
Whisper halüsinasyonları. Filtreler (`scripts/make_drop_list.py` → `merge --drop-uids`):
- **clean**: CER ≤ 0,10, OVRL ≥ 2,8, ≥ 2 kelime → kalite ≥ 55 satırların ~%87'si (`data/tr55/clean`).
- **hq**: CER ≤ 0,05, OVRL ≥ 3,0, kalite ≥ 70, ≥ 3 kelime (`data/tr55/hq`) — son ince ayar aşaması için.

## 6. Koşu günlüğü

- **tr-pilot** (4 shard, tek GPU, 4k güncelleme; HF: `VoiceHub/dacvae-tts-tr-pilot`): WER 1,11 → 0,97, CER 0,92 → 0,69,
  SIM 0,92 → 0,95 (2k → 4k). Tavan (gerçek konuşma, codec'ten geçmiş): WER %5,1 / CER %1,6, DNSMOS OVRL 3,27.
- **tr-nano-a** (17 shard, kalite ≥ 55, GPU 0, `nano_tr.yaml`, bütçe 12000, 40k güncelleme, 0,56 s/güncelleme, tepe ~17 GB):
  monitor (48 vaka, 10 görülmemiş konuşmacı, large-v3, g=3, 32 adım): 5k 0,86/0,57 · 10k 0,57/0,35 · 15k 0,40/0,235 ·
  20k 0,305/0,184 · 25k 0,256/0,148 · 30k 0,223/0,126 · 35k 0,181/0,106 (WER/CER); SIM 0,955 → 0,966 (35k'da 0,948);
  val flow 0,664 → 0,637 (5k → 15k), metin kazancı 0,015 → 0,027. Eğitim 20:24'te bitti (40k, 6,2 saat).
  **Görülmemiş 5 cümle testi (A-40k, g=5, 32 adım, 5 farklı prompt):** WER 0,141 / CER 0,057 / SIM 0,936 / DNSMOS OVRL 2,92;
  cümle bazında WER 0,05–0,32 (`VoiceHub/dacvae-tts-tr-nano-a/custom-sentences-step-0040000`, metinler README'de).
- **Örnekleyici taraması (A, 15k, aynı 48 vaka):** g=2 0,466 · g=3 0,399 · **g=4 0,368** · g=3 + `guidance_until` 0,5 0,399
  (aynı; 54 vs 64 ileri geçiş) · noise_scale 0,9 0,417 · sway 0 0,408 · duration_scale 0,9 0,425 · **16 adım 0,394**
  (32 adımla aynı). **A 20k:** g=3 0,305 · g=4 0,274 · g=5 0,257 · **g=6 0,238** (SIM 0,966 → 0,962) · g=4 + until 0,5 + 16 adım 0,302:
  guidance 6'ya kadar WER düşüyor. **A 40k + DNSMOS:** g=3 0,202 / OVRL 3,07 · g=4 0,180 / 3,02 · **g=5 0,162 / 3,02** ·
  g=6 0,148 / 2,95 (SIM 0,947 → 0,945; gerçek konuşmanın codec sonrası OVRL'si 3,27). Çalışma noktası g=5, 32 adım.
- **tr-nano-b-ke4** (aynı veri, GPU 1, `nano_tr_ke4.yaml`: batch expansion 4, bütçe 7000, 0,58 s/güncelleme):
  5k 0,873/0,576 · 10k **0,489/0,313** (A: 0,570/0,347) · 15k **0,356/0,223** (A: 0,399/0,235) → Ke=4 aynı adımda önde
  (Supertonic bulgusu erken evrede tekrarlandı) · 20k 0,324/0,186 (A: 0,305/0,184) · 25k **0,233/0,137** (A: 0,256/0,148) ·
  30k **0,179/0,099**, SIM 0,963 (A: 0,223/0,126) · 35k 0,198/0,110, SIM 0,964 (A: 0,181/0,106, SIM 0,948) → WER başa baş,
  B SIM'i koruyor; round-2 tarifi Ke=4. A'da val flow 30k'dan sonra 0,628'de plato, train flow düşmeye devam (hafif aşırı uyum).
  **B 40k (21:45'te bitti):** WER 0,166 / CER 0,089 / SIM 0,964 (g=3) — A 40k: 0,202 / 0,113 / 0,947 → **B nihai model adayı**.
  5 görülmemiş cümle (g=5): B 0,121 / 0,044 / SIM 0,952 / OVRL 3,08; A 0,141 / 0,057 / 0,936 / 2,92.
  **B 40k guidance/DNSMOS:** g=3 0,169 / 3,11 · g=4 0,163 / 3,08 · **g=5 0,133 / 3,01** · g=6 0,131 / 2,96.
  **Freya-TR-Eval (495 cümle, 24 görülmemiş prompt, g=5, 32 adım, HF Whisper-large-v3 greedy):** B WER **%10,1** / CER %5,4 /
  SIM 0,945 / OVRL 2,84 (261 cümle hatasız); A %10,5 / %5,6. Makale: Piper 4,4 · MMS 6,8 · FreyaTTS-183M 8,0/3,0 · XTTS-v2 11,1 ·
  F5-TTS 24,3. Hatalar: 197 değiştirme / 111 ekleme / 88 silme (3.911 kelime); eklemeler kısa cümlelerde süre kuralının fazla
  uzunluk vermesinden ("dolgu" kelimeler) → süre kuralı kısa metinler için düzeltilmeli. Prompt'a göre WER %5–23.
  **Standart protokol (faster-whisper large-v3, beam 5, aynı WAV'lar):** B-40k WER **%9,1** / CER %4,6 / 273 hatasız cümle →
  FreyaTTS-183M (8,0) ile XTTS-v2 (11,1) arasında. (Greedy HF Whisper %10,1 ölçtü; farklı ASR çözümlemesi ~1 puan fark yaratıyor.)
- **tr-stage2-b** (GPU 1, 21:46'da başladı): B-40k'dan `--init-from`, clean cache, LR 4e-4, Ke=4, bütçe 7000, 20k güncelleme,
  0,68 s/güncelleme → ~01:10'da biter. Başlangıç flow 0,53 (sıcak başlangıç doğrulandı). Monitor (g=3): 2,5k 0,185 · 5k 0,220 ·
  7,5k 0,197 · 10k 0,184/0,090 · 12,5k 0,189 · 15k **0,171/0,087** — yüksek LR'li sıcak başlangıç önce bozuyor, LR düştükçe B-40k
  (0,166/0,089) seviyesine dönüyor · 17,5k **0,151/0,079** · **20k (01:06'da bitti): g=3 0,147/0,070 · g=4 0,140/0,073 · g=5 0,144/0,070 ·
  g=6 0,127/0,063** (B-40k: g=5 0,133, g=6 0,131) → LR kuyruğunda B-40k'yı geçti; 5 cümle: WER 0,121 / CER 0,048.
  **Freya-TR-Eval (faster-whisper large-v3 beam 5, g=5, 24 görülmemiş prompt): WER %7,1 / CER %3,6 / SIM 0,944 / OVRL 2,82**
  (B-40k aynı protokolde %9,1 / %4,6) → makaledeki FreyaTTS-183M'nin (8,0 / 3,0) WER'inin altında. HF: `VoiceHub/dacvae-tts-tr-stage2-b`.
- **tr-w512-clean** monitor (g=3): 5k 0,817 · 10k 0,373 · 15k 0,248 · 20k 0,208 · 25k **0,179/0,105** (A 25k 0,256, B 25k 0,233)
  · 30k **0,153/0,096** · 35k 0,152/0,094 (B-40k 0,166/0,089'un altında) → genişlik 512 + temiz veri aynı adımda açık ara önde; 60k ~06:40.
- **tr-stage3-hq** (GPU 1, 01:45'te başladı): stage-2-20k'dan `--init-from`, hq cache (20.966 satır: CER ≤ 0,05, OVRL ≥ 3,0, kalite ≥ 70),
  LR 2e-4, 10k güncelleme (03:22'de bitti). Monitor (g=3): 2,5k 0,140/0,066 · 5k 0,133/0,065 · 7,5k 0,139 · **10k: g=3 0,140/0,076 ·
  g=4 0,133/0,080 · g=5 0,115/0,067**; 5 cümle 0,121/0,048. **Freya-TR-Eval (beam 5): WER %6,9 / CER %3,5 / SIM 0,944 / OVRL 2,82**
  (stage-2: 7,1 / 3,6) → küçük ek kazanç; DNSMOS'ta kazanç yok. HF: `VoiceHub/dacvae-tts-tr-stage3-hq`.
  Süre ölçeği denemesi (Freya, stage-3): duration_scale 0,9 → WER %9,5 / CER %5,3 (1,0: %6,9 / %3,5) → kısa cümlelerde de kural
  1,0'da kalmalı; "dolgu" eklemeleri süre kısaltmakla çözülmüyor (kelime yutma artıyor).
- **tr-w512-clean** devam: 40k 0,136/0,074 · 45k 0,125/0,062 · 50k **0,097/0,051** · 55k 0,106/0,054 · **60k 0,098/0,052** (g=3;
  06:05'te bitti, 9,5 saat) → en iyi model. **60k nihai:** g=4 0,102 · g=5 0,099/0,053 · g=6 0,099/0,050 (SIM ~0,96);
  5 cümle **0,081/0,035**; **Freya-TR-Eval (beam 5): WER %4,3 / CER %2,5 / SIM 0,946 / OVRL 2,89** (stage-3: 6,9/3,5; B-40k: 9,1/4,6)
  → makaledeki FreyaTTS-183M (8,0/3,0) ve XTTS-v2'nin (11,1) altında, Piper (4,4) düzeyinde. HF: `VoiceHub/dacvae-tts-tr-w512-clean`.
- **tr-w512-stage2-hq** (GPU 1, 06:18'de başladı): C-60k'dan `--init-from`, hq cache, LR 2e-4, 10k güncelleme → ~08:00.
- **tr-w512-clean** (round 2, GPU 0, 20:35'te başladı): genişlik 512 / 8 head (66,5M), Ke=4, bütçe 6000, clean cache
  (32.226 satır), 60k güncelleme, 0,60 s/güncelleme → ~06:40'ta biter.
- Planlanan: B-40k'dan `--init-from` ile clean cache üzerinde 2. aşama (LR 4e-4, 20k, GPU 1, B bitince ~21:45).

## 7. Sonuç tablosu (23 Eylül 2026, 08:00)

Tüm koşular tek RTX 4090'da, `Vyvo/tr-dataset-12` (kalite ≥ 55 → 37k satır / temiz → 32k / hq → 21k). Monitor = 48 görülmemiş-konuşmacı
vakası (çapraz-cümle prompt, Whisper-large-v3, g=3 aksi belirtilmedikçe). Freya = Freya-TR-Eval 495 cümle, 24 görülmemiş prompt, g=5,
32 Euler adım, faster-whisper large-v3 beam 5, Türkçe normalizasyon (sayılar okunuşa, İ/ı, noktalama yok).

| Koşu | Model | Veri | Adım | Monitor WER/CER (g=3) | Monitor g=5 | 5 cümle WER/CER | Freya WER / CER | SIM | DNSMOS | HF |
|---|---|---|---:|---|---|---|---|---:|---:|---|
| pilot | 51M | 4 shard | 4k | 0,97 / 0,69 | – | – | – | 0,95 | – | `VoiceHub/dacvae-tts-tr-pilot` |
| A | 51M (Ke=2) | kalite≥55 | 40k | 0,202 / 0,113 | 0,162 / 0,099 | 0,141 / 0,057 | %10,5 / %5,6 (greedy) | 0,944 | 2,83 | `…-tr-nano-a` |
| B | 51M (Ke=4) | kalite≥55 | 40k | 0,166 / 0,089 | 0,133 / 0,069 | 0,121 / 0,044 | **%9,1 / %4,6** | 0,945 | 2,84 | `…-tr-nano-b-ke4` |
| stage-2 | 51M ← B-40k | clean | +20k | 0,147 / 0,070 | 0,144 / 0,070 | 0,121 / 0,048 | %7,1 / %3,6 | 0,944 | 2,82 | `…-tr-stage2-b` |
| stage-3 | 51M ← stage-2 | hq | +10k | 0,140 / 0,076 | 0,115 / 0,067 | 0,121 / 0,048 | %6,9 / %3,5 | 0,944 | 2,82 | `…-tr-stage3-hq` |
| **C** | **66,5M (w512, Ke=4)** | **clean** | **60k** | **0,098 / 0,052** | **0,099 / 0,053** | **0,081 / 0,035** | **%4,3 / %2,5** | 0,946 | 2,89 | `…-tr-w512-clean` |
| C-stage2 | 66,5M ← C-60k | hq | +10k | 0,098 / 0,046 (g=4) | 0,084 / 0,042 | 0,040 / 0,013 | %4,6 / %2,5 | 0,942 | 2,87 | `…-tr-w512-stage2-hq` |

Referanslar (FreyaTTS makalesi, aynı 495 cümle, Whisper-large-v3): Piper 4,4 · MMS-TTS 6,8 · FreyaTTS-183M 8,0 / 3,0 · XTTS-v2 11,1 · F5-TTS 24,3.
Gerçek konuşmanın codec sonrası tavanı: WER %5,1 / CER %1,6, DNSMOS OVRL 3,27 (8 vaka).

**Ana dersler**
1. Genişlik 448 → 512 (+30 % parametre) ve CER/DNSMOS filtreli veri birlikte en büyük kazancı verdi: Freya WER 9,1 → 4,3.
2. Batch expansion Ke=4 (Supertonic) erken evrede ve SIM'de yardımcı; 20k'da fark kapanıp 25k+'da yeniden açıldı.
3. Sıcak başlangıçlı ince ayar (init-from, LR 4e-4 → 2e-4, temiz/hq veri) 51M modelde Freya WER'i 9,1 → 7,1 → 6,9'a indirdi; ilk 5–10k
   adımda geçici bozulma normal (LR kuyruğunda toparlıyor).
4. Guidance 5 (32 Euler adım, sway −1) WER için en iyi nokta; 6 küçük ek kazanç ama DNSMOS düşüyor; 16 adım ≈ 32 adım.
   `guidance_until 0,5` WER'i değiştirmeden %15 hesap tasarrufu sağlıyor. noise_scale 0,9 ve duration_scale 0,9 zararlı.
5. DNSMOS OVRL 2,8–2,9'da takılı (tavan 3,27): kalite kazancı için codec/latent tarafı veya daha uzun eğitim gerekiyor;
   hq alt-kümesiyle ince ayar DNSMOS'u artırmadı.
6. Kalan hata türleri: kısa cümlelerde kuyruk "dolgu" kelimeleri (süre kuralı), nadir/yabancı özel adlarda harf hataları,
   prompt kalitesine bağlı WER dağılımı (%5–23).

**Yayımlanan model (Freya'ya göre seçildi, ezber riskine karşı iç-alan monitor dikkate alınmadı):** C-60k →
`VoiceHub/dacvae-tts-tr-w512` (model deposu: `model.pt` + `dacvae_tts` kodu + model kartı) ve Gradio demo
`VoiceHub/dacvae-tts-tr-demo`. C-stage2-hq iç-alan monitorde ve 5 cümlede daha iyi görünmesine rağmen Freya'da geriledi
(4,3 → 4,6 WER, SIM 0,946 → 0,942) → eğitim dağılımına uyum; yayımlanmadı.

**Demo Space:** `Vyvo/dacvae-tts-tr-demo` (ZeroGPU; VoiceHub ve kişisel hesapta Gradio Space için PRO gerekti → 402). Uygulama: checkpoint
menüsü (tüm koşular), özel repo/dosya ile keyfi checkpoint yükleme, Whisper (HF transformers `openai/whisper-large-v3-turbo`; Space imajı CUDA 13 olduğundan faster-whisper/ctranslate2 çalışmadı)
ile WER/CER doğrulama; uzaktan API testi geçti (yükleme ~3 s, üretim ~1 s, doğrulama WER 0).
