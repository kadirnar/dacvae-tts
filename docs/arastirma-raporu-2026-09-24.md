# DACVAE-TTS Türkçe: blok blok mimari, eğitim ve kalite araştırması (24 Eylül 2026)

**Kapsam.** Bu rapor, deponun tüm kodunun (model, eğitim, veri, codec, çıkarım, optimizer, süre, metrikler, demo) okunması
ve altı paralel literatür taramasının (DiT omurgası; metin/hizalama/süre; eğitim hedefi ve hız; konuşmacı benzerliği;
DACVAE'de ses kalitesi; Türkçe değerlendirme ve veri) sentezidir. Amaç: **küçük parametreli, hızlı eğitilen** bir
Türkçe zero-shot TTS'te **ses kalitesi, WER, CER ve SIM**'i birlikte iyileştirmek. Codec olarak Meta **DACVAE** sabit
kalır; her öneri bu latent uzayı için değerlendirildi.

Önceki sonuçların günlüğü: [turkce-arastirma-2026-09-22.md](turkce-arastirma-2026-09-22.md).
Uygulama planı ve issue'lar: [yol-haritasi.md](yol-haritasi.md).

**Kanıt etiketleri:** **[G]** güçlü (birden çok TTS makalesi/ablation veya yerel ölçüm), **[O]** orta (tek TTS makalesi
veya tutarlı görüntü-DiT kanıtı), **[Z]** zayıf (yalnız LLM/görüntü, mekanizmaya dayalı tahmin veya çelişkili kanıt).
Sayılar aksi belirtilmedikçe ilgili makalenin kendi raporudur; "yerel" yazanlar bu depoda ölçülmüştür.

---

## 0. Yönetici özeti

1. **Önce ölçüm düzeltilmeli.** Mevcut SIM metriği (`microsoft/wavlm-base-plus-sv`) aynı konuşmacı çiftlerine 0,89–0,96,
   farklı konuşmacılara bile 0,60–0,84 veriyor; bizim 0,94–0,95'lerimiz metriğin tavanında ve sistemleri ayırt etmiyor.
   Literatür standardı **SIM-o** (WavLM-large + ECAPA-TDNN, seed-tts-eval) aynı konuşmacıda ~0,6–0,7, farklıda ~0 verir.
   Ayrıca: Freya-TR-Eval'deki "WER 4,3 vs FreyaTTS 8,0" karşılaştırması birebir değil (Freya 8 kHz'e indirerek puanlıyor);
   495 cümlelik sette ~±0,7 WER puanından küçük farklar istatistiksel olarak ayırt edilemiyor; "24 görülmemiş prompt"
   yalnız 10 konuşmacıdan geliyor ve bölüm-içi diarizasyon etiketleri yüzünden aynı kişi train'de başka etiketle olabilir;
   best-of-3 sonuçları seçici (Whisper-turbo) ile hakemin (large-v3) aynı aileden olması nedeniyle şişik.
2. **Mimari büyük ölçüde doğru.** Ayrı self-attention + **LARoPE**'li cross-attention, paylaşımlı düşük-rank adaLN,
   QK-norm, patch 1, EDM ön-koşullandırma, logit-normal *t*, batch expansion Ke=4, CTC yardımcı kaybı ve Muon kanıtlarla
   destekleniyor. Joint/MM-DiT attention, patch/downsampling, katman paylaşımı, fonem/BPE girdisi, büyük önceden eğitilmiş
   metin kodlayıcı **önerilmiyor**.
3. **En büyük kaldıraçlar** (maliyete göre sıralı):
   - **Eğitim hızı ×1,8–2,5** (kalite değişmeden): adım başına ~10–15 host–device senkronizasyonunu kaldırmak, blok bazlı
     `torch.compile` + uzunluk dolgusu, aktivasyon checkpointing'i azaltmak, metin-negatiflerinin ekstra ileri geçişini
     kaldırmak (ΔFM tarzı latent negatifler). 60k güncelleme 9,5 saat → ~4–5,5 saat. [O]
   - **Süre ve cümle sonu:** sessizlik-farkında, hece tabanlı süre kuralı; süre-çeşitli yeniden sıralama; eğitimde
     kuyruk-sessizlik dolgusu. Kısa cümle sonundaki "dolgu kelime" eklemelerini hedefler. [O]
   - **SIM:** doğru metrik + guidance taraması (çıkarım), **TLA-SA** konuşmacı hizalama kaybı (+0,03–0,06 SIM-o, <%2 ek
     hesap), aynı/farklı cümle prompt karışımı, daha fazla Türkçe konuşmacı. [O–G]
   - **WER/CER:** A-DMA konuşma terimi / **speech-REPA** (önceden hesaplanmış çok dilli SSL özelliklerine hizalama; WER
     −%10–15 göreli, ~2× hızlı yakınsama), Türkçe metin normalizasyonu hataları (ör. "4'e" → "dörte"), CTC hedeflerinin
     küçük harf karaktere çevrilmesi. [O]
   - **Ses kalitesi:** DNSMOS açığının (2,89 vs codec tavanı 3,27) çoğu kırpılmadan değil **üretim hatasından**; en güçlü
     eğitim-tarafı kaldıraç bileşik ödüllü **GRPO** (F5'te DNSMOS +0,25, WER düşerek). Eğitimsiz: CFG'yi yalnız gürültülü
     yarıda uygulamak, tanh öncesi kazanç kontrolü, bileşik best-of-N seçici. Veri tarafında Sidon restorasyonu (Türkçe
     FLEURS'ta DNSMOS 3,07 → 3,45, CER değişmeden). [O]
4. **Küçük model ilkesi:** 66,5M'yi koruyup kazanılan hızı daha fazla güncellemeye harcamak, parametreyi azaltmaktan daha
   iyi (448 → 512 genişlik en büyük kazancı verdi). Yeni önerilerin toplam parametre maliyeti <2M. Parametre kısmak
   gerekirse ilk aday metin kodlayıcı (parametrelerin %22'si).

---

## 1. Mevcut durum

### 1.1 Mimari (configs/nano_tr_w512.yaml)

| Parça | Seçim |
|---|---|
| Codec | Donmuş `facebook/dacvae-watermarked`: 48 kHz, hop 1920 → 25 frame/s, 128 kanal sürekli VAE posterior ortalaması; −16 LUFS; kanal bazlı standardizasyon |
| Metin | `turkish-v1` normalizasyonu (sayılar okunuşa, İ/ı), UTF-8 byte (sözlük 260), prompt ve hedef transkripti tek akışta (joined) |
| Metin kodlayıcı | byte embedding + sinüzoidal konum, 4 × (depthwise conv k7 + MLP), 4 × pre-LN transformer (RoPE, GELU) |
| Üretici | 12 DiT bloğu, genişlik 512, 8 head, patch 1. Blok: paylaşımlı adaLN + rank-64 blok düzeltmesi (sıfır-init kapılar) → self-attention (RoPE, QK-RMSNorm) → cross-attention (LARoPE, γ=10) → GELU FFN (×3). Afin'siz LayerNorm |
| Koşul | zaman sinüzoidi → MLP + prompt latentlerinin MLP'sinin ortalaması (global "ses" vektörü) |
| Prompt | aynı cümlenin ilk %10–60'ı (within-utterance, F5/E2 infilling), %30 prompt dropout, %20 ortak CFG dropout |
| Hedef | EDM tarzı birim-varyanslı hız hedefi, tabakalı logit-normal *t*, frame ağırlıklı MSE, Ke=4 batch expansion, blok 8'de CTC (λ 0,1), atla/tekrarla metin negatifleri (hinge, λ 0,2) |
| Optimizer | Muon (gizli matrisler, parçalı ortogonalleştirme) + AdamW; LR 8e-4, 2k ısınma, cosine → %10, wd 0,01, EMA 0,9999, bf16, grad-clip 1 |
| Süre | prompt'un byte başına frame oranı (F5 kuralı); çıkarımda `clamp`/`predictor`/`auto` |
| Örnekleme | Euler 32 adım, sway −1, CFG 5 (tek batch'te iki dal) |

### 1.2 Parametre dağılımı (yerel ölçüm, `FlowTTS(ModelConfig)`)

| Modül | 66,5M (w512) | Pay |
|---|---:|---:|
| Üretici FFN'leri | 18,9M | %28,4 |
| Üretici self-attention | 12,6M | %19,0 |
| Üretici cross-attention | 12,6M | %19,0 |
| Metin kodlayıcı transformer blokları | 10,5M | %15,8 |
| Metin kodlayıcı MLP'leri | 4,2M | %6,3 |
| adaLN (paylaşımlı + düşük-rank) | 6,4M | %9,6 |
| Diğer (zaman, referans, CTC, giriş/çıkış, embedding) | 1,3M | %1,9 |

### 1.3 Sonuçlar (özet)

En iyi model C (66,5M, 60k güncelleme, tek 4090'da 9,5 saat, temiz veri): Freya-TR-Eval WER %4,3 / CER %2,5,
SIM (base-plus-sv) 0,946, DNSMOS OVRL 2,89 (gerçek konuşmanın codec sonrası tavanı 3,27). `auto` süre modu 3,61/1,81;
`auto` + best-of-3 1,59/0,72 (bkz. §2.4 uyarısı). Kalan hata türleri: kısa cümle sonunda dolgu kelimeler, tek kelime
tekrar/düşmesi, nadir/yabancı özel adlarda harf hataları, prompt'a göre %5–23 arası değişen WER, yüksek CFG'de
decoder'ın tanh tavanına dayanması.

---

## 2. Ölçüm güvenilirliği (her şeyden önce)

### 2.1 SIM metriği yanlış ölçekte [G]

- Kullandığımız `wavlm-base-plus-sv` model kartına göre aynı/farklı konuşmacı eşiği ~0,86; aynı konuşmacıda 0,89–0,96,
  farklıda 0,60–0,84 veriyor. seed-tts-eval'in modeli aynı VoxCeleb kliplerinde aynı konuşmacı 0,60–0,69, farklı
  −0,17–0,18 veriyor.
- **Standart (SIM-o):** UniSpeech `ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large')`, s3prl WavLM-Large
  omurgası + ECAPA başlığı, `wavlm_large_finetune.pth`. Ses: kanal 0, 16 kHz'e yeniden örnekleme, kırpma/normalizasyon
  yok, tam klip; üretilen ses (prompt kareleri hariç) ↔ **orijinal** prompt wav'ı arasında kosinüs. seed-tts-eval ve
  F5-TTS kodu birebir böyle. Orijinal indirme bağlantısının süresi dolmuş; HF aynası:
  `bezzam/wavlm_large_finetune_seed_tts_eval` (sha256 `51f07e3b…f7f94b`). Lisans CC BY-SA 3.0 (değerlendirme için sorun yok).
- `monitor.py` ve `eval_sentences.py` codec'ten geçirilmiş prompt'a karşı ölçüyor: bu **SIM-r**'dir, SIM-o değil.
- Tipik SIM-o değerleri: Seed-TTS gerçek ses 0,73–0,76; F5-TTS 336M 0,66–0,76; ZipVoice 123M 0,67–0,75; SupertonicTTS
  60M 0,60; Türkçe MiniMax seti: MiniMax-Speech 0,779, ElevenLabs 0,596, OmniVoice 0,851, VoxCPM2 0,871.
- Sonuç: "bağımsız konuşmacı guidance'ı işe yaramadı" gibi SIM'e dayalı eski kararlar **yeniden ölçülmeli**. Kötüye
  kullanılamayan ikinci bir bağımsız metrik (CAM++ veya SpeechBrain ECAPA) de raporlanmalı — özellikle ileride WavLM-SV
  ile eğitim/yeniden sıralama yapılırsa.

### 2.2 WER karşılaştırılabilirliği ve istatistik [G]

- **FreyaTTS protokolü:** tüm sistem çıktıları 8 kHz'e indirilip 16 kHz'e geri çıkarılıyor, faster-whisper large-v3
  (beam 5), jiwer ile korpus düzeyinde WER/CER (CER boşlukları sayıyor), kesme işareti boşluğa çevriliyor; klonlama
  baz modellerine tek sabit referans veriliyor; FreyaTTS'nin kendisi tek sesli. Gerçek FLEURS-tr konuşması bu protokolde
  %9,7 WER alıyor. Yani bizim tam bant, 24 podcast prompt'lu %4,3'ümüz ile Freya'nın 8,0'ı birebir değil; kendi
  sayımızı **Freya'nın 8 kHz betiğiyle** de raporlamalıyız.
- **Güven aralıkları (yerel, yayımlanan sonuçlardan):** korpus WER 4,32 için %95 GA cümle-bootstrap [3,59; 5,09],
  konuşmacı-kümelenmiş [3,07; 5,62]; CER 2,50 → [2,03; 2,99] / [1,68; 3,38]. Konuşmacı başına WER 1,2–9,8. Eşli
  karşılaştırmalar: süre tahmincisi ΔWER −0,18 [−1,26; +0,64] (berabere), `guidance_until 0,7` +0,10 (berabere), `clamp`
  −0,72 [−1,49; 0,00] (sınırda). **~0,7 puandan küçük farklar bu sette çözülemiyor.** 5 cümlelik set ±6,8 WER; 8 klipli
  codec tavanı yorumlanamaz.
- **Seçici yanlılığı:** best-of-N'de adayları Whisper-turbo seçip large-v3 ile puanlamak aynı aile önyargısı taşır
  (arXiv 2607.08256: aynı aileden seçici/hakem, "oracle" boşluğunun 2–3 katını geri kazanıyor). Seçici farklı aileden
  olmalı (ör. Omnilingual ASR) veya ikinci hakemle doğrulanmalı.
- **Deterministiklik:** `Evaluator.score` faster-whisper'ın sıcaklık fallback'ini açık bırakıyor (0 → 1,0); kötü
  kliplerde çözümleme örneklemeye kayabiliyor → `temperature=0`, `without_timestamps=True`.
- **ASR önyargısı:** eğitim verisi large-v3 CER'iyle filtrelendi ve large-v3 ile puanlanıyor; model "Whisper dostu"
  konuşmayı öğrenebilir. İkinci, farklı aileden hakem (omniASR_CTC_3B_v2; Türkçe FLEURS WER 9,2 / CER 2,0) önerilir.
- **CER birincil olmalı:** Türkçe'de large-v3 WER'i CER'in ~3,5 katı (FLEURS-tr 5,66 / 1,62); tek yanlış ek bütün uzun
  kelimeyi yanlış sayar.

### 2.3 Konuşmacı sızıntısı [O]

`speaker_split` konuşmacı etiketinin hash'ine göre bölüyor; etiketler `<bölüm>_speaker_k` biçiminde bölüm-içi
diarizasyon. Tekrarlayan podcast sunucuları başka bölümün etiketiyle train'de olabilir; bu hem "görülmemiş konuşmacı"
WER'ini hem SIM'i şişirir. Çözüm: bölüm/program düzeyinde ayırma ve WavLM-ECAPA merkezlerinin bölümler arası
kümelenmesiyle test konuşmacılarının train kümelerine yakınlığının (ör. kosinüs < 0,6) denetlenmesi.

### 2.4 Metrik metin normalizasyonu hataları [O]

`metric_text_turkish`: (1) Mn-silme satırı ölü kod — alnum filtresi U+0307'yi önce boşluğa çeviriyor ("i̇stanbul" →
"i stanbul"; Python-`lower()` kullanan ikinci bir ASR'de patlar); (2) â/î/û katlanmıyor ("kâr" ≠ "kar"); (3) Roma
rakamları ("II." → "ıı"); (4) boşluksuz sıra sayıları ("21.yüzyıl"); (5) saat okunuşları ("3.30'a" → "üç nokta otuza");
(6) yazım varyantları ("avro"/"euro"); (7) birim/kısaltmalar; (8) tireli kelimeler ("e-posta" iki kelime); (9) kısaltma
harf düzeltmesi ("COVID" → "covıd"). Freya'yı yalnız 4,32 → ~4,37 değiştirir ama podcast monitörü ve eğitimdeki CER
filtresi için önemli. Referans uygulama: **trnorm** (Apache-2.0).

Eğitim normalizasyonunda da hata var: **ünsüz yumuşaması eksik** — "4'e" → "dörte" (doğrusu "dörde"), "4'ü" → "dörtü",
"2024'e" → "iki bin yirmi dörte". Bu hatalı metin eğitime sızdı. Ayrıca "T.C.", "A.Ş.", "Ltd. Şti." açılmıyor. Düzeltme
yeni bir normalizasyon sürümü (`turkish-v2`) olarak yapılmalı ki mevcut cache'ler ve checkpoint'ler bozulmasın.

### 2.5 Önerilen değerlendirme protokolü v2

1. Birincil hakem faster-whisper large-v3 (fp16, `language="tr"`, beam 5, `temperature=0`, `without_timestamps=True`);
   ikincil hakem omniASR_CTC_3B_v2. Kazanç yalnız iki hakem aynı yönde ise iddia edilir.
2. `turkish-v2` metrik normalizasyonu (her iki tarafa aynen), trnorm çıktılarıyla birim testli.
3. Metrikler: boşluksuz CER (birincil), WER (korpus + cümle ortalaması), S/D/I dağılımı, %WER>50 ve hatasız cümle oranı;
   SIM-o (orijinal prompt'a karşı) + gerçek-ses tavanı; DNSMOS SIG/BAK/OVRL + UTMOSv2 (+ Distill-MOS/NISQA-48k isteğe bağlı);
   çıktı bant genişliği, kırpılma oranı, LUFS, RTF; prompt'un kendi DNSMOS'u.
4. Setler: Freya-TR-Eval (register ve uzunluk kovalarına göre + 8 kHz sütunu), MiniMax-MLS-Türkçe (100 cümle, 2 sabit
   prompt; ticari ve büyük açık sistemlerle doğrudan karşılaştırma), ≥200 vakalık iç podcast seti (≥40 gerçekten
   görülmemiş konuşmacı, gerçek ses referanslı), ≥100 cümlelik zor set (sayılar, tarihler, kısaltmalar, yabancı adlar, sorular).
5. Tavanlar: gerçek ve DACVAE'den geçirilmiş gerçek ses, ≥200 klip.
6. 3 gürültü tohumu + ikinci prompt çekilişi; konuşmacı-kümelenmiş bootstrap (B ≥ 5000); eşli karşılaştırmada GA 0'ı
   içeriyorsa sonuç "berabere".

---

## 3. Blok blok değerlendirme

Her satır: **ne yapıyoruz → kanıt → karar**.

### 3.1 Codec ve latent uzayı

| Konu | Kanıt | Karar |
|---|---|---|
| DACVAE yapısı | DAC'ın RVQ'su yerine KL-düzenlenmiş VAE; 48 kHz, 25 Hz, C=128; decoder başı `tanh(conv(snake(x))) + α·filigran` → 0,999 "kırpılması" aslında tanh doygunluğu. Resmi `encode()` örnek döndürüyor, biz ortalamayı kullanıyoruz | Koru |
| Posterior ortalaması vs örnek | Yerel: posterior std 0,0026–0,0046, örnekleme log-mel'i yalnız 0,021 değiştiriyor | Ortalama doğru [G] |
| Kanal bazlı standardizasyon | Kanal std'leri 0,61–1,00 (1,6× yayılım); Echo'nun "zarar verdi" gözlemi varyansı çok farklı PCA latentlerine ait | Koru; global ölçek A/B'si değersiz [Z] |
| PCA/boyut azaltma | Yerel: 96/64/32 bileşen log-mel 0,67/1,12/1,70 (codec tabanı 0,58); %99 varyans için 120 bileşen gerek | **Reddet** [G] |
| Patch 2 | Yerel: komşu frame korelasyonu 0,054; P=1'in 4k adımı P=2'nin 16k adımını geçti | Patch 1 koru [G] |
| Latent gürültü artırımı | Posterior gürültüsü σ'nın ~%0,5'i; decoder gürültüye dayanıklı değil (0,2σ izotropik hata = codec'in kendi hatası kadar bozulma) | Reddet [O] |
| Latent ölçekleyerek ses seviyesi | Yerel: −18 dB kazanç latent normunu yalnız %4 değiştiriyor, yönü değiştiriyor (kosinüs 0,86) | Asla latent ölçekleme [G] |
| Filigran dalı | LAION ve Irodori filigranı kapatıyor; klonlama sistemi olduğu için yayımlanan modelde açık kalmalı | Yalnız teşhis amaçlı A/B |
| Daha düşük boyutlu/ince ayarlı codec | LongCat: sabit kapasitede 64 > 128 > 256 boyut TTS için; Irodori'nin 32-d Semantic-DACVAE'si UTMOSv2 2,28 → 2,40 | Codec değişikliği sayılır; kapsam dışı (gelecek iş) |

**Sonuç:** Latent uzay zor (tam rank, zamansal beyaz, decoder'a duyarlı) ama DACVAE'yi değiştirmeden yapılabilecek en iyi
kullanım zaten mevcut olana çok yakın. Kalite açığı üretici hatasından geliyor → çözüm eğitim hedefi, guidance ve
post-training tarafında.

### 3.2 Metin girdisi ve metin kodlayıcı

- **Birim:** UTF-8 byte. Türkçe'de iki baytlık harfler (ı ü ş ç ğ ö) harflerin ~%11,8'i → ~%10 daha uzun dizi; maliyet
  ihmal edilebilir. Latin alfabesinde byte/karakter cezasını ölçen yayımlanmış çalışma yok. Türkçe imlası %95,4 saydam
  (İngilizce %36); espeak-ng Türkçe çift ünsüzleri birleştiriyor ("elli" → "eli"); ZMM-TTS'te ham IPA karaktere üstün
  değil; MaskGCT'de BPE G2P'den kötü (WER 4,04 vs 2,47). **Karar: byte'ı koru [O]; fonem/BPE reddet [O].** Yan etkiler:
  LARoPE'de iki baytlık harf diyagonal önvarsayımın iki katı payını alıyor; byte kuralı ı/ş/ü'lü cümlelere fazla süre
  veriyor; CTC hızlı konuşmacılarda (16–19 byte/s vs 25 fps) sınırda ve `zero_infinity` bu örnekleri sessizce sıfırlıyor.
  Ucuz düzeltme: **CTC hedeflerini Türkçe küçük harf karakter + boşluk (noktalama yok) yap** [Z, sıfır maliyet]. Yeniden
  eğitim yapılırsa ~95 sembollü karakter sözlüğü veya LARoPE anahtarlarını karakter indeksiyle konumlama A/B'si.
- **Kodlayıcı:** F5'te ConvNeXt metin blokları kaldırılınca WER 4,17 → ~5,5; ZipVoice'ta metin kodlayıcısız 1,69 → 2,04;
  SupertonicTTS 6 ConvNeXt + 4 self-attention. DiTTo'da 415M ByT5 (yalnız metin) WER 6,22 vs konuşmayla birlikte eğitilmiş
  85M SpeechT5 3,07 — **hizalanmış olmak boyuttan önemli.** Irodori v4'te ModernBERT-ja standart CER'i kötüleştirdi.
  **Karar: mevcut kodlayıcıyı koru [O]; BERTurk/ModernBERT ekleme.** Parametre kısmak gerekirse ilk aday: 4 → 2 attention
  bloğu (~5M tasarruf) — A/B ile.

### 3.3 Üretici DiT bloğu

| Konu | Kanıt | Karar |
|---|---|---|
| Attention düzeni | Cross-attention çok küçük ölçekte çalışıyor (SupertonicTTS 19M WER 2,41; DiTTo-S 42M 3,07). F5/E2 dolgu-token yaklaşımı 60k adımda hizalanamıyor (ZipVoice'ta 20,19 WER; F5-small 100k'da 29,5). F5'in 151M MMDiT'i "hızlı öğrenip hızlı çöktü"; DiT-Air'de tek akış küçük ölçekte MMDiT'ten kötü. Echo-tarzı joint attention ≤100M'de ablation'sız ve LARoPE'yi kaldırır | **Koru** [O] |
| LARoPE | Aynı 19M modelde WER 2,41 → 2,25; uzun cümlelerde 4,98 → 2,16; 200k'da CER 2,00 → 1,23; süre ölçeklemeye dayanıklılık. Yerel: metin kullanımını ~10× erken açıyor | **Koru** [G] |
| Normalizasyon | RMSNorm/LayerNorm farkı ≤140M'de ihmal edilebilir (SR-DiT 4,58 → 4,56); QK-norm küçük ölçekte küçük ama yüksek LR/Muon için kararlılık sağlıyor | Koru; RMSNorm yalnız hız için isteğe bağlı |
| adaLN | Paylaşımlı + düşük-rank (EzAudio SOLA / Echo); DiTTo'da global adaLN WER 3,38 → 2,93 ve %30 daha az parametre; saf adaLN-single EzAudio'da çöküyor | **Koru** [G] |
| FFN aktivasyonu | GLU eşit parametrede T5/LightningDiT'de iyi, 140M SR-DiT'te nötr/kötü | SwiGLU/GEGLU yalnız A/B [Z] |
| FFN içinde depthwise conv | ZipVoice'ta conv modülleri kaldırılınca WER 1,69 → 9,79; U-DiT/SANA'da kazanç; FastSpeech CMOS −0,11 (conv'suz). Ters: F5 "+Conv2Audio" 4,17 → 5,78 | A/B: k=5 depthwise conv, +0,1M, ~%2 hesap [O–Z] |
| Value residual | 140M DiT'te (SR-DiT) FID 4,02 → 3,64, ablation'ın en büyük tek mimari kazancı; ~0 parametre | A/B [Z–O] |
| Attention kapısı | Qwen gated attention (LLM) kararlılık ve daha yüksek LR; Echo/Irodori/Darya kullanıyor; TTS ablation'ı yok | Head-wise kapı A/B (+0,1M) [Z] |
| Uzun skip | DiTTo (cross-attention DiT, bizim gibi): giriş→çıkış skip WER 3,30 → 2,93, SIM-r 0,573 → 0,588. Ters: F5 (in-context) 4,17 → 5,17 | A/B: concat(h0, h12) → LN → Linear, +0,5M [O] |
| Downsampling / U-Net | 25 Hz'de zarar (DiTTo U-Net 3,70 vs 2,93; yerel P=2) | Reddet [O] |
| Pozisyon | RoPE > mutlak (SR-DiT, LightningDiT); yarım-head RoPE marjinal | Koru |
| Derinlik/genişlik | DiTTo S→B sıçramasının çoğu SIM; MobileLLM'de derin-dar daha iyi; katman paylaşımı eşit hesapta kaliteyi düşürüyor | İsteğe bağlı 16×448 vs 12×512 A/B; paylaşım yok |
| Çıkış başı | JiT: x-tahmini yalnız token boyutu ≈ genişlik iken gerekli (bizde 128/512); DiT/F5 son normda adaLN kullanıyor | EDM'yi koru; final adaLN A/B [Z] |
| Koşula havuzlanmış metin | DiTTo: 3,00 → 2,93 | Final adaLN ile birlikte A/B [Z] |

### 3.4 Referans / konuşmacı koşullandırma

- **In-context infilling koru.** Koel-TTS'te görülmemiş konuşmacıda in-context (0,637) > donmuş SV embedding (0,619) >
  ayrı kodlayıcı + cross-attention (0,601); ayrı kodlayıcı görülen konuşmacılara aşırı uyuyor. YourTTS 87M (global
  embedding) 0,462 vs F5 158M in-context 0,584. [O]
- **Global ses vektörünü koru;** MiniMax'ta embedding + prompt birlikte +0,016/+0,046. İsteğe bağlı A/B: öğrenilen MLP
  yerine donmuş **CAM++** 192-d embedding → Linear → adaLN (~200k konuşmacıyla eğitilmiş önbilgi; ~0,1M). Ayrı referans
  kodlayıcı ancak ≥20 s çok klipli referanslara geçilirse (Irodori: tek klip 0,661 → 30 s 0,752 → 120 s 0,775).
- **Eğitim çiftleri:** F5/E2/Voicebox yalnız cümle içi eğitimle bile çapraz-cümle SIM-o 0,66–0,67'ye ulaşıyor; ama
  VoiceStar'da aynı konuşmacının farklı cümlesinden prompt karıştırmak (CPM) WER'i 8,49 → 6,42 düşürüyor (SIM ~sabit),
  prompt tekrarını güvenli kılıyor. Cümle-içi eğitim prozodi kopyalamayı ödüllendiriyor. **Öneri:** %50–60 cümle-içi +
  %40–50 aynı `<bölüm>_speaker_k` etiketli farklı cümle (1–3 klip birleştirme, 2–15 s), prompt kesimini sessizlikte yapmak,
  saniye bazlı 3–12 s prompt dağılımı. [O]
- **Yapısal gözlem (bizim modele özgü):** joined düzen + LARoPE'de frame *i*, token *i·S/L*'ye eşleniyor; prompt/hedef
  sınırındaki önvarsayım yalnız prompt'un byte başına frame oranı hedefinkine eşitken doğru. Byte kuralı tam da bunu
  zorluyor; başka her süre sınırda bir "kırılma" yaratıyor ve cümle-içi eğitim modele bu kırılmayı hiç göstermiyor. Tüm
  süre ölçeklemelerinin zarar vermesinin ve prompt'a göre WER'in %5–23 salınmasının makul açıklaması bu. CPM + yalnız
  prompt'a tempo pertürbasyonu (VoiceStar: ±%25, WER 6,42 → 5,66) modeli bu kırılmaya dayanıklı kılar. [O]

### 3.5 Eğitim hedefi

| Konu | Kanıt | Karar |
|---|---|---|
| EDM ön-koşullandırma | Yerel: her *t*'de v-tahmininden iyi; x-tahmini temiz uçta bozuluyor; SR-DiT'te x0-tahmini latentte belirgin kötü | Koru [G] |
| Logit-normal *t* | SD3: 61 formülasyonda en iyi ortalama sıra; BareWave (TTS): uniform'dan hızlı erken yakınsama; eğitimin son fazında uniform'a geçmek SIM 0,522 → 0,543, UTMOS 3,70 → 3,82 | Koru + son %15–20'de uniform A/B [Z–O] |
| Batch expansion Ke=4 | SupertonicTTS: Ke 1→4 +%64 süre vs 4× B +%253; yerel: Ke4, Ke2'yi 0,02–0,04 WER geçti ve SIM'i korudu | Koru [G] |
| CTC (A-DMA) | F5-small 585 h: yalnız metin CTC WER 7,47 → 2,48; blok 8 iyi | Koru; hedefleri karaktere çevir |
| Metin negatifleri (hinge) | Adım başına ek metin kodlama + üretici ileri/geri → ~%20–25 ek hesap. RobustSpeechFlow'un makaleye sadık hali **latent** negatifler (uzunluk korumalı tekrar/atlama + sessizlik dolgusu) ve **doğru** metin; kayıp L_pos − 0,2·L_rand − 0,2·L_aug, ek ileri geçiş yok; Seed WER 1,44 → 1,38 | ΔFM/latent negatiflerle değiştir veya kaldır; eşit duvar-saatinde A/B [O] |
| **Speech-REPA / A-DMA konuşma terimi** | A-DMA: metin CTC 8'de + HuBERT hizalama 12'de WER 2,35/SIM 0,609 (yalnız CTC 2,60/0,586); ana tablo 2,68 → 1,97, ~2× yakınsama. BareWave (983M): WavLM REPA WER 3,32 → 2,86, SIM 0,471 → 0,522. AG-REPA: WER 5,82 → 3,45 (daha zayıf kanıt). Aynı katmana CTC ve REPA koymak kötü; erken katman kötü | **Ekle:** önceden hesaplanmış çok dilli SSL (mHuBERT-147 veya w2v-BERT 2.0) 25 fps'e havuzlanmış, blok 10–11, Conv1d(k3) projektör, negatif kosinüs, λ 0,5–1; ~<%2 ek hesap; depolama ~10 GB (PCA-256 ile ~3 GB) [O–G] |
| **TLA-SA konuşmacı hizalama** | Ara blok özelliklerini hedef frame'lerde ortalayıp katman başına MLP ile WavLM-SV embedding'ine kosinüs; katman ağırlıkları zaman embedding'inden softmax. LibriTTS F5-benzeri: Sim-WavLM 0,398 → 0,458 ve **farklı** SV modeli (ERes2Net) 0,500 → 0,571 (metrik hilesi değil); CosyVoice 2 100k h: 0,606 → 0,644 (en), 2,9× hızlı SIM yakınsaması; WER ±0,1 | **Ekle:** önceden hesaplanmış 256-d WavLM-ECAPA (veya CAM++) utterance embedding'i; ~1,5–3M yalnız-eğitim parametresi; <%2 hesap [O–G] |
| Model-guidance eğitimi | Hedefe v + w·Δv (w=0,7) → CFG'siz örnekleme; F5 üzerinde MOS 4,026 → 4,159, WER 2,28 → 1,96; w ≥ 1 çöküyor | İnce ayar olarak A/B: CFG doygunluğunu ve 2× çıkarım maliyetini kaldırır [O] |
| Latent SV kaybı / GRPO | DMOSpeech (damıtma bağlamı) SIM 0,53 → 0,70; F5R GRPO SIM 0,698 → 0,730, WER 2,10 → 1,48 | Post-training fazına (§3.10) |

### 3.6 Süre

- **Kural vs tahminci:** DMOSpeech 2 (Seed-en): gerçek süre WER 1,821; F5 hız kuralı 2,028; **gözetimli tahminci 3,750
  (kuraldan kötü)**; GRPO ile optimize edilmiş tahminci 1,752 (gerçek süreden bile iyi). Yani naif tahminci kuraldan kötü,
  metrik-optimize tahminci en iyi. Irodori v4.1: gövde dondurulup yalnız süre tahmincisi yeniden eğitilince (fazla uzun
  tahmini düzeltmek için) CER 5,35 → 4,69. [G]
- **Kısa cümle / cümle sonu:** F5 kısa metinlerde (≤10 byte) hızı yavaşlatıyor, referans kenarlarındaki sessizliği
  kırpıp 50 ms ekliyor, referans metnini ". " ile bitiriyor; FreyaTTS 1–2 kelimelik girdiler için kısa segmentlerle SFT ve
  süre tabanı uyguluyor; Cross-Lingual F5-TTS 2, prompt'lara %30–70 sessizlik ekleyerek eğitilen hece tabanlı hız
  tahmincisiyle sessizlik-dolgulu prompt'larda göreli hatayı %70+ → %13–19'a indiriyor. **Whisper tarafı:** large-v3
  konuşmasız girdilerde %40,3 halüsinasyon üretiyor; Türkçe'de "Altyazı M.K." gibi sabit dizgeler — "dolgu" hatalarımızın
  bir kısmı ASR artefaktı olabilir. Podcast verisinde transkript edilmemiş "yani/şey" dolguları var; model fazla frame
  aldığında bunları üretmeyi öğrenmiş olabilir.
- **Öneriler:** (i) değerlendirmede sondaki sessizliği kırp + Silero VAD + halüsinasyon dizgesi filtresi; (ii)
  sessizlik-farkında hece tabanlı **artikülasyon hızı** kuralı (prompt kenarları kırpılır, >200 ms duraklamalar hariç
  hız, hedef noktalamasından duraklama bütçesi); (iii) tohumlar yerine **süre faktörleri {0,9; 1,0; 1,1}** arasında
  yeniden sıralama; (iv) DMOSpeech 2 fikri RL'siz: korpus çiftleri 5 süre faktöründe sentezlenip CER+SIM ile puanlanır,
  tahminci en iyi faktöre fit edilir; (v) eğitimde hedeflerin %30–50'sine 0–0,8 s kodlanmış-sessizlik latenti eklenip
  kaybın bu karelerde de hesaplanması, çıkarımda süre ×~1,05 + kırpma; (vi) örneklerin ~%25'inde `prompt_fraction_max`
  0,85 → kısa hedef kapsamı. Süre değişiklikleri LARoPE kırılmasıyla etkileşir → CPM/tempo eğitimiyle birlikte ele alınmalı.

### 3.7 Optimizer, çizelge, EMA, düzenlileştirme

- **Muon koru** [O]: Moonlight'ta AdamW'ye göre ~2× hesap verimi; adil ayarlanmış karşılaştırmalarda 0,1B'de 1,3–1,4×.
  DiT'te vanilla Muon ≈ AdamW, adaLN/QKV'yi parça parça ortogonalleştiren CMuon 2×'ten fazla — bizim `FUSED_ROWS`
  uygulamamız zaten bu. Sonra: küçük LR taraması (0,5×/1,5×), NorMuon (+%11 verim, 1,1B).
- **WSD çizelgesi** [O]: 1-sqrt soğumalı WSD cosine'e eşit ve herhangi bir noktadan dallanmaya izin veriyor; soğuma fazını
  temiz alt kümede (MiniCPM) ve uniform *t* ile yapmak, mevcut iki aşamalı "sıcak başlangıç ince ayarını" tek koşuya katar.
- **Aşırı uyum** [Z–O]: 51M modelde val flow 30k'dan sonra platoda; F5-TTS DiT'te 0,1 dropout kullanıyor. A/B: wd
  0,05–0,1 ve attention/FFN dropout 0,1.
- **EMA** [O, görüntü]: EDM2'ye göre en iyi EMA uzunluğu yapılandırmaya ve CFG'ye çok duyarlı; iki EMA izi (0,999 ve
  0,9999) veya post-hoc güç-EMA anlık görüntüleri. <15k adımlık ince ayarlarda 0,9999 fazla yavaş.

### 3.8 Verimlilik (tek RTX 4090)

Yerel analiz: ~44M üretici × ~24k frame × (6 + 2 yeniden hesap) + hinge ≈ 9 TFLOP/adım; 0,60 s'de ~15 TFLOP/s, yani
4090'ın ~165 TFLOPS bf16 kapasitesinin **~%9'u** — model hesaplama değil ek yük/bellek sınırlı. Kod gerçekleri:
`mask_values`/`per_example_mse`/`flow_loss` içindeki `.any()`/`isfinite` denetimleri, `negatives()`'teki `.cpu()`, CTC
uzunluk listesi ve kayıp/gradyan denetimleri adım başına ~10–15 senkronizasyon yapıyor. SDPA'ya verilen her boolean maske
FlashAttention'ı devre dışı bırakıyor (bellek-verimli çekirdeğe düşüyor).

Öneri sırası: (1) değer denetimlerini debug bayrağının arkasına al, negatif/CTC uzunluklarını dataloader'da üret (%3–8);
(2) hinge'in ek ileri geçişini kaldır (~%20); (3) her `Block`'u ayrı derle (regional compile), ses uzunluğunu 64'ün,
metni 32'nin katına doldur, yalnız batch boyutunu dinamik işaretle; PyTorch ≥ 2.9 (1,3–1,6×); (4) compile'ın bellek
tasarrufuyla checkpointing'i kapat veya seçici checkpointing (1,2–1,3×). Toplam gerçekçi **1,8–2,5×**. FP8 bu matris
boyutlarında daha yavaş (torchao tablosu); CUDA graphs ve FlexAttention paketleme şekiller statikleşince. Çıkarımda metin
K/V'si her adımda yeniden hesaplanıyor → önbelleğe alınabilir.

### 3.9 Örnekleme ve guidance

- Yerel: `guidance_until 0,5` WER'i değiştirmeden %15 hesap tasarrufu; `noise_scale 0,9` ve `duration_scale 0,9` zararlı;
  16 adım ≈ 32 adım (WER için); CFG 5 çalışma noktası; APG (η 0,5, β −0,3) ve CFG-rescale φ 0,7 kırpılmayı azaltıp WER'i
  artırdı. Kırpılma DNSMOS'u yalnız ~0,1 düşürüyor (g=3 → 5: 3,11 → 3,01); APG kırpılmayı büyük ölçüde kaldırıp yalnız
  +0,07 kazandırdı → açığın çoğu kırpılma değil.
- Literatür: guidance aralığı (yüksek gürültüde zararlı, düşükte gereksiz; görüntüde FID 1,81 → 1,40); görüntü CFG
  düzeltmeleri TTS'te çoğunlukla başarısız (CFG-Zero* ve sıfır-init F5'te baz CFG'den kötü); F5'te Euler ≈ midpoint
  (32 NFE, UTMOS 3,90 vs 3,89) ama midpoint 1,7× yavaş; SIM için MegaTTS 3 çoklu koşul CFG (konuşmacı 3,5 / metin 2,5)
  SIM-O 0,68 → 0,71; Selective CFG (erken normal, geç konuşmacı-vurgulu) F5'te SIM +0,007–0,011, zh'de kazanç yok.
- **Öneriler (eğitimsiz):** (i) CFG yalnız t<0,5'te ve o pencerede daha yüksek ölçek (≈6); (ii) geç fazda (t≥0,5) APG/rescale,
  erken fazda düz CFG — pencere başına ayrı η/φ için küçük kod değişikliği; (iii) **tanh öncesi kazanç:** decoder'ın
  tanh'tan önceki Conv(→1) çıktısına ilk geçişte ölçülen tepe/RMS'ye göre g ≤ 1 çarpanı (ses seviyesini doygunlaşan
  doğrusal-olmayan katmandan önce kontrol eder, yörüngeyi değiştirmez → WER bedeli yok); (iv) {32, 64} adım × sway
  {−1, −0,5} taramasını DNSMOS/UTMOS ile de puanla; (v) üretilen latentlerin kanal başına std/basıklığını gerçek latentlerle
  karşılaştır, şişkinse örnekleme sonrası moment eşleme; (vi) best-of-N seçiciye DNSMOS/UTMOS ve SIM terimleri (SIM için
  değerlendirme modelinden **farklı** konuşmacı modeli).

### 3.10 Post-training

- **Bileşik ödüllü GRPO** [G, F5 üzerinde]: FlowTTS-GRPO WER + SIM + DNSMOS P.835 ödülüyle DNSMOS 3,15 → 3,41 (test-en),
  WER 1,88 → 1,73, SIM2 0,753 → 0,790; 1289 adım, 8 GPU, grup 10, 16 adım, 2 adımlık pencerede SDE σ=0,5. Ödül hilesi
  riski: yalnız SCOREQ ödülüyle tutulan UTMOS 4,51 → 1,23'e çöktü; WER + Distill-MOS + UTMOSv2 eşit ağırlıklı topluluk
  dayandı. 66M'de maliyet Whisper/puanlama ağırlıklı, ~1–2 GPU-gün. **Önce ücretsiz oracle testi:** DNSMOS+UTMOS ile
  best-of-8 → RL'nin ulaşabileceği üst sınır. Mevcut `posttrain.reward` DNSMOS'a çok düşük ağırlık veriyor.
- **Az adımlı çıkarım** [O]: ana eğitim düz flow matching kalmalı; sonrasında CFG-kaynaşık MeanFlow/IntMeanFlow damıtması
  32 ileri geçişi 4'e indirir (IntMeanFlow F5: 3 NFE WER 1,60 vs öğretmen 1,87). Sıfırdan MeanFlow eğitimi yavaşlatıyor.

---

## 4. Metrik bazında yol

| Hedef | Kaldıraçlar (öncelik sırasıyla) |
|---|---|
| **Ölçümün kendisi** | SIM-o + CAM++, ikinci ASR hakemi, deterministik Whisper, `turkish-v2` metrik normalizasyonu, konuşmacı-kümelenmiş GA, konuşmacı sızıntısı denetimi, Freya 8 kHz sütunu, MiniMax-Türkçe seti |
| **WER / CER** | Değerlendirme hijyeni (VAD, halüsinasyon filtresi) → artikülasyon hızı kuralı + süre-çeşitli yeniden sıralama → Türkçe normalizasyon düzeltmeleri → CTC karakter hedefleri → kuyruk-sessizlik dolgusu + kısa hedef kapsamı → speech-REPA → CPM + tempo pertürbasyonu → latent negatifler → GRPO |
| **SIM** | Doğru metrik → guidance taraması (ortak g 2–5; iç içe metin 2,5 / konuşmacı 3,5; geç/erken pencere) → TLA-SA → CPM → daha fazla konuşmacı (YODAS/CV/TSC) → farklı-model SIM terimli best-of-N → donmuş CAM++ → adaLN |
| **Ses kalitesi** | UTMOSv2/Distill-MOS/NISQA-48k + prompt DNSMOS → CFG yalnız t<0,5 → tanh öncesi kazanç → bileşik best-of-N → geç-faz APG → model-guidance ince ayarı → GRPO → Sidon restorasyonlu veri |
| **Eğitim hızı** | Senkronizasyonların kaldırılması → latent negatifler → regional compile + dolgu → checkpointing kapatma → WSD (tek koşuda aşamalı eğitim) → speech-REPA'nın yakınsama hızlandırması |

---

## 5. Veri

- **Mevcut filtre doğru seviyede** [G]: Raon-OpenTTS'te en kötü %15'i atmak WER 2,19 → 2,00 / SIM 0,661 → 0,672, %50'yi
  atmak zararlı (2,32). Bizim filtre satırların ~%25'ini atıyor; %30'u aşmayın. Emilia DNSMOS ≥3,0 filtreli ortalaması 3,26;
  bizim medyan 3,28 → veri kalitesi darboğaz değil. Ucuz ekler: Silero konuşma oranı < 0,8 filtresi, saniye başına
  karakterde 1,5×IQR aykırı değer kesimi.
- **Türkçe açık veri** (Emilia, MLS, VoxPopuli, Granary'de Türkçe yok):

| Kaynak | Saat / konuşmacı | Lisans | Not |
|---|---|---|---|
| YODAS tr000 (elle altyazı) | 588,7 h | CC-BY-3.0 | Yeniden transkript + filtre; ~150–300 h kullanılabilir tahmini |
| YODAS tr100 (otomatik altyazı) | 4.067,7 h | CC-BY-3.0 | Daha gürültülü; ikinci aşama |
| Common Voice tr (v26–27) | ~130 h doğrulanmış, ~1.840 konuşmacı | CC0 | En iyi konuşmacı çeşitliliği; DNSMOS sonrası ~50–80 h. **Freya'nın 495 cümlesi önce çıkarılmalı** (kısa-yerli bölümü CV17/CoVoST2 metinlerinden) |
| ISSAI TSC | 218 h | MIT / CC-BY-4.0 | İnsan transkripti; 94 h'lik filtreli sürüm |
| KIRAAT | 3.106 h, 90 sesli kitap konuşmacısı | **Lisans yok** | Yalnız hukuki onayla; ağırlık yayımlanmamalı |
| Yalnız değerlendirme | FLEURS-tr, MediaSpeech-tr, Antalia | CC-BY | Eğitimde kullanmayın |

  Faz 1 (CV + YODAS tr000 + TSC-94h): ~70 h → ~350–450 h, birkaç yüz → 2000+ konuşmacı; beklenen ana kazanç SIM ve
  görülmemiş konuşmacıya genelleme (ZipVoice 123M: LibriTTS 0,610 → Emilia 0,668). Önce her kaynağın %5 örneğinde filtre
  verimini ölçün.
- **Restorasyon** [O]: Sidon (MIT, 48 kHz çıkış, 104 dil): TED-LIUM'da F5 eğitimi MOS orijinal 3,25 / Demucs 3,27 /
  VoiceFixer 3,77 / **Sidon 4,25**; Türkçe FLEURS'ta DNSMOS 3,07 → 3,45, CER 0,040 → 0,041. Demucs/UVR ayrıştırması
  kazanç vermiyor — müzikli klipleri atmak daha ucuz.
- **Bant genişliği** [G]: latent model eğitim sesinin bant genişliğini üretir; MP3'lerimiz 12–16 kHz → çıktı ~16 kHz ile
  sınırlı. DNSMOS/UTMOS 16 kHz'de çalıştığı için 8 kHz üstünü göremez. Adımlar: klip başına `bandwidth_hz` ve çıktı bant
  genişliği metriği → Sidon ile tam bant hedefler → (isteğe bağlı) bant kovası embedding'i.

---

## 6. Küçük model önerisi

- **Bütçe:** 12 × 512 (66,5M) korunmalı. Kanıt: 448 → 512 (+%30) + temiz veri Freya WER'ini 9,1 → 4,3'e indirdi; DiTTo
  S (42M) → B (152M) sıçraması esas olarak SIM'de. Önerilen yeni mimari seçeneklerin toplamı <1,5M (uzun skip 0,5M, conv-FFN
  0,1M, kapı 0,1M, final adaLN 0,04–0,5M); yardımcı kayıpların projektörleri (~3–5M) **yalnız eğitimde** var, çıkarımda
  atılır.
- **Daha küçük gerekiyorsa sıra:** (1) metin kodlayıcı attention 4 → 2 (~−5M); (2) FFN çarpanı 3 → 2,5 + conv-FFN;
  (3) 12 × 448. 12 × 384 (~37M) yalnız hızlı deney/ablation modeli olarak.
- **Hız bütçesi:** kalite-nötr verimlilik işleri 60k güncellemeyi ~4–5,5 saate indirir; aynı süreye 100k+ güncelleme
  sığar. Speech-REPA'nın yakınsama hızlandırmasıyla birlikte hedef, 30–40k güncellemede mevcut 60k kalitesine ulaşmak.

---

## 7. Önceliklendirilmiş yol haritası

Ayrıntılı iş listesi, issue bağlantıları ve kabul ölçütleri: [yol-haritasi.md](yol-haritasi.md). Özet:

- **Faz 0 — Ölçüm ve temizlik:** değerlendirme protokolü v2; konuşmacı-kümelenmiş istatistik; `turkish-v2` metin
  normalizasyonu; konuşmacı sızıntısı denetimi; doküman temizliği (bu rapor).
- **Faz 1 — Hız ve ucuz kazanımlar:** eğitim hızı (senkronizasyon, regional compile, checkpointing); latent negatifler;
  süre modeli ve cümle sonu; örnekleyici/çıktı kalitesi.
- **Faz 2 — Mimari ve hedef A/B'leri:** DiT blok seçenekleri; speech-REPA + TLA-SA; eğitim çiftleri (CPM, kısa hedef,
  kuyruk sessizliği, CTC karakter hedefleri); çizelge/düzenlileştirme (WSD, uniform-*t* soğuma, dropout, çift EMA,
  model-guidance).
- **Faz 3 — Veri ve post-training:** Türkçe veri genişletme hattı; bileşik ödüllü GRPO.

Her A/B: w512 baz, 20k güncelleme (~3,2 saat), **baz 2 tohumla** (gürültü tabanını ölçmek için), tek değişken,
değerlendirme protokolü v2 ile; CER kazancı tohum yayılımından büyükse terfi.

---

## 8. Önerilmeyenler

Joint/MM-DiT attention (LARoPE'yi kaldırır, küçük ölçekte çökme kanıtı), F5-tarzı dolgu-token in-context metin (60k'da
hizalanamıyor), 25 Hz'de patch/downsampling, katman paylaşımı, differential attention, fonem/espeak ve BPE girdisi,
BERTurk/ModernBERT metin kodlayıcı, PCA/whitening/latent gürültü artırımı, latent ölçekleyerek ses seviyesi kontrolü,
x-tahmini, sıfırdan MeanFlow/shortcut eğitimi, FP8, CFG-Zero* ve başlangıç gürültüsü kısaltma, Demucs ile müzik ayrıştırma,
lisanssız veri (KIRAAT) ile yayımlanan model.

---

## 9. Yerel ölçüm arşivi (silinen belgelerden korunan kanıt)

Aşağıdaki ölçümler 18–23 Eylül'deki belgelerden (`iyilestirme-yol-haritasi.md`, `research-2026-09-21/`) aktarıldı;
ham dosyalar git geçmişinde duruyor.

**DACVAE latent sondası** (LibriSpeech test-clean, 40 konuşmacı, 234 cümle, 43.682 frame; kaynak 16 kHz → 48 kHz):

| Ölçüm | Değer |
|---|---:|
| Kanal std (min / medyan / maks) | 0,61 / 0,69 / 1,00 |
| Posterior std (medyan) | 0,0034 |
| SNR < 25 olan kanal | 0 / 128 |
| KL | ~5,5 nat/kanal, 702 nat/frame |
| PCA %90 / %95 / %99 varyans | 89 / 102 / 120 bileşen |
| Etkin rank (katılım oranı) | 77–83 / 128 |
| Lag-1 zamansal otokorelasyon (medyan) | 0,054 |

Decoder hassasiyeti (log-mel L1; orijinal ↔ codec = 0,577): posterior örnekleme 0,021; normalize uzayda izotropik hata
β=0,05/0,1/0,2/0,3/0,5 → 0,16/0,31/0,57/0,81/1,27; PCA k=96/64/32/16 → 0,67/1,12/1,70/2,26; tek kanala 1σ gürültü
0,17–0,42.

**Giriş seviyesi:** −6/−12/−18 dB kazanç → latent kosinüs 0,963/0,916/0,858, norm oranı ~0,96–1,04, codec hatası
1,06×/1,20×/1,42× → seviye latentte ölçek değil yön olarak kodlanıyor; −16 LUFS normalizasyonu şart.

**Küçük ölçekli A/B'ler (LibriSpeech, 8 saat):** EDM ön-koşullandırma her *t*'de v-tahmininden iyi (P=2'de −%5,4, P=1'de
−%1,5); P=1 farkı eğitimle açılıyor (4k'da −%10,3 → 16k'da −%11,0); LARoPE metin kazancını 2k adımda 0,0056'ya çıkarıyor
(baz 14k adıma kadar ~0,001) ama toplam flow kaybını ~%1,2 kötüleştiriyor → model seçimi flow kaybıyla yapılmamalı; metin
kodlayıcıya 2 self-attention katmanı metin kazancını 16k'da 0,0130 → 0,0183'e çıkardı. CFG'nin iki dalını tek batch'te
çalıştırmak adım süresini ~2× kısalttı (çıktı farkı 0).

**Örnekleyici dolgusu:** konuşmacının en uzun referansıyla maliyetlendirme %35,6 dolgu israfı veriyordu; gerçek epoch
maliyeti + frame bütçesiyle %0,5 (gerçek veride %23,0 → %0,33).

---

## 10. Başlıca kaynaklar

F5-TTS 2410.06885 · E2-TTS 2406.18009 · SupertonicTTS 2503.23108 · LARoPE 2509.11084 · ZipVoice 2506.13053 ·
DiTTo-TTS 2406.11427 · A-DMA 2505.19595 · RobustSpeechFlow 2605.22083 · FreyaTTS 2607.09530 · MegaTTS 3 2502.18924 ·
DMOSpeech 2410.11097 · DMOSpeech 2 2507.14988 · F5R-TTS 2504.02407 · FlowTTS-GRPO 2606.23190 · Flow-GRPO 2505.05470 ·
TLA-SA 2511.09995 · BareWave 2606.09048 · REPA 2410.06940 · HASTE 2505.16792 · ΔFM 2506.05350 · MeanFlow 2505.13447 ·
IntMeanFlow 2510.07979 · Model-guidance 2504.20334 · Selective CFG 2509.19668 · Guidance interval 2404.07724 ·
APG 2410.02416 · VoiceStar 2505.19462 · Koel-TTS 2502.05236 · Voicebox 2306.15687 · MiniMax-Speech 2505.07916 ·
OmniVoice 2604.00688 · XTTS 2406.04904 · Moonlight 2502.16982 · CMuon 2608.02502 · NorMuon 2510.05491 · WSD 2405.18392 ·
EDM2 2312.02696 · SD3 2403.03206 · DiT-Air 2503.10618 · SR-DiT 2512.12386 · LightningDiT 2501.01423 · Gated attention
2505.06708 · Value residual 2410.17897 · LongCat-AudioDiT 2603.29339 · Raon-OpenTTS 2605.20830 · Emilia 2407.05361 ·
Sidon 2509.17052 · Omnilingual ASR 2511.09690 · Whisper halüsinasyonları 2501.11378 · Seçici/hakem yanlılığı 2607.08256 ·
Echo-TTS (jordandarefsky.com/blog/2025/echo) · Irodori-TTS (github.com/Aratako/Irodori-TTS) ·
seed-tts-eval (github.com/BytedanceSpeech/seed-tts-eval) · trnorm (github.com/ysdede/trnorm).
