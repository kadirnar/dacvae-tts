# DACVAE-TTS: kod denetimi, yerel ölçümler ve iyileştirme yol haritası

**Tarih:** 21 Eylül 2026
**Kapsam:** `src/dacvae_tts` altındaki tüm kodun okunması, yerel GPU'da (RTX 5070 Ti) yapılan ölçümler ve küçük kontrollü eğitimler, güncel flow/diffusion TTS makale–blog–repo taraması.
**§1–§4'teki ölçümler sırasında model kodu değiştirilmedi.** Sonradan yapılan tek kod değişikliği (varsayılan optimizer'ın Muon olması, 22 Eylül) [Ek B](#ek-b--22-eylül-2026-varsayılan-optimizer-muon)'de kayıtlıdır. Ölçüm betikleri ve ham sonuçlar [`research-2026-09-21/`](research-2026-09-21/) klasöründedir.

Önceki belgeler ([kapsamlı rapor](kapsamli-tts-arastirma-raporu.md), [mimari denetimi](architecture-audit.md), [deney kapıları](experiments.md)) doğruluk/maskeleme/AdaLN alternatifleri/ref-text konularını zaten işliyor. Bu belge onları tekrar etmez; **orada olmayan** bulgulara ve önceliklendirmeye odaklanır.

## Kanıt etiketleri

| Etiket | Anlamı |
|---|---|
| **[K]** | Depodaki kodda doğrulandı (`dosya:satır`) |
| **[Ö]** | Bu çalışmada yapılan yerel ölçüm; betik + JSON `research-2026-09-21/` içinde |
| **[L]** | Makale tablosunda / kaynak kodunda doğrulanmış yayımlanmış sonuç. ✔ işaretliler bu oturumda kaynağından ikinci kez teyit edildi |
| **[Y]** | Yazar iddiası; kontrollü ablation yok |
| **[H]** | Hipotez / çıkarım; bizim modelde gösterilmedi |

Yerel eğitim deneyleri **oyuncak ölçeklidir** (8,8 saat LibriSpeech, birkaç bin güncelleme, metrik = görülmemiş konuşmacılarda flow hatası). Konuşma kalitesi, WER veya klonlama iddiası **değildir**; yalnız "aynı bütçede hangi ayar daha hızlı öğreniyor" sorusuna ışık tutar.

---

## 0. Yönetici özeti

Önem sırasıyla:

1. **En büyük risk kapasite/temsil uyumsuzluğu.** Gerçek konuşmada ölçtüm: DACVAE latenti neredeyse *beyaz gürültü kadar yoğun* — 128 kanalın hepsi aktif, PCA'da %90 varyans için 89 bileşen gerekiyor, ardışık frame korelasyonu medyan **0,054** [Ö]. P=2 paketleme bu yüzden hiçbir artıklık sömüremiyor ve 256 genişlikli Tiny'ye 256 boyutlu yoğun token tahmin ettiriyor (genişlik/token oranı **1,0**). İncelenen tüm DACVAE-ailesi sistemler patch=1 kullanıyor ve oranları 10–36; 128-boyutlu DACVAE üzerinde belgelenmiş en küçük çalışan model 500M [L]. Yerel A/B: **P=1, her gürültü seviyesinde daha iyi; P=1'in 1000. adımı P=2'nin 4000. adımını geçiyor** [Ö].
2. **Hizalama mekanizması en zayıf halka.** Yalnız-konvolüsyonlu byte encoder (alım alanı 19–25 byte) + metin/ses arasında hiçbir konumsal bağ içermeyen cross-attention [K]. Aynı mimari ailesi, aynı veri ölçeğinde (0,06B, ~10k saat, 5M cümle) Seed-TTS test-en WER **1,44** alıyor; ama üç farkla: cross-attention'da **LARoPE**, self-attention'lı text encoder, **context-sharing batch expansion** [L ✔]. Yerel deney: mevcut model 14k adım boyunca metni hiç kullanmıyor (karıştırılmış metin kaybı değiştirmiyor); LARoPE ile metin kullanımı 2k adımda başlıyor ve ~10× güçlü [Ö].
3. **Eğitim veriminde ~5× bedava kazanç duruyor** [Ö]: örnekleyici batch'leri yanlış maliyete göre sıralıyor (simülasyonda **%35,6 padding → %0,4**), yapılandırılmış mikro-batch GPU'nun ~%22–31'ini kullanıyor (B=64 + `torch.compile` → **4,2×**), text/referans encoder her adımda iki kez çalışıyor (**−%8–10**).
4. **Inference'ta 2× bedava kazanç** [Ö]: CFG'nin iki dalı ayrı forward yerine tek batch'te → **2,1× hızlı, bit-düzeyinde aynı**. Model tamamen kernel-launch'a bağlı (B=1 ile B=16 aynı süre) → CUDA graph ile daha da fazlası.
5. **Flow hedefi:** uniform *t* yerine (stratified) logit-normal — ailedeki herkes böyle [L ✔]. x-tahmini (JiT) burada **net kazanç değil** (yerel A/B: gürültülü uçta iyi, temiz uçta belirgin kötü); ikisinin iyi yanlarını birleştiren **EDM-tarzı ön-koşullandırma her *t*'de v-tahmininden iyi** çıktı [Ö]. P=1 + EDM + logit-normal toplanabilir: bu üçüyle **13,4M'lik Tiny, aynı adım sayısında 44M'lik mevcut Small'u geçiyor** (görülmemiş konuşmacı flow hatası 1,305 ↔ 1,394; oyuncak ölçek) [Ö].
6. **Veri hattı:** DACVAE'nin kendi API'si girişi −16 dB'ye normalize ediyor, bizim hat etmiyor [K]; −6 dB seviye farkı latenti **%27** değiştiriyor, −18 dB'de codec hatası **1,42×** [Ö]. Büyük encode'dan önce düzeltilmeli (sonradan değiştirmek 4M satırı yeniden encode etmek demek).
7. **Değerlendirme literatürle karşılaştırılamaz durumda:** SIM modeli (`wavlm-base-plus-sv`) ve WER normalizasyonu standart değil; UTMOS ve standart test setleri yok [K]. Standart SIM checkpoint'i (`wavlm_large_finetune.pth`) ve Seed-TTS test-en seti bu makinede zaten cache'li.
8. **Post-training ayarları literatürden uzak:** DPO β=10 (yayımlanmış değerler 2000–5000); distillation hedefi aynı `(x_t, t)` için çelişen iki hedef öğretiyor [K+L].
9. **Bütçe gözlemi:** ölçülen hızla planlanan Tiny koşusu 8 GPU'da saat mertebesinde, Small birkaç saat [Ö, tek GPU'dan ekstrapolasyon]. Yani hesap bütçesi, literatürün kanıt sunduğu bölgede (≥150–300M, genişlik ≥768) bir **Base** yapılandırmasını rahatça kaldırır; bu model ileride Tiny/Small için öğretmen de olur.

---

## 1. Yerel ölçümler (yeni kanıt)

### 1.1 DACVAE latent sondası — gerçek konuşma

LibriSpeech test-clean, 40 konuşmacı, 234 cümle (3–15 s), 43.682 frame; deponun kendi `Codec`/`read_audio` yolu, FP32. **Sınır:** kaynak 16 kHz, 48 kHz'e yükseltildi; tam bant konuşmada yüksek frekans kanalları farklı davranabilir.

| Ölçüm | Değer | Yorum |
|---|---:|---|
| Kanal std (min / medyan / maks) | 0,61 / 0,69 / 1,00 | Kanallar zaten benzer ölçekte |
| Posterior std (medyan) | 0,0034 | VAE fiilen deterministik; mean/sample farkı önemsiz |
| SNR < 25 olan kanal sayısı | **0 / 128** | "Kullanılmayan kanal" yok (ilk hipotezim çürüdü) |
| KL | ~5,5 nat/kanal, **702 nat/frame** | Çok yüksek bilgi yoğunluğu |
| PCA: %90 / %95 / %99 varyans | 89 / 102 / 120 bileşen | Doğrusal düşük-boyutlu alt uzay yok |
| Katılım oranı (efektif rank) | 77–83 / 128 | Neredeyse izotropik |
| Lag-1 zamansal otokorelasyon (medyan) | **0,054** | Ardışık frame'ler doğrusal olarak ilişkisiz |

Decoder hassasiyeti (log-mel L1 mesafesi; ölçek için *orijinal ↔ codec reconstruction* = **0,577**):

| Müdahale | Mesafe |
|---|---:|
| Posterior örnekleme (σ≈0,003) | 0,021 |
| Normalize uzayda izotropik hata β=0,05 / 0,1 / 0,2 / 0,3 / 0,5 | 0,16 / 0,31 / **0,57** / 0,81 / 1,27 |
| PCA kırpma k=96 / 64 / 32 / 16 | 0,67 / 1,12 / 1,70 / 2,26 |
| Tek kanala 1σ gürültü (min / medyan / maks) | 0,17 / 0,22 / 0,42 |

Çıkarımlar:

- **Kanal-bazlı normalizasyon zararsız ama önemsiz**: kanal std'leri zaten 0,6–1,0 aralığında. Echo'nun "kanal-bazlı normalizasyon zarar verdi" gözlemi [Y ✔] PCA ile küçültülmüş, varyansları çok farklı başka bir latente ait; buraya taşınmaz. Düşük öncelik.
- **PCA ile boyut azaltma uygulanabilir değil**: 96 bileşen bile codec'in kendi hatasından büyük bozulma veriyor. Saf rotasyon da flow-matching hedefini değiştirmez [H: izotropik gürültü + MSE rotasyona simetrik].
- **Kalite hedefi zorlu**: üreticinin latent hatası codec'in kendi hatası mertebesinde kalsın istiyorsak normalize uzayda kanal başına göreli hata ≈ 0,2 (x-MSE ≈ 0,04) gerekiyor. Decoder, girişindeki gürültüye dayanıklı eğitilmemiş (σ=0,003).
- **P=2 paketleme yanlış araç**: komşu frame'ler ilişkisizken iki frame'i tek tokena koymak sıkıştırma değil, yalnız token başına tahmin yükünü ikiye katlamak.

### 1.2 Giriş seviyesine duyarlılık

Aynı 24 cümle farklı kazançla encode edildi:

| Kazanç | Latent göreli değişim | Kosinüs | Codec gidiş-dönüş hatası (0 dB'ye oranla) |
|---:|---:|---:|---:|
| −6 dB | %27 | 0,963 | 1,06× |
| −12 dB | %40 | 0,916 | 1,20× |
| −18 dB | %52 | 0,858 | **1,42×** |
| +6 dB | %30 | 0,958 | 1,00× |

Latent normu neredeyse sabit (0,96–1,04): seviye, latentte *ölçek* olarak değil dağınık biçimde kodlanıyor. Yani kayıt seviyesi üreticinin modellemek zorunda kaldığı bir rahatsızlık değişkeni; referans ile hedef farklı seviyedeyse model hangisini üreteceğini bilemez.

### 1.3 Örnekleyici padding simülasyonu

Deponun gerçek `BucketBatchSampler`'ı, sentetik korpus (320k cümle, 20k konuşmacı, 1–15 s; yalnız göreli karşılaştırma anlamlı):

| Politika | Padding israfı | Denetlenen (hedef) frame / toplam padded hesap | Batch başına padded frame (p5–p95) |
|---|---:|---:|---|
| **Mevcut:** `hedef + konuşmacının en uzun referansı`, sabit 16 | **%35,6** | 0,32 | 5.200–11.520 |
| Gerçek epoch maliyeti `hedef + o epoch'ta seçilen referans`, sabit 16 | **%0,4** | 0,50 | 1.978–9.120 |
| Gerçek maliyet + frame bütçesi 6000 (üst sınır 64) | %0,5 | 0,50 | **5.544–5.985** |

Gerçek veriyle teyit: A/B deneylerinin LibriSpeech cache'inde (3.416 satır, 51 konuşmacı) aynı karşılaştırma **%23,0 → %0,33** padding israfı ve 0,38 → 0,50 denetlenen oran veriyor.

Neden [K]: `data.py:119-123` maliyeti konuşmacı grubunun *en uzun* cümlesiyle hesaplıyor; çok cümleli konuşmacılarda bu hep ~15 s olduğundan sıralama fiilen yalnız hedef uzunluğuna göre yapılıyor, gerçek referans uzunluğu ise 1–15 s arasında serbest. Referans seçimi `(seed, epoch, index)`'in deterministik fonksiyonu (`data.py:173-176`), dolayısıyla örnekleyici gerçek maliyeti önceden hesaplayabilir.

### 1.4 GPU mikro-benchmark (RTX 5070 Ti, BF16, ref 6 s + hedef 10 s)

| Ölçüm | Tiny | Small |
|---|---:|---:|
| CFG adımı: iki ayrı forward (mevcut) | 11,6 ms | 16,1 ms |
| CFG adımı: tek batch'li forward (B=2) | **5,5 ms** | **8,2 ms** |
| İki yolun çıktı farkı (maks. mutlak) | 0,0 | 0,0 |
| Forward B=1 / B=16 | 5,5 / 5,6 ms | 8,1 / 8,2 ms |
| Forward, referans prefix'li / prefix'siz (B=16) | 5,6 / 5,7 ms | 8,2 / 8,4 ms |
| Eğitim adımı B=16: mevcut / koşullar tek sefer | 30,7 / **27,5 ms** | 44,1 / **40,5 ms** |

B=1 ile B=16'nın aynı sürmesi modelin bu boyutlarda **kernel-launch'a bağlı** olduğunu gösteriyor: GPU boş bekliyor.

Eğitim verimi (hedef ses saniyesi / saniye, tek GPU):

| Model | P | B=16 | B=32 | B=64 | B=128 | B=64 + compile | B=128 + compile |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tiny | 2 | 5.251 | 10.464 | 16.526 | 16.759 | 22.148 | **23.923** |
| Tiny | 1 | 5.178 | 8.865 | 9.564 | 9.051 | 13.202 | 13.603 |
| Small | 2 | 3.570 | 6.417 | 7.016 | 6.712 | – | – |
| Small | 1 | 3.300 | 3.839 | 3.832 | OOM (16 GB) | – | – |

- `configs/tiny.yaml` (B=16×2 birikim) eager tepe verimin ~%31'inde, compile tepe verimin ~%22'sinde; `configs/small.yaml` (B=8×4, üstelik gradient checkpointing açık) daha da düşükte.
- P=1'in gerçek bedeli doygun batch'te ~1,75× hesap (Tiny); mevcut B=16'da ise **sıfır** (30,5 ↔ 30,9 ms).
- Small B=16 eğitim belleği 1,8 GB (P=2) / 2,7 GB (P=1): 24–80 GB kartlarda gradient checkpointing'e gerek yok.

**Bütçe ekstrapolasyonu [Ö→H]:** Tiny planı 200k adım × 256 çift ≈ 51M çift. Ortalama hedef ~7 s varsayımıyla ≈ 360M hedef-saniye; P=1 + compile ile GPU başına ~13,6k s/s → ~7 GPU-saat, ideal ölçeklemeyle 8 GPU'da **~1 saat**. Small (P=1, 300k adım) ≈ 30–40 GPU-saat. Bunlar tek 5070 Ti ölçümünden ekstrapolasyondur; 8 GPU ölçeklemesi ve gerçek korpusta veri yükleme hızı ölçülmedi. Sonuç yine de açık: **darboğaz hesap değil**; aynı takvimde 10–20× daha büyük bir model eğitilebilir.

### 1.5 Küçük ölçekli kontrollü eğitimler (A/B)

Kurulum: deponun `LatentDataset` / `BucketBatchSampler` / `collate` / `FlowTTS` bileşenleri değiştirilmeden; yalnız kayıp sarmalayıcısı varyanta göre değişiyor. Eğitim 3.416 cümle / 51 konuşmacı / 8,06 saat; doğrulama **görülmemiş 8 konuşmacı**, 413 cümle. AdamW 3e-4 (Small 2e-4), B=32 çift, BF16, tohum 42. Metrik: sabit *t* ∈ {0,05 … 0,95} ızgarası ve sabit gürültüyle hedef frame'lerde hız MSE'si (v-MSE) ve ima edilen temiz-latent MSE'si (x-MSE); tüm parametrizasyonlar iki niceliğe de çevrildiği için satırlar karşılaştırılabilir. `base` iki ayrı koşuda bit-düzeyinde aynı sonucu verdi (1,4973). Protokol ayrıntısı: [`research-2026-09-21/README.md`](research-2026-09-21/README.md).

**4000 güncelleme sonunda (düşük = iyi):**

| Model | Varyant | Tahmin | *t* | P | v-MSE | x-MSE | v-MSE @ t=0,05 / 0,45 / 0,95 |
|---|---|---|---|---:|---:|---:|---|
| Tiny 13,5M | **base (mevcut)** | v | uniform | 2 | 1,4973 | 0,4971 | 1,310 / 1,674 / 1,251 |
| Tiny | lognorm | v | logit-N | 2 | 1,5106 | 0,5008 | 1,361 / 1,632 / 1,345 |
| Tiny | xpred | x | uniform | 2 | 2,0000 | 0,4763 | 1,167 / 1,685 / **6,049** |
| Tiny | edm | EDM | uniform | 2 | 1,4165 | 0,4658 | 1,160 / 1,641 / 1,099 |
| Tiny | p1 | v | uniform | 1 | 1,3434 | 0,4477 | 1,166 / 1,447 / 1,119 |
| Tiny | xpred_p1 | x | uniform | 1 | 1,3939 | 0,4545 | 1,162 / 1,512 / 1,282 |
| Tiny | p1_lognorm | v | logit-N | 1 | 1,3317 | 0,4443 | 1,175 / 1,408 / 1,161 |
| Tiny | edm_p1 | EDM | uniform | 1 | 1,3228 | 0,4425 | 1,159 / 1,423 / 1,095 |
| Tiny | **edm_p1_lognorm** | EDM | logit-N | 1 | **1,3048** | **0,4382** | 1,160 / 1,384 / 1,095 |
| Small 44M | base (mevcut) | v | uniform | 2 | 1,3943 | 0,4606 | 1,173 / 1,562 / 1,135 |
| Small | p1 | v | uniform | 1 | 1,3135 | 0,4406 | 1,165 / 1,387 / 1,114 |
| Small | edm_p1 | EDM | uniform | 1 | **1,2971** | **0,4365** | 1,159 / 1,374 / 1,093 |

Karşılaştırma ölçeği: veriyi yapısız birim-varyanslı Gauss sayan en iyi *doğrusal* kestiricinin v-MSE'si t=0,05 / 0,45 / 0,95'te 1,105 / 1,980 / 1,105'tir. `base` uçlarda bu önemsiz çözümden bile kötü (1,310 / 1,251): ağ, tam-rank 256 boyutlu girdiyi 256 genişlikli artık akışından "kimlik" olarak geçirmekte zorlanıyor — token boyutu ≈ genişlik darboğazının doğrudan imzası. P=1 ve EDM ön-koşullandırma tam da bu uçları düzeltiyor.

Okuma:

1. **P=1 en büyük tekil kazanç** (Tiny −%10,3; Small −%5,8; Small'da EDM ile birlikte −%7,0) ve her *t* değerinde geçerli. Adım verimi: `p1` 1000. adımda 1,454 ile `base`'in 4000. adımdaki 1,497'sini geçiyor.
2. **EDM ön-koşullandırma her *t*'de v-tahmininden iyi** (P=2'de −%5,4; P=1'de −%1,5). Kazanç uçlarda yoğunlaşıyor. x-tahmini ise temiz uçta bozuluyor: ağ kimlik eşlemesini öğrenmek zorunda kaldığı için *t*=0,95'te x-MSE 0,0151 (v-tahmini 0,0031); §1.1'e göre bu, son adımlarda duyulur ince-ayrıntı gürültüsü demek.
3. **Logit-normal**: P=2'de toplamda hafif kötü, P=1'de hafif iyi; her iki durumda orta-*t* kaybını düşürüyor, uçları bir miktar kötüleştiriyor — tasarım gereği. Bu metrik karar için yeterli değil.
4. **Üçü toplanabilir:** `edm_p1_lognorm` −%12,9. **13,4M'lik Tiny bu üç değişiklikle (1,305), aynı adım sayısında 44M'lik mevcut Small'u (1,394) ve Small-P1'i (1,314) geçiyor.**

**Koşullandırma-kullanım teşhisi** (orta-*t* v-MSE farkı; 4000 adım, Small base / p1 / edm_p1): doğru metin yerine *başka cümlenin metni* verilince kayıp yalnız +0,0005 / +0,0009 / +0,0007 artıyor → **model metni henüz hiç kullanmıyor**; *başka konuşmacının prompt'u* verilince +0,018 / +0,017 / +0,014 → konuşmacı koşulu kullanılıyor. Metni *sıfırlamak* ise +0,02–0,03 "kazanç" gösteriyor ama bu, eğitimde hiç görülmeyen bir girdi kombinasyonunun (yalnız-metin düşürme) yapay etkisi; karıştırma tabanlı teşhis doğru olanı. Literatürle uyumlu: önce tını oturur, hizalama geç gelir (F5-small: WER 100k adımda 29,5 → 200k'da 4,6) [L].

**Uzun koşu (16.000 güncelleme, ~149 epoch; Tiny, v-tahmini, uniform *t*):**

| Adım | P=2 v-MSE | P=1 v-MSE | P=1 + LARoPE v-MSE | metin kazancı: P=1 / +LARoPE | konuşmacı kazancı: P=1 / +LARoPE |
|---:|---:|---:|---:|---|---|
| 2.000 | 1,554 | 1,413 | 1,409 | 0,0008 / **0,0056** | 0,020 / 0,020 |
| 4.000 | 1,497 | 1,352 | 1,353 | 0,0009 / **0,0089** | 0,023 / 0,022 |
| 8.000 | 1,441 | 1,293 | 1,301 | 0,0010 / **0,0108** | 0,021 / 0,023 |
| 12.000 | 1,406 | 1,251 | 1,279 | 0,0011 / **0,0129** | 0,024 / 0,024 |
| 16.000 | **1,382** | **1,230** | 1,245 | 0,0019 / **0,0130** | 0,024 / 0,025 |

("Kazanç" = karıştırılmış koşulla kayıp − doğru koşulla kayıp, *t* ∈ {0,25; 0,4; 0,55; 0,7}. P=2 koşusunun 16k sonundaki metin kazancı 0,0008. P=1 sütunları LARoPE betiğinin kendi baseline'ıdır; başlangıç ağırlıkları §1.5 tablosundaki `p1` koşusundan farklı olduğundan 4k değeri 1,352 ↔ 1,343.)

5. **P=1 farkı eğitimle kapanmıyor, açılıyor** (4k'da −%10,3 → 16k'da −%11,0). P=1'in 4.000. adımı (1,34–1,35), P=2'nin 16.000. adımından (1,382) iyi: doygun batch'teki ~1,75× adım maliyeti düşüldükten sonra bile ≥2× hesap verimi. Doğrulama kaybı 149 epoch boyunca düşmeye devam etti; bu ölçekte aşırı uyum işareti yok.
6. **LARoPE metin kullanımını baştan açıyor.** Cross-attention'a segment-normalize LARoPE (γ=10; referans frame'leri ↔ referans metni [0,1), hedef ↔ hedef metni [1,2)) ekleyen scratch alt-sınıfta metin kazancı 2k adımda 0,0056, 16k'da 0,013; baseline 14k adıma kadar ~0,001'de düz, ancak 16k'da kıpırdamaya başlıyor (0,0019). Yani hizalama baseline'da da sonunda gelecek gibi, ama LARoPE ile **~10× daha erken ve güçlü**. Konuşmacı kazancı iki modelde aynı.
7. **Ama toplam flow kaybı LARoPE ile ~%1,2 daha kötü** (16k: 1,245 ↔ 1,230). Flow kaybını akustik ayrıntı domine ediyor, içerik değil; bu yüzden bu fark WER hakkında bir şey söylemiyor — depodaki "modeli flow kaybıyla seçme" uyarısının somut örneği. Olası neden [H]: tüm head'lerde rotary, cross-attention'ı konumdan bağımsız "genel bağlam" yolu olarak kullanmayı zorlaştırıyor; Echo RoPE'u head'lerin yalnız yarısına uyguluyor [L]. Denenecek: rotary'yi head'lerin yarısına uygulamak.
8. **Text encoder'a 2 self-attention katmanı** (+1,05M parametre) LARoPE'nin üstüne metin kazancını her adımda artırıyor: 8k'da 0,0108 → 0,0128, 16k'da 0,0130 → **0,0183** (baseline 0,0019); konuşmacı kazancı da en yüksek (0,026). Toplam kayıp LARoPE ile aynı (1,244), baseline'ın ~%1,2 gerisinde.

Bu üç gözlem birlikte A5 önerisini yerel olarak destekliyor, fakat nihai kanıt değil: karar WER ile verilmeli ve bunun için §F'deki değerlendirme düzeneği + C5'teki eğitim-içi sentez gerekiyor.

### 1.6 Veri yükleyici üst sınırı

Mevcut `LatentDataset` + `collate` yolu, tamamı sayfa önbelleğinde duran küçük cache üzerinde (yani **en iyi durum**), B=64:

| Worker | 0 | 1 | 2 | 4 |
|---|---:|---:|---:|---:|
| çift / s | 1.646 | 1.526 | 2.602 | 3.591 |

GPU'nun tam verimde tükettiği hız (§1.4, hedef ~10 s): Tiny P=2 + compile ≈ 2.400 çift/s, Tiny P=1 + compile ≈ 1.360, Small P=1 ≈ 380. Yani `train_8gpu.sh`'ın varsayılan 2 worker'ı Tiny için en iyi durumda bile sınırda. Gerçek korpusta (≈256 GB, rastgele erişim) 8 GPU × ~2.000 çift/s × çift başına ~130–250 KB ≈ **2–4 GB/s rastgele okuma** gerekir; tek bir NVMe'nin soğuk rastgele okuma hızının üstünde. Seçenekler: (i) batch expansion (B3) çift/s ihtiyacını Ke kat düşürür; (ii) cache'i RAM'e sığdırmak (256 GB çoğu 8-GPU sunucuda mümkün); (iii) shard-yerel karıştırma. Bu, büyük koşudan önce gerçek diskte ölçülmeli.

### 1.7 Hazırlık (encode) hızı ve 4M satır için süre tahmini

Deponun `prepare` komutu, tek RTX 5070 Ti, katı FP32, 6 CPU thread, LibriSpeech (12,7 saat giriş, 8,8 saati kabul edildi): **220× gerçek zaman**. Sürenin %95'i GPU'daki encoder (137 / 144 s); CPU'dan veri bekleme 6 s.

| `--bucket-size` | encoder çağrısı | çağrı başına kayıt | hız |
|---:|---:|---:|---:|
| 256 (varsayılan) | 2.449 | 1,6 | 220× |
| 2.048 | 880 | 4,4 | 224× |
| 8.192 | 477 | 8,0 | 228× |

Eşit-uzunluk şartı yüzünden varsayılan pencerede batch'ler fiilen tek kayıtlık; ama pencereyi büyütmek de hızı değiştirmiyor: 48 kHz'lik konvolüsyonel encoder tek kayıtta bile GPU'yu dolduruyor. Yani bu aşama hesap-sınırlı; kazanç yalnız daha hızlı GPU, `--codec-compile` (önceki yerel ölçüm ~1,17×) veya daha çok GPU'dan gelir.

4M satırın kaç saat ses ettiği ortalama süreye bağlı (bilinmiyor; `audit-cache` raporundaki `hours` alanı kesin değeri verir):

| Ortalama klip | Toplam ses | Encode: 1 GPU | Encode: 8 GPU (ideal) | Latent cache |
|---:|---:|---:|---:|---:|
| 5 s | 5.556 saat | ~25 saat | ~3,2 saat | ~128 GB |
| 8,3 s (LibriSpeech örneklemindeki ortalama) | ~9.200 saat | ~42 saat | ~5,2 saat | ~212 GB |
| 10 s | 11.111 saat | ~50 saat | ~6,3 saat | ~256 GB |

Eğitim süresi için §1.4'teki verimler geçerli (tek 5070 Ti ölçümünden ekstrapolasyon; 8 GPU ölçeklemesi ve disk hızı ölçülmedi). 4M satır üzerinden **bir epoch**, ortalama 8,3 s klip ve aynı uzunlukta referansla ≈ 1,66 milyar frame:

| Ayar | GPU başına frame/s | 1 epoch, 8 GPU | YAML planı (Tiny 200k / Small 300k adım × 256 çift), 8 GPU |
|---|---:|---:|---:|
| Tiny, mevcut config (B=16, %36 padding israfı) | ~135k geçerli | ~26 dk | ~5,5 saat |
| Tiny P=2, B=128 + compile + düzeltilmiş örnekleyici | ~950k | ~3,6 dk | ~47 dk |
| Tiny P=1, B=128 + compile + düzeltilmiş örnekleyici | ~540k | ~6,4 dk | ~1,4 saat |
| Small P=2, B=64 | ~280k | ~12 dk | ~4 saat |
| Small P=1, B=32 | ~153k | ~23 dk | ~7,2 saat |

Hızlı ayarlarda gerçek sınır büyük olasılıkla veri yükleme olacak (§1.6).

---

## 2. Bulgular ve öneriler

Her madde: **sorun → kanıt → öneri → maliyet/risk → nasıl ölçülür**.

### A. Mimari ve kapasite

#### A1. Paketlemeyi kaldır: `patch_size: 1` varsayılan olsun
- **Sorun [K]:** `configs/tiny.yaml`, `configs/small.yaml` `patch_size: 2`; `model.py:110` giriş `(2C+1)·P → D`, çıkış `D → C·P`. Tiny'de token boyutu = genişlik = 256.
- **Kanıt:** §1.1 (latent beyaz; komşu frame korelasyonu ~0,05) ve §1.5 (P=1 her *t*'de iyi) [Ö]. Echo, Irodori, LAION DACVAE-Echo, SAM Audio, MOSS-SoundEffect: hepsi patch=1 [L]. RAE Teorem 1: v-tahmininde genişlik *d* < token boyutu *n* ise kayıp ≥ Σ_{i>d} λ_i; genişlik 384 olan DiT-S, 768-boyutlu tokenlarla tek bir görüntüyü bile ezberleyemiyor, derinliği ikiye katlamak yardım etmiyor [L]. JiT Tablo 2: 768-boyutlu token / genişlik 768'de v-tahmini FID 96,5, x-tahmini 8,6 [L ✔].
- **Öneri:** P=1'i Tiny/Small varsayılanı yap; 25 Hz'de 30 s = 750 token, attention için önemsiz.
- **Maliyet:** doygun batch'te ~1,75× hesap, ~+%50 aktivasyon belleği. §1.4'teki bütçe bunu rahatça karşılıyor.
- **Ölçüm:** mevcut `tiny_packing_one.yaml` ile eşleştirilmiş koşu; WER/SIM + sabit *t*-ızgarasında doğrulama kaybı.

#### A2. Genişlik/token oranını yükselt; bir "Base" yapılandırması ekle
- **Kanıt:** Ailede genişlik/token-boyutu oranı 10–36; 128-boyutlu DACVAE-tipi latentte belgelenmiş en küçük çalışan TTS 500M / genişlik 1280 (Irodori v1 — yazarı sonra 128-boyutu bıraktı) [L]. Text-Audiobox dublaj ablation'ı: 300M WER %44,5, 1B %23,0, 3B %14,0 (bu ablation'ın eğitim bütçesi teyit edilemedi) [L]. Tek küçük başarı örneği SupertonicTTS (oran ~1,8) ama amaca özel 24-boyutlu, pürüzsüz bir latent üzerinde [L ✔].
- **Öneri:** Tiny/Small ürün hedefi olarak kalsın; yanına **Base (≈150–300M, genişlik ≥768, P=1)** ekle. İki işlevi var: (a) literatürün kanıt sunduğu bölgede bir çapa — Base de anlaşılır konuşma üretemezse sorun küçük modelde değil latenttedir; (b) Tiny/Small için distillation öğretmeni.
- **Risk:** Base bile doğrulanmış aralığın (≥500M) altında.

#### A3. AdaLN parametrelerini (%35) genişliğe çevir
- Önceki raporun §8.9'u seçenekleri sayıyor; yeni kanıt: DiTTo'da *global* AdaLN, blok-başına olandan hem %30 küçük hem daha iyi (WER 2,93 vs 3,38); EzAudio'da saf AdaLN-single 128-kanallı latentte çöktü, paylaşımlı + düşük-rank (rank 32) kararlı; Echo/Irodori düşük-rank AdaLN kullanıyor [L].
- **Öneri:** paylaşımlı modülasyon MLP'si + blok başına düşük-rank (r=32–64) düzeltme; boşalan ~%30 parametreyle eşit bütçede Tiny genişliği 256 → ~320–352. Saf AdaLN-single **kullanma**.

#### A4. RoPE + QK-norm (sonra RMSNorm, SwiGLU, bias'sız)
- **Sorun [K]:** `model.py:207` mutlak sinüzoidal konum yalnız girişte ekleniyor; hedefin konumu prefix uzunluğuna bağlı; `Attention` içinde QK-norm yok.
- **Kanıt [L]:** Ailede RoPE + QK-norm evrensel. LightningDiT ablation: SwiGLU 12,5→10,1, RMSNorm →9,25, RoPE →**7,13** FID. SD3: QK-RMSNorm BF16'da attention-logit ıraksamasını gideriyor.
- **Ölçüm:** kümülatif ekle; eğitimde görülenden uzun cümlelerde WER.

#### A5. Text encoder'a self-attention; cross-attention'a LARoPE
- **Sorun [K]:** `model.py:47-69` yalnız depthwise conv (k=7, 3–4 katman → alım alanı 19–25 byte), global bağlam yok. `model.py:94` cross-attention'da metin ve ses konumları arasında hiçbir ilişki kodlanmıyor; model "hedef frame *i/N* ↔ byte *j/M*" köşegenini sıfırdan keşfetmek zorunda.
- **Kanıt [L ✔]:** LARoPE (cross-attention'da RoPE açısı `γ·p/L`, γ=10; sorgu ve anahtar kendi uzunluklarıyla normalize): 19M model, 200k iterasyonda CER 2,00→**1,23**; 10–30 s cümlelerde WER 4,98→**2,16**; süre ölçekleme altında daha dayanıklı. Parametre maliyeti sıfır. Aynı grubun 0,06B modeli (~10k saat, 5M cümle, ham metin, cross-attention + LARoPE + batch expansion ×6, ayrı eğitilmiş cümle-düzeyi duration) Seed-TTS test-en WER **1,44** / SIM 0,60. ZipVoice'ta text encoder'ı kaldırmak WER 1,69→2,04; tek-biçimli hizalama önceliğini (average upsampling) kaldırmak **1,69→20,19** [L ✔]. Yayımlanmış hiçbir cross-attention NAR sisteminde yalnız-conv text encoder yok [H].
- **Yerel kanıt [Ö]:** §1.5 — segment-normalize LARoPE ile metin kullanımı ~10× daha erken başlıyor; 2 self-attention katmanı bunu biraz daha artırıyor. Toplam flow kaybı ~%1 geride kaldığı için nihai karar WER ile verilmeli.
- **Öneri:** (i) conv'lardan sonra 2–4 RoPE'li self-attention bloğu; (ii) cross-attention'da LARoPE. Bizim [referans ‖ hedef] düzeninde segment başına normalizasyon: referans frame'leri ↔ referans metni [0,1), hedef frame'leri ↔ hedef metni [1,2) — çalışan scratch uygulaması: [`research-2026-09-21/scripts/ab_larope.py`](research-2026-09-21/scripts/ab_larope.py). Rotary'yi head'lerin yarısına uygulama varyantını da dene [H].
- **Önceki raporla ilişki:** oradaki T2 ("öncelikli kalite hipotezi") artık kendi ölçeğimizde yayımlanmış sayısal destekle geliyor; LARoPE ise orada yok.

#### A6. Referans yolu: önbelleklenebilir prefix veya ayrı referans encoder
- **Sorun [K+Ö]:** Geçerli frame'lerin %50'si referans (§1.3); inference'ta prefix her adımda ve her CFG dalında yeniden işleniyor; prefix tokenları AdaLN(zaman) alıp gürültülü hedefe baktığından K/V önbelleği matematiksel olarak geçersiz.
- **Seçenekler:** (i) prefix tokenlarına zamandan bağımsız modülasyon + prefix→yalnız-prefix attention maskesi → K/V adımlar arasında ve batch-expansion çekimleri arasında paylaşılır; (ii) Echo/Irodori tarzı ayrı referans encoder + K/V birleştirme: bir kez çalışır, referans transkripti gerekmez, referans serbestçe kırpılıp birleştirilebilir (Irodori SIM: tek klip 0,661 → 30 s 0,752 → 120 s 0,775) [L].
- **Korunacak artı:** Mevcut "farklı cümle prefix + yalnız hedefte kayıp" kurulumu eğitim/inference uyumlu; Text-Audiobox tam da bu kuruluma ulaşmak için ayrı bir fine-tuning aşaması eklemiş [L].
- **Öncelik:** A1/A5/B1'den sonra.

#### A7. Inference uzunluk koruması (gizli hata riski)
- **Sorun [K]:** `inference.py:159` referans ≤30 s, `inference.py:190` hedef ≤30 s → 1500 frame'e kadar; eğitimde toplam en fazla 750 frame (2×15 s) ve konumlar mutlak sinüzoidal. İkinci yarı hiç görülmemiş konumlarda üretilir.
- **Öneri:** RoPE/LARoPE doğrulanana kadar `referans + hedef ≤ eğitimde görülen maksimum` kısıtı; uzun metin için cümle bazlı parçalama + 0,1–0,15 s çapraz geçiş (F5/ZipVoice pratiği) [L].

### B. Eğitim hedefi

#### B1. Zaman örneklemesi: stratified logit-normal
- **Sorun [K]:** `model.py:255` `torch.rand(b)`.
- **Kanıt [L ✔]:** Echo, Irodori, LAION, Meta (Movie Gen/SAM Audio/Text-Audiobox), ACE-Step logit-normal(0,1) kullanıyor; SD3'te 61 varyant arasında ortalama sıra 1,54 vs uniform 5,67; LightningDiT FID 16,6→14,0.
- **Yerel not [Ö]:** uniform *t*-ızgaralı doğrulama kaybında etki küçük ve yönü karışık: P=2'de hafif kötü (1,511 vs 1,497), P=1'de hafif iyi (1,332 vs 1,343; orta-*t* kaybı 1,399 vs 1,428). Beklenen: logit-normal uçları bilerek az örnekler, bu metrik ise her *t*'yi eşit tartar. Karar WER/SIM ile verilmeli.
- **Kayma (shift) [H]:** SD3/RAE'nin boyuta bağlı gürültü-yönlü kaydırma kuralı boyutlar arası *artıklığa* dayanır; bizim latent beyaz olduğundan büyük kaydırma gerekmeyebilir. `t = sigmoid(N(μ,1))`, μ ∈ {0; −0,4; −0,8} taraması (bizde t=1 veri). *t*'yi [0,001; 0,999]'a kırp.

#### B2. Parametrizasyon: v yerine EDM-tarzı ön-koşullandırma
- **Kanıt [Ö]:** §1.5. x-tahmini gürültülü uçta iyi, temiz uçta kötü (kimlik eşlemesini ağ öğrenmek zorunda; *t*=0,95'te x-MSE 0,0151 vs v-tahmini 0,0031). v-tahmini temiz uçta analitik atlama bağlantısına sahip ama gürültülü uçta tam-rank gürültüyü ağdan geçirmek zorunda. EDM ön-koşullandırma (birim varyanslı veri, doğrusal yol):
  `s = t² + (1−t)²`, `x̂ = (t/s)·x_t + ((1−t)/√s)·F`, hedef `F* = ((1−t)·x₁ − t·ε)/√s` (her *t*'de birim varyans), örneklemede `v̂ = ((2t−1)/s)·x_t + F/√s` (1/(1−t) tekilliği sadeleşir, kırpma gerekmez).
- **Öneri:** ağ çıkışını `F` olarak yorumla; ~15 satır (kayıp + `sample()`). Latent istatistikleri zaten birim varyansa normalize.
- **Risk:** yalnız oyuncak ölçekte gösterildi; koşullandırma kullanımı v-tahmininden farklı gelişebilir (§1.5 teşhisleri) — uzun koşuda WER ile doğrula.

#### B3. Context-sharing batch expansion (Ke=4–6)
- **Kanıt [L ✔]:** SupertonicTTS (B, Ke)→WER: (64,1) 3,11; (256,1) 2,88; **(64,4) 2,64** — Ke 1→4 iterasyon süresine +%64, 4×B ise +%253 ekliyor. RobustSpeechFlow Ke=6 kullanıyor.
- **Bize uyumu [Ö]:** model launch-bound olduğundan DiT'i B·Ke üzerinde çalıştırmak doyuma kadar neredeyse bedava; veri yükleme yükü Ke kat azalır. Mevcut tasarımda yalnız text encoder ve referans özeti paylaşılabilir; A6-(i) yapılırsa prefix K/V de paylaşılır.

#### B4. Bağımsız metin / konuşmacı dropout'u ve guidance
- **Sorun [K]:** `model.py:265` tek ortak `drop`, p=0,1 (`config.py:16`).
- **Kanıt [L]:** Echo eğitimde %10/%10 bağımsız dropout; inference'ta bağımsız guidance (metin 3 / konuşmacı 5–8), yalnız gürültülü yarıda (`cfg_min_t=0.5`) ✔; F5 `audio_drop_prob=0.3` + `cond_drop_prob=0.2`; MegaTTS 3 CFG'siz WER 6,85 / SIM 0,43 → CFG'li 1,79 / 0,68; ZipVoice zamana bağlı CFG (erken adımlarda yalnız metin düşürülür) ✔.
- **Öneri:** yalnız-referans ~0,2–0,3, ikisi birden 0,1, yalnız-metin ~0,05–0,1; inference'ta üç dal tek batch'te.

#### B5. Yardımcı hizalama kayıpları (A-DMA)
- **Kanıt [L]:** F5-small, LibriTTS: DiT ara katmanına CTC başlığı (λ=0,1) ile ~100 epoch'ta WER 7,47→**2,48**; HuBERT kosinüs hizalaması 7,47→3,52; ikisi birlikte nihai 2,68→1,97, "yakınsama 2× hızlı".
- **Bize uyumu:** CTC hedef byte sayısından fazla frame ister; İngilizce ~15 byte/s, P=1'de 25 frame/s yeterli, P=2'de (12,5/s) **değil** — P=1 için bir gerekçe daha. SSL hizalaması eğitim-zamanı harici öğretmen gerektirir → "sıfırdan" politikasına dair bilinçli karar (§H2).

#### B6. Duration başlığını ayır ve kural tabanıyla kıyasla
- **Sorun [K]:** `training.py:41-45` duration kaybı ortak text/referans encoder'a gradyan gönderiyor.
- **Kanıt [L]:** Irodori v4 ortak eğitimde süreleri fazla tahmin etti; v4.1'de donmuş model üzerinde ayrı eğitim CER 5,35→4,69. DMOSpeech 2 (aynı üretici): GT süre WER 1,82; **konuşma-hızı kuralı 2,03; ayarsız predictor 3,75**; RL-ayarlı 1,75. Sessizlik eklenmiş prompt'lar süre MAE'sini 0,73→4,50 s, WER'i 1,98→3,46 yapıyor.
- **Öneri:** girişleri `detach` et veya üretici yakınsadıktan sonra ayrı eğit; `run-eval`'e `rule` modu ekle (`ref_frame/ref_byte × hedef_byte`); hafif kısa tahmine yanlı ol (LARoPE Tablo 2: ×0,85 → WER 2,48; ×1,2 → 2,98).

#### B7. Ucuz ekler (A/B'si kolay, kanıtı orta)
- Hız-yönü (kosinüs) kaybı: FasterDiT 150k adımda FID 27,2→21,4; LightningDiT 13,99→12,52 [L]. Ses için kanıt yok.
- Muon (2-B gizli matrislerde): Echo "küçük ablation'larda Adam'dan iyi" [Y ✔]; 22,9M U-Net'te nihai kayıp %18 düşük; DiT-XL'de erken hızlı ama yakınsamada AdamW ile berabere [L]. **22 Eylül 2026'da varsayılan optimizer yapıldı — bkz. Ek B.**
- Skip/repeat kontrastif negatifler (RobustSpeechFlow): İngilizce CER 0,48→0,35 [L ✔ özet]. Baseline oturduktan sonra.
- Minibatch-OT eşleme, Min-SNR, Huber: **atla** — koşullu üretimde OT yüksek NFE'de zarar veriyor (ImageNet 3,50→5,73) [L].

### C. Eğitim verimliliği ve altyapı

| # | Sorun [K] | Öneri | Kanıt |
|---|---|---|---|
| C1 | `data.py:119-123` kötümser maliyet | Örnekleyicide epoch'un gerçek referansını hesapla (vektörel NumPy); sıralamayı `hedef + gerçek referans`'a göre yap | §1.3: %35,6 → %0,4 |
| C2 | `tiny.yaml` B=16×2, `small.yaml` B=8×4 + checkpointing; `--frame-budget 0` | Frame bütçeli dinamik batch varsayılan; Tiny ≥64, Small ≥32 eşdeğeri; birikim 1; `--compile`; Small'da checkpointing kapalı | §1.4: 4,2× |
| C3 | `training.py:34-43` koşullar iki kez encode | `conditions()` bir kez; `forward(..., drop=, cached=)` ve `predict_duration(..., cached=)` zaten destekliyor | §1.4: −%8–10 |
| C4 | `training.py:405` yalnız `last.pt` | Her N adımda kalıcı checkpoint + son K tanesi; belgeler zaten "flow kaybıyla seçme" diyor, seçim için birden çok checkpoint şart | – |
| C5 | Eğitim sırasında sentez/ASR ölçümü yok | 5–10k adımda bir sabit 100–200 örnek: WER, SIM-o, UTMOS, süre MAE. F5-small'da WER 100k adımda 29,5 → 200k'da 4,6: kayıp eğrisi bunu göstermez [L] | – |
| C6 | `validate()` rastgele *t* | Sabit *t*-ızgarası + **koşullandırma-kullanım teşhisi**: doğru koşul ↔ *karıştırılmış* metin / *karıştırılmış* referans kayıp farkı. ASR gerektirmeyen "hizalama oturdu mu?" sinyali | §1.5 |
| C7 | EMA yalnız 0,999, ısınmasız | 0,999 / 0,9995 / 0,9999'u paralel tut, `min(1−(1+adım)^−0.75, maks)` ısınması; WER ile seç. EDM2: optimum keskin ve CFG ölçeğine bağlı [L] | – |
| C8 | Weight decay tüm parametrelere uygulanıyor (Muon'a geçişte bilerek değiştirilmedi; `optim.py`) | bias, norm, embedding, çıkış projeksiyonunda 0; LR taraması {3e-4, 6e-4, 1e-3} (Supertonic 5e-4, wd=0 ✔) | Ek B |
| C9 | Yüksek batch'te veri yükleme | Örnek başına 2 SQLite sorgusu + 2 memmap; gerçek korpusta çift/s ölç, B3 yükü Ke kat azaltır | §1.6 |

### D. Inference

| # | Öneri | Kanıt |
|---|---|---|
| D1 | `sample()` içinde koşullu + null dalı tek B=2 forward | §1.4: 2,1×, fark 0,0 [Ö]; F5, ZipVoice, Echo, Irodori hepsi böyle [L] |
| D2 | Uzunluk kovalı CUDA graph / `mode="reduce-overhead"` | B=1 ≡ B=16 süre [Ö] |
| D3 | Guidance taraması g ∈ {1,5; 2; 2,5; 3} + geç kesme | g=1,5 literatüre göre düşük: F5 eşdeğeri 3, CosyVoice 2 1,7, ZipVoice 2–4 [L]. CFG-Zero*/görüntü-tipi çizelgeler TTS'te olumsuz [L] |
| D4 | EPSS zaman ızgaraları (mevcut `times=` argümanı yeter) | F5: 7 NFE WER 2,45 / SIM 0,66 ↔ 32 NFE 2,37 / 0,66; naif 7 NFE 4,16 [L]. Izgara mel + g=3 için ayarlı; yeniden doğrula |
| D5 | Euler'de kal | F5 Tablo B.4: eşit NFE'de midpoint/Heun kazanç yok [L] |
| D6 | `posttrain.py:84-86` adayları, `posttrain.py:320-328` trajektorileri batch'le | Launch-bound rejimde ~N× [Ö→H] |
| D7 | Codec decode: alım alanı kadar örtüşmeli parçalı (tam eşdeğer) decode | Üretici hızlanınca baskın maliyet ~100M'lik decoder |
| D8 | SmoothCache / DiTReducio: düşük öncelik | "Daha az adım"dan iyi değil; DiTReducio SIM 0,640→0,590 [L] |
| D9 | Ucuz örnekleyici düğmeleri: başlangıç gürültüsünü 0,8–0,9 ile ölçekleme, adım sayısını 32–40'a çıkarma | Echo preset'leri; Echo'da bir artefakt 30→60 adımda kayboldu [Y] |
| D10 | CFG'siz eğitim ("model guidance", w=0,7) | F5 mimarisi, LibriTTS: 32 NFE WER 2,28→1,96, RTF 0,31→0,17; tek makale, ~1,5× eğitim maliyeti [L] |

### E. Veri hattı (büyük encode'dan ÖNCE)

1. **−16 dB (LUFS) seviye normalizasyonu** — prepare'de ve inference referansında aynı şekilde. [K] `codec.py:13` `no-gain`; DACVAE `compress()` varsayılanı `normalize_db=-16`; Irodori `--ref-normalize-db -16` [L]; §1.2 [Ö]. `PREPROCESSING` sürüm etiketini değiştir.
2. **Kenar sessizliğini ≤0,5 s'ye kırp; prompt'a sessizlik ekleme augmentasyonu** (HiFiTTS-2 pratiği; B6'daki süre kanıtı) [L].
3. **Bant genişliği denetimi ve etiketi.** HiFiTTS-2: 44,1/48 kHz verinin çoğu "fiilen karışık bant"; −50 dB kestirici (NeMo-SDP `BandwidthEstimationProcessor`) [L]. 48 kHz model, 16 kHz kaynaklı veride yüksek bandı boş öğrenir. Politika: <13 kHz'i filtrele veya bant etiketiyle koşullandır (ikincisi TTS'te test edilmemiş [H]).
4. **Tek ASR geçişi → silmeden metadata:** verilen transkript ↔ ASR WER/CER, karakter/s, konuşma oranı, DNSMOS. Raon-OpenTTS (0,3B): birleşik **%15 kuyruk kesimi** WER 2,19→2,00, SIM 0,661→0,672; **%50 kesim daha kötü** [L]. DNSMOS eşiği 2,8: SIM-o 0,707→0,749 [L].
5. **Metin normalizasyonu:** NeMo-text-processing veya WeTextProcessing (Apache-2.0), eğitim ve inference'ta aynı. Byte/karakter girdi küçük ölçekte çalışıyor (Supertonic, F5) [L]; bytes ↔ phoneme için küçük flow modellerde kontrollü ablation bulunamadı. espeak-ng/phonemizer GPL-3.0; misaki, g2p_en Apache-2.0.
6. **Süre sınırı:** `--max-seconds 15` LibriSpeech örnekleminde satırların %18'ini, **ses süresinin %31'ini** (3,9 / 12,7 saat) reddetti [Ö]; kullanıcının korpusunda oran farklı olacaktır ama ölçülmeli. Konumlar göreli hale gelip (A4) frame bütçeli batch (C2) açıldıktan sonra 20–30 s'ye çıkarmak veri kaybını azaltır; F5 30 s kullanıyor [L].
7. **Eğitim sesini "iyileştirme"yi A/B'siz benimseme:** F5 bakımcısı LibriTTS-R (enhanced) ile "oldukça kötü WER ve SIM" bildiriyor [Y]; Sidon ile temizlenmiş veri MOS'u artırmış ama WER/SIM raporlanmamış [L].
8. **Tam bant ek veri:** HiFiTTS-2'nin 44,1 kHz alt kümesi (≥13 kHz bant, 31,7k saat, 4.631 konuşmacı, gerçek konuşmacı kimlikleri, CC-BY-4.0) 48 kHz model için doğrudan uygun [L].
9. **Konuşmacı etiketi bağımlılığını gevşet:** tek-cümleli/etiketsiz satırlar için aynı-cümle içi kırpma + kayıp maskesi modu (Supertonic, F5, E2, ZipVoice, MegaTTS 3 etiket kullanmıyor) [L]. VoiceStar'da karışık eğitim WER 8,49→6,42, SIM −0,014 [L, AR model].

### F. Değerlendirme

1. **Karşılaştırılabilir düzenek:** Seed-TTS test-en (1.088 örnek; yerelde cache'li) + LibriSpeech-PC test-clean (1.127 cümle; hazır paket `k2-fsa/TTS_eval_datasets`). WER: Whisper-large-v3, noktalama silinmiş + küçük harf, cümle-başı WER ortalaması. **SIM-o:** WavLM-large + ECAPA `wavlm_large_finetune.pth` (yerelde cache'li), üretilen ↔ **orijinal** prompt. UTMOS `utmos22_strong`; DNSMOS kalsın. 3 tohum ortalaması [L].
2. **Mevcut durum [K]:** `metrics.py:123` `wavlm-base-plus-sv` (SUPERB EER ~%4 ↔ standart modelde ~%0,5). `candidates` yolu (`posttrain.py:79`) referans olarak *codec'ten geçirilmiş* sesi yazıyor → bu bir SIM-r; `run-eval` ise orijinal dosyayı kullanıyor (SIM-o). İkisi aynı tabloda karıştırılmamalı. SV modeli ölçeği değiştirir (CosyVoice 2: WavLM 0,652 ↔ ERes2Net 0,736) [L].
3. **Codec tavanı satırı:** DACVAE'den geçirilmiş gerçek ses, aynı metriklerle.
4. **MOS kestiricilerine temkin:** TTSDS2'de insan MOS'u ile Spearman (temiz alan): UTMOS 0,51, DNSMOS 0,41, UTMOSv2 0,39; hepsi 16 kHz'te → 8 kHz üstünü görmüyor [L]. 48 kHz avantajı yalnız dinleme testiyle gösterilebilir (≥20 dinleyici × 30 cümle; webMUSHRA).

Hedef referansları (WER % / SIM-o / UTMOS) [L]:

| Model | LibriSpeech-PC test-clean | Seed test-en |
|---|---|---|
| Gerçek ses | 2,23 / 0,69 (Whisper) | 2,14 / 0,734 / 3,52 |
| F5-TTS 336M | 2,42 / 0,66 | 1,83 / 0,67 |
| F5-small 155M (Emilia) | 2,10 / 0,615 / 3,84 | 1,96 / 0,628 / 3,66 |
| ZipVoice 123M | 1,64 / 0,668 / 3,98 | 1,70 / 0,697 / 3,82 |
| SupertonicTTS 44M (945 saat) | 2,41 (LS-PC); LS-clean 2,64 / SIM 0,472 | – |
| Supertone 0,06B (~10k saat) ✔ | – | 1,44 / 0,60 |
| DMOSpeech 2 (4 adım) | – | 1,75 / 0,698 |

44M için makul ilk hedef [H]: LibriSpeech-PC WER ≤ 3, SIM-o ≥ 0,50.

### G. Post-training

1. **DPO ölçeği [K+L]:** `posttrain.py:205,212` β=10, ortalama-indirgenmiş MSE üzerinde. Diffusion-DPO 2000–5000; diffusers uygulaması aynı indirgemeyle β=5000. β=10'da sigmoid doğrusal bölgede kalır, kaybedeni itmeyi hiçbir şey sınırlamaz. {500, 2000, 5000} tara (LR ∝ 1/β); kazanan/kaybeden kayıplarını ayrı, örtük ödül doğruluğunu ve ‖v_θ−v_ref‖² değerini logla; SIM'de erken dur. ARDM-DPO: kararsız, SIM düşüyor; INTP/F5: WER 3,44→2,38 ama SIM 0,670→0,652 [L].
2. **Çiftleri on-policy tazele:** TangoFlux'ta çevrimdışı çiftler iki iterasyondan sonra bozuluyor; kazanan-FM çapası ve gerçek-veri replay'i (ikisi de mevcut) destekleniyor [L].
3. **Distillation hedef çelişkisi [K]:** `posttrain.py:310-329` rehberli sekant hedefi ile rehbersiz anlık FM replay'ini aynı `(x_t, t)` için karıştırıyor; öğrencinin guidance/adım-boyu girdisi yok. **ZipVoice tarzı** en güvenilir yol [L ✔]: öğrenciye ω gömmesi, veri/gürültü interpolasyonundan `x_t`, iki CFG'li öğretmen adımı, sekant hızına L2, ardından EMA öz-öğretmen. 123M'de 4 NFE: WER 1,51 / SIM 0,657 (öğretmen 16 NFE: 1,64 / 0,668); consistency (UTMOS 3,30) ve ReFlow (WER 2,56) belirgin kötü. ~118M altı doğrulanmamış.
4. **En ucuz RL hedefi duration başlığı:** DMOSpeech 2 GRPO ile WER 3,75→1,75 [L].
5. **Ödül istismarı nöbetçileri:** eğitimde kullanılmayan UTMOS, F0 varyasyon katsayısı (DMOSpeech 2'de 0,666→0,464 mod daralması), konuşma hızı; SIM/DNSMOS ödüllerini CER ≤ τ ile kapıla [L].

### H. Stratejik kararlar

**H1. Codec.** Frozen Meta DACVAE-128 kesin tercih; kanıtlar onun kullanımdaki en zor latentlerden biri olduğunu söylüyor: §1.1; Irodori v1→v2 geçişi (128 → 32-boyutlu WavLM-distile DACVAE; yazar v2 kalitesini buna bağlıyor [Y]); Semantic-VAE + F5: WER 2,65→2,10, SIM 0,59→0,64 [L]; LongCat: latent boyutu 64→128→256 arttıkça reconstruction iyileşip TTS kalitesi tutarlı biçimde düşüyor [L]. İngilizce/çok dilli 32-boyutlu Semantic-DACVAE yayımlanmamış [L]. Yol: önce A1+A2 ile mevcut codec'in sınırını ölç; Base de takılırsa İngilizce 32-boyutlu semantic fine-tune'u ayrı proje olarak değerlendir.

**H2. "Sıfırdan" politikası ve eğitim-zamanı öğretmenler.** A-DMA'nın SSL hizalaması frozen bir HuBERT/WavLM gerektirir; inference'a bağımlılık eklemez ama ASR/DNSMOS hakemlerinden farklı olarak eğitimi etkiler. CTC başlığı harici model gerektirmez. Öneri: önce CTC; SSL hizalamasını açık politika kararıyla aç.

---

## 3. Önceliklendirilmiş yol haritası

**Aşama 0 — yeniden eğitim gerektirmeyen düzeltmeler (1–2 gün)**
C1, C2, C3, C4, C6, D1, D6, A7, F1–F3. Beklenen: ~5× eğitim verimi, ~2× inference, literatürle kıyaslanabilir ölçüm.

**Aşama 1 — büyük encode ve ilk ciddi koşudan önce**
E1–E5 (özellikle E1: sonradan değiştirmek tüm cache'i yeniden üretmek demek), A1 (P=1), B1 (logit-normal), B2 (EDM), C5, C7, C8.

**Aşama 2 — eşleştirilmiş ablation'lar (her biri tek değişken; §1.4'teki hızla her Tiny koşusu saatler mertebesinde)**
1. A5: self-attention'lı text encoder → + LARoPE
2. B3: batch expansion (256×1 ↔ 64×4)
3. B4: bağımsız dropout + üç dallı guidance
4. B5: CTC başlığı
5. A3+A4: düşük-rank AdaLN → genişlik; RoPE + QK-norm
6. B6: duration ayrıştırma + kural tabanı
7. A2: Base yapılandırması
8. A6, E6

Önceki rapordaki A1–A5 (AdaLN'siz adaylar) bu listeye göre **daha düşük öncelikli**: parametre payını A3 çözüyor; ölçülen darboğazlar token boyutu ve hizalama.

**Aşama 3 — iyi bir baseline'dan sonra**
G1–G5, D3–D4 taramaları, ZipVoice-tarzı distillation.

---

## 4. Bu çalışmada çürütülen / zayıflayan hipotezler

Olumsuz sonuçlar da kayıt altında:

- "DACVAE'de kullanılmayan kanallar var, kanal-bazlı normalizasyon onları şişiriyor" → **yanlış** (§1.1: tüm kanallar aktif, std'ler benzer).
- "PCA ile latent boyutu ucuza düşürülür" → **yanlış** (96 bileşen bile codec hatasından kötü).
- "x-tahmini (JiT) token boyutu ≈ genişlik rejiminde net kazanç" → **burada doğrulanmadı** (§1.5: uçlara göre karışık). EDM ön-koşullandırma daha tutarlı.
- "Logit-normal örnekleme doğrulama kaybını düşürür" → uniform-ızgara metriğinde etki küçük ve **yönü karışık** (P=2'de hafif kötü, P=1'de hafif iyi); bu metrik soruyu cevaplayamaz, WER gerekir.
- "Metni sıfırlayarak ölçülen koşullandırma kazancı metin kullanımını gösterir" → **yanıltıcı**; yalnız-metin düşürme eğitimde hiç görülmeyen bir girdi. Karıştırılmış metin/referans doğru teşhis (§1.5).
- "P=1 bedavadır" → yalnız mevcut küçük batch'te; doygun batch'te ~1,75×.

---

## Ek A — Aşama 0/1 için uygulama taslakları

Bunlar taslaktır; depoya uygulanmadı. İlk ikisi bu çalışmada ölçüm betiklerinde çalıştırılıp doğrulandı.

**A.1 Koşulları bir kez encode et (C3)** — `flow_loss`'a `cached=` parametresi ekleyip `model(..., drop=drop, cached=cached)` olarak geçirmek yeterli; `forward` cache'lenmiş tensörlere dropout maskesini zaten uyguluyor (`model.py:190-196`), duration ise bilerek düşürülmemiş koşulları görüyor:

```python
# training.py, Objective.forward
cached = self.model.conditions(batch["prompt"], batch["prompt_mask"], batch["tokens"], batch["segments"])
details = flow_loss(self.model, batch, dropout, return_details=True, cached=cached)
duration = self.model.predict_duration(
    batch["prompt"], batch["prompt_mask"], batch["tokens"], batch["segments"], cached=cached
)
```

**A.2 CFG dallarını tek forward'da çalıştır (D1)** — ölçülen fark 0,0:

```python
# model.py, sample(): döngüden önce bir kez
zeros = torch.zeros_like
prompt2, mask2 = torch.cat([prompt, zeros(prompt)]), torch.cat([prompt_mask, zeros(prompt_mask)])
cond2 = (torch.cat([cond[0], zeros(cond[0])]), cond[1].repeat(2, 1), torch.cat([cond[2], zeros(cond[2])]))
valid2, tokens2, segments2 = valid.repeat(2, 1), tokens.repeat(2, 1), segments.repeat(2, 1)
# her adımda
x2 = torch.cat([x, x.masked_fill(prompt_mask[..., None], 0)])
v, u = model(x2, t.repeat(2), prompt2, mask2, valid2, tokens2, segments2, cached=cond2).chunk(2)
v = u + guidance * (v - u)
```

**A.3 Örnekleyicide gerçek epoch maliyeti (C1)** — referans seçimini vektörel ve epoch-deterministik yap; aynı fonksiyonu hem `LatentDataset.__getitem__` hem `BucketBatchSampler` kullansın. Eşleştirme RNG'si değişir → `pairing_version` artırılmalı (henüz eğitilmiş model olmadığından maliyetsiz):

```python
def reference_indices(group_start, group_end, seed, epoch):
    rng = np.random.default_rng([seed, epoch])
    count = group_end - group_start                       # konuşmacının cümle sayısı (>= 2)
    ref = group_start + (rng.random(len(count)) * (count - 1)).astype(np.int64)
    return ref + (ref >= np.arange(len(count)))           # kendisini atla

# sampler.batches():  costs = lengths + lengths[reference_indices(..., self.epoch)]
```

**A.4 EDM ön-koşullandırma (B2)** — ağ çıkışı `F`; latentler birim varyansa normalize olduğu için σ_data = 1:

```python
# kayıp (hedef frame'lerde):
root = (t**2 + (1 - t)**2).sqrt()                          # t: [B,1,1]
target = ((1 - t) * x1 - t * noise) / root                 # her t'de birim varyans
loss = per_example_mse(model(xt, ...), target, mask)
# örnekleme:
v = ((2 * t - 1) / root**2) * x + model(x, ...) / root     # 1/(1-t) tekilliği yok
```

**A.5 Koşullandırma-kullanım teşhisi (C6)** — doğrulamada, aynı `(x_t, t, gürültü)` için üç forward: doğru koşullar; `tokens/segments` batch içinde bir kaydırılmış; referans prefix'i başka örneğin referansıyla (hedef uzunluğuna döşenerek) değiştirilmiş. `text_gain = L(karışık metin) − L(doğru)`; sıfıra yakınsa model metni kullanmıyor. Çalışan örnek: [`research-2026-09-21/scripts/ab_train.py`](research-2026-09-21/scripts/ab_train.py) içindeki `conditioning_gain`.

---

## Ek B — 22 Eylül 2026: varsayılan optimizer Muon

Kullanıcı kararıyla AdamW yerine **Muon** varsayılan yapıldı ([`src/dacvae_tts/optim.py`](../src/dacvae_tts/optim.py)); `train.optimizer: adamw` / `--optimizer adamw` eski davranışı geri getirir. Bu, bu belgedeki ölçümlerden *sonra* yapılan tek model-kodu değişikliğidir; §1'deki tüm AdamW sonuçları eski optimizer'a aittir.

**Uygulama [K]:**
- PyTorch 2.8'de `torch.optim.Muon` yok (2.9'da geldi); depo `torch>=2.5` desteklediği için bağımsız uygulama yazıldı. Newton–Schulz (5 adım, katsayılar 3,4445 / −4,7750 / 2,0315; CUDA'da BF16), Nesterov momentum 0,95.
- **Muon grubu** (Tiny parametrelerinin %97,4'ü, Small'un %98,7'si): attention `q`/`kv`/`out`, FFN, AdaLN projeksiyonu, text-encoder ve zaman/referans/duration MLP matrisleri. Satır yönünde birleştirilmiş matrisler ayrı ayrı ortogonalleştirilir: `kv` 2 parça, AdaLN `9D` projeksiyonu 9 parça.
- **AdamW grubu:** embedding'ler, konvolüsyon filtreleri, sınır projeksiyonları (`input`, referans girişi), çıkış başlıkları (`output`, duration), bias ve norm kazançları, `[1,D]` vektörleri — referans uygulamanın ve Echo/Irodori tariflerinin izlediği ayrım [L].
- Güncellemeler `0,2·√max(satır, sütun)` ile ölçeklenir (Moonlight: `W ← W − η(0,2·O·√max(A,B) + λW)` [L ✔]), böylece `learning_rate` ve `weight_decay` iki grup için de aynı anlamı taşır. Weight decay davranışı bilerek değiştirilmedi (C8 hâlâ açık).
- Aynı şekilli matrisler tek batch'li iterasyonda işlenir (Tiny'de 162 blok → birkaç `bmm`); model launch-bound olduğu için bu önemli. Ölçülen adım maliyeti B=32'de **≈ +%8**.
- Post-training, checkpoint'te kayıtlı optimizer ailesini kullanır: Moonlight'ta hem Muon ile pretrain hem Muon ile finetune edilen model ablation'larda en iyisi; optimizer'lar uyuşmadığında Muon'un avantajı kayboluyor [L ✔]. Birebir resume ve 2-rank DDP testleri Muon ile geçiyor; toplam 103 test.

**Oyuncak ölçek karşılaştırması [Ö]** (§1.5 protokolü, 4000 adım, görülmemiş konuşmacı v-MSE):

| Varyant | LR | AdamW | Muon |
|---|---:|---:|---:|
| base (v, uniform, P=2) | 3e-4 | **1,4973** | 1,5089 |
| base (v, uniform, P=2) | 1e-3 | **1,4150** | 1,4330 |
| p1 | 3e-4 | 1,3434 | **1,3253** |
| edm_p1_lognorm | 3e-4 | 1,3048 | **1,2815** |

Okuma: Muon ilk ~1000 adımda geride başlıyor, P=1 ayarlarında 2000. adımdan sonra öne geçiyor (−%1,3 / −%1,8); P=2 baseline'da iki LR'de de AdamW'nin hafif gerisinde (+%0,8 / +%1,3). Yani bu ölçekte **kabaca başa baş**, önerilen P=1 ayarlarında hafif lehte; literatürdeki "erken hızlı, yakınsamada berabere" tablosuyla uyumlu. Yan bulgu: **LR 1e-3, iki optimizer için de 3e-4'ten belirgin iyi** (AdamW 1,497 → 1,415) — C8'deki LR taramasını destekliyor; kısa koşular yüksek LR'yi kayırdığı için tam koşuda doğrulanmalı. Gerçek karar yine WER/SIM ile verilmeli.

---

## 5. Kaynaklar

Yerel: [`model.py`](../src/dacvae_tts/model.py), [`training.py`](../src/dacvae_tts/training.py), [`data.py`](../src/dacvae_tts/data.py), [`inference.py`](../src/dacvae_tts/inference.py), [`posttrain.py`](../src/dacvae_tts/posttrain.py), [`metrics.py`](../src/dacvae_tts/metrics.py), [`codec.py`](../src/dacvae_tts/codec.py), [`prepare.py`](../src/dacvae_tts/prepare.py).

| Konu | Kaynak |
|---|---|
| Echo-TTS (blog ✔, kod) | https://jordandarefsky.com/blog/2025/echo/ · https://github.com/jordandare/echo-tts |
| Irodori-TTS, Semantic-DACVAE | https://github.com/Aratako/Irodori-TTS · https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim |
| LAION DACVAE-Echo | https://github.com/LAION-AI/jax-dacvae-echotts · https://github.com/LAION-AI/scaled-echo-tts |
| DACVAE, SAM Audio, Movie Gen Audio, Text-Audiobox | https://github.com/facebookresearch/dacvae · https://arxiv.org/abs/2512.18099 · https://arxiv.org/abs/2410.13720 · https://arxiv.org/html/2609.03992v1 |
| SupertonicTTS ✔, LARoPE ✔, RobustSpeechFlow ✔ | https://arxiv.org/abs/2503.23108 · https://arxiv.org/abs/2509.11084 · https://arxiv.org/abs/2605.22083 |
| F5-TTS, EPSS, A-DMA | https://arxiv.org/abs/2410.06885 · https://arxiv.org/abs/2505.19931 · https://arxiv.org/abs/2505.19595 |
| ZipVoice ✔, MegaTTS 3, DiTTo-TTS | https://arxiv.org/abs/2506.13053 · https://arxiv.org/abs/2502.18924 · https://arxiv.org/abs/2406.11427 |
| JiT ✔, RAE, SD3, LightningDiT, EDM | https://arxiv.org/abs/2511.13720 · https://arxiv.org/abs/2510.11690 · https://arxiv.org/abs/2403.03206 · https://arxiv.org/abs/2501.01423 · https://arxiv.org/abs/2206.00364 |
| Semantic-VAE, EzAudio | https://arxiv.org/abs/2509.22167 · https://arxiv.org/abs/2409.10819 |
| DMOSpeech 2, IntMeanFlow | https://arxiv.org/abs/2507.14988 · https://arxiv.org/abs/2510.07979 |
| Diffusion-DPO, TangoFlux, INTP | https://arxiv.org/abs/2311.12908 · https://arxiv.org/abs/2412.21037 · https://arxiv.org/abs/2505.04113 |
| TTS'te CFG hileleri, model guidance | https://arxiv.org/abs/2509.19668 · https://arxiv.org/abs/2504.20334 |
| Seed-TTS-eval, TTSDS2 | https://github.com/BytedanceSpeech/seed-tts-eval · https://arxiv.org/abs/2506.19441 |
| Raon-OpenTTS (filtreleme), HiFiTTS-2 (bant), Emilia, DNSMOS filtreleme | https://arxiv.org/abs/2605.20830 · https://arxiv.org/abs/2506.04152 · https://arxiv.org/abs/2407.05361 · https://arxiv.org/abs/2406.05699 |
| VoiceStar | https://arxiv.org/abs/2505.19462 |
| Muon (blog), Moonlight ✔ | https://kellerjordan.github.io/posts/muon/ · https://arxiv.org/abs/2502.16982 |

**Sınır:** ✔ işaretsiz [L] maddeleri araştırma ajanları kaynağından okuyup raporladı; ben yalnız karar açısından en kritik olanları ikinci kez teyit ettim. Yayımlanmış sonuçlar yazarların kendi veri/değerlendirme koşullarına aittir. Yerel ölçümler tek bir RTX 5070 Ti ve 16 kHz kaynaklı LibriSpeech ile sınırlıdır.
