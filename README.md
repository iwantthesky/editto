# Editto

Editto, konuşma kayıtlarındaki nefes benzeri sesleri yerel olarak tespit edip azaltmak için geliştirilmiş Windows uygulamasıdır. Kaynak sesin üzerine yazmaz; sonucu yeni bir WAV dosyası olarak üretir.

Uygulama özellikle konuşmayı korumaya öncelik verir. Kullanıcı, algılanan bölgeleri dinleyebilir, yanlış tespitleri kaldırabilir ve nefesleri hafifçe azaltma, güçlü azaltma, sessize alma veya tamamen kesme seçeneklerinden birini kullanabilir.

## Hazır Windows sürümü

Python veya PyTorch kurmadan kullanmak için **Releases** bölümündeki `Editto-Kurulum-1.2.0-Windows-x64.exe` dosyasını indirin ve çalıştırın.

Windows SmartScreen, kurulum dosyası henüz ticari bir kod imzalama sertifikasıyla imzalanmadığı için uyarı gösterebilir.

## Dahil edilen modeller

- **Editto Precision v2** — varsayılan ve mevcut kişisel karşılaştırmada en dengeli model.
- **Editto Precision v1** — önceki Precision sürümü.
- **Editto Classic** — klasik model.
- **Editto Legacy** — eski uyumluluk modeli.
- **Editto Neural v3 (Deneysel)** — tam kaydı kayan pencerelerle tarayan küçük CNN.

Teknik model kimlikleri geriye dönük uyumluluk için korunmuştur. Neural v3 henüz farklı konuşmacı ve mikrofonlardan oluşan yeterli bağımsız testte genel başarı kanıtlamadığı için deneysel olarak işaretlenir ve varsayılan değildir.

## Kendi sesiniz ve mikrofonunuzla model eğitme

Editto yalnız hazır modelleri kullanmakla sınırlı değildir. Kullanıcılar kendi ses kayıtlarını, mikrofonlarını ve kayıt ortamlarını kullanarak kişisel bir nefes modeli hazırlayabilir.

Önerilen akış:

1. Aynı metni veya doğal konuşmayı kullandığınız farklı oturumlarda birkaç kayıt alın.
2. Mümkünse kullanacağınız her mikrofonla ayrı kayıt yapın.
3. Kayıtları yerel etiketleme arayüzüne ekleyin.
4. Dalga biçiminde nefes bölgelerini elle seçip `breath` olarak etiketleyin.
5. Konuşma, sürtünme, ağız sesi ve ortam gürültüsü örneklerini sırasıyla `speech` veya `noise` olarak işaretleyin.
6. Aynı kaydın parçalarını hem eğitim hem test bölümüne koymayın. Test için tamamen ayrı kayıtlar ayırın.
7. Modeli eğitin, ayrı kayıtlarda yanlış konuşma kesmelerini dinleyin ve ancak yeterli sonuç aldıktan sonra varsayılan yapın.

Yerel arayüzü kaynak koddan başlatmak için:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m breath_cleaner.labeler --labels dataset\candidates\labels.csv
```

Ardından tarayıcıdan `http://127.0.0.1:8765/model` adresini açın.

Klasik kişisel modeli eğitmek için:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m breath_cleaner.train_model --labels dataset\candidates\labels.csv
```

Kayıt bazında ayrım kullanan Precision v2 eğitim akışı:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\train_kabum_v2.py
```

Deneysel CNN'i kişisel kayıtlarla uyarlamak için PyTorch gerekir:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\finetune_personal_cnn.py
```

Ham sesler, etiket tabloları ve üretilen kişisel modeller varsayılan olarak Git'e alınmaz. Bunlar kullanıcının bilgisayarında kalmalıdır.

## Veri güvenliği ve değerlendirme

- İşleme yereldir; uygulamanın çalışması için sesleri bir bulut servisine göndermek gerekmez.
- Uzun bir konuşma klibini otomatik olarak tamamen “nefessiz” kabul etmeyin.
- Fısıltı, `s`, `ş`, `f`, `h`, ağız sesi ve mikrofon sürtünmesi nefesle karışabilir.
- Eğitim, doğrulama ve test kayıtlarını konuşmacı/kayıt bazında ayırın.
- Precision, recall ve konuşmadaki yanlış pozitifleri ayrı raporlayın.
- Küçük veya yalnız tek konuşmacılı testleri genel başarı kanıtı olarak sunmayın.

## Kaynaktan komut satırı kullanımı

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m breath_cleaner.cli input.wav -o output.wav --segments-json segments.json
```

FFmpeg PATH üzerinde bulunuyorsa M4A ve yaygın video/ses biçimleri de açılabilir.

## Proje durumu

Editto aktif geliştirme aşamasındadır. Varsayılan model konuşmayı koruma önceliğiyle seçilmiştir; farklı dil, kişi, mikrofon ve kayıt ortamlarında sonuçlar değişebilir. Çıktıyı kullanmadan önce algılanan bölgeleri dinleyerek kontrol edin.

