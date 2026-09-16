# Editto'ya katkı

Editto topluluk katkılarına açıktır. Hata düzeltmeleri, kullanıcı arayüzü,
belgelendirme, testler, farklı işletim sistemi desteği ve daha güvenilir nefes
modelleri için issue veya pull request gönderebilirsiniz.

## Başlamadan önce

1. Depoyu fork edin ve değişikliğiniz için ayrı bir dal açın.
2. Kişisel ses kaydı, etiket tablosu, API anahtarı veya makineye özel yol
   eklemeyin.
3. Eğitim ve test kayıtlarını kayıt/konuşmacı bazında ayırın. Aynı kaydın
   parçalarını iki tarafa dağıtmayın.
4. Yeni bir modeli varsayılan yapmayın. Önce mevcut varsayılanla aynı bağımsız
   testte karşılaştırın ve konuşmaya zarar veren yanlış pozitifleri raporlayın.
5. Deneysel modelleri açıkça `experimental` olarak işaretleyin.

## Yerel geliştirme

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m breath_cleaner.labeler --labels dataset\candidates\labels.csv
```

Arayüz `http://127.0.0.1:8765/model` adresinde açılır.

## Pull request kontrolü

- Testler geçiyor.
- Kaynak sesin üzerine yazılmıyor.
- Varsayılan model veya eşik değişiyorsa karşılaştırmalı ölçüm ekleniyor.
- Yeni bağımlılıkların lisansı ve amacı açıklanıyor.
- Ham kayıtlar ve kişisel etiketler commit edilmiyor.

Katkılar MIT lisansı altında yayımlanır. `models/` altındaki ağırlıklar için
`MODEL_LICENSE.md` geçerlidir.
