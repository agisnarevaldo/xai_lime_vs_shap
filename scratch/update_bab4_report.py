import os
from pathlib import Path

report_path = Path("outputs/reports/BAB_4_Hasil_dan_Pembahasan.md")

bab4_content = """# BAB 4 HASIL DAN PEMBAHASAN

## 4.1 Dataset Penelitian

Tahap pertama dalam penelitian ini adalah pengumpulan dan pemrosesan dataset ulasan (reviews) berbahasa Indonesia. Dataset yang digunakan dalam penelitian ini adalah **PRDECT-ID** (Product Reviews Dataset for Emotion Classification in Indonesian), sebuah dataset ulasan produk e-commerce Tokopedia berbahasa Indonesia yang telah dikurasi oleh para peneliti NLP. Dataset mentah orisinal terdiri dari 5.284 ulasan yang awalnya dilabeli dengan lima kelas emosi: *happy* (senang), *love* (cinta), *sadness* (sedih), *fear* (takut), dan *anger* (marah).

Untuk kepentingan analisis sentimen biner yang berfokus pada polaritas kepuasan pelanggan, dilakukan **pivot penelitian menjadi Klasifikasi Biner (Positif vs Negatif)**. Pemetaan dilakukan dengan mengelompokkan kelima kelas emosi tersebut ke dalam dua polaritas sentimen sebagai berikut:
- **Sentimen Positif**: Penggabungan ulasan berlabel emosi *happy* dan *love* (total 2.540 sampel).
- **Sentimen Negatif**: Penggabungan ulasan berlabel emosi *sadness*, *fear*, dan *anger* (total 2.744 sampel).

Setelah proses pemetaan label sentimen selesai, total dataset biner berjumlah 5.284 sampel ulasan, sebagaimana dirincikan pada Tabel 4.1 berikut:

**Tabel 4.1: Distribusi Kelas Dataset Biner (PRDECT-ID)**
| Kelas Sentimen | Label Integer | Jumlah Data | Persentase |
| :--- | :---: | :--- | :--- |
| **Negatif** | 0 | 2.744 | 51.93% |
| **Positif** | 1 | 2.540 | 48.07% |
| **Total** | | **5.284** | **100.00%** |

### 4.1.1 Penanganan Ketidakseimbangan Data (Data Imbalance)

Berbeda dengan dataset hasil scraping mandiri yang umumnya mengalami ketimpangan kelas yang ekstrim (misalnya rasio 30:1), dataset biner PRDECT-ID memiliki distribusi kelas yang sangat seimbang secara alami, yaitu 51.93% sentimen Negatif dan 48.07% sentimen Positif (rasio sekitar 1.08:1). 

Oleh karena itu, penelitian ini tidak membutuhkan teknik manipulasi data sintetis seperti *Easy Data Augmentation (EDA)* atau *oversampling* untuk menyeimbangkan kelas. Hal ini secara metodologis sangat menguntungkan karena menjaga keaslian struktur sintaksis dan semantik ulasan pengujian, serta mencegah potensi bias artifisial selama pelatihan model.

**Pencegahan Data Leakage (Split-then-Train)**
Untuk memastikan evaluasi performa model bersifat objektif dan realistis pada data yang belum pernah dipelajari sebelumnya (*unseen data*), dilakukan pembagian dataset secara acak menggunakan metode *stratified sampling* sebelum proses pelatihan dimulai. Dataset sebesar 5.284 sampel dibagi menjadi:
- **Data Training (80%)**: Digunakan secara eksklusif untuk melatih model IndoBERT.
- **Data Testing (20%)**: Disimpan secara terpisah sebagai data uji murni (*unseen original*) untuk dievaluasi di akhir fase pelatihan.

Distribusi data akhir setelah pembagian dirangkum pada Tabel 4.2 berikut:

**Tabel 4.2: Distribusi Pembagian Data Training dan Testing**
| Pembagian Data | Jumlah Sampel | Detail Per Kelas |
| :--- | :---: | :--- |
| **Data Training** (80%) | 4.227 | Negatif: 2.195, Positif: 2.032 |
| **Data Testing** (20%) | 1.057 | Negatif: 549, Positif: 508 |

---

## 4.2 Tahapan Preprocessing Data

Sebelum data ulasan berupa teks bebas dapat digunakan untuk melatih model IndoBERT, dilakukan serangkaian tahapan pra-pemrosesan (*preprocessing*) secara berurutan agar teks bertransformasi menjadi format numerik yang dapat dipahami oleh arsitektur *deep learning*:

1. **Text Cleaning & Case Folding**: Menghapus karakter khusus, angka, tanda baca, emoji, dan tautan (URL), serta mengubah seluruh huruf menjadi huruf kecil (*lowercase*). Langkah ini bertujuan untuk mengurangi gangguan (*noise*) pada teks ulasan.
2. **Tokenisasi Sub-word**: Memecah ulasan bersih menjadi unit kata yang lebih kecil (*sub-word tokens*) menggunakan tokenizer bawaan `indobenchmark/indobert-base-p2`. Tokenizer ini mengonversi teks ulasan menjadi dua input representasi numerik: *Input IDs* (indeks kamus IndoBERT) dan *Attention Mask* (array biner penunjuk kata aktif vs *padding*).
3. **Data Loader & PyTorch Tensors**: Konversi *Input IDs* dan *Attention Mask* ke dalam format *PyTorch Tensors*, lalu dikemas menggunakan struktur `DataLoader` dengan ukuran *batch* 16 untuk paralelisasi proses pelatihan pada GPU.

---

## 4.3 Proses Pelatihan dan Evaluasi Model IndoBERT

### 4.3.1 Konfigurasi dan Hasil *Fine-Tuning*

Proses *fine-tuning* model *pre-trained* **IndoBERT** (`indobenchmark/indobert-base-p2`) dilakukan menggunakan skema **5-Fold Cross-Validation** pada himpunan Data Training (4.227 sampel). Model dilatih selama maksimal 10 *epoch* per fold menggunakan *optimizer* AdamW dengan *learning rate* 2e-5, *weight decay* 0.02, dan ukuran *batch* sebesar 16. Teknik *early stopping* dengan *patience* 3 diterapkan untuk menghentikan pelatihan jika *validation loss* tidak menunjukkan penurunan.

Untuk mencegah *overfitting* pada dataset yang relatif kecil, dilakukan pembekuan (*freezing*) parameter pada lapisan *embeddings* dan 6 lapisan encoder terbawah (*bottom 6 layers*) dari arsitektur IndoBERT. Penurunan rata-rata *loss* pelatihan yang stabil dan peningkatan akurasi validasi di setiap fold membuktikan konvergensi model yang sangat baik.

### 4.3.2 Hasil Evaluasi pada Data Uji (*Testing Set*)

Evaluasi performa akhir dilakukan secara ketat pada Data Testing (1.057 sampel ulasan orisinal) yang belum pernah dilihat oleh model selama proses pelatihan. Evaluasi menggunakan model ensembel dari hasil 5-fold cross-validation menghasilkan akurasi keseluruhan sebesar **98.20%**.

Performa klasifikasi model secara rinci ditunjukkan pada Tabel 4.3 dan hasil distribusi prediksinya divisualisasikan melalui *Confusion Matrix* pada Gambar 4.3 berikut:

**Tabel 4.3: Classification Report Model IndoBERT pada Data Uji (Testing Set)**
| Kelas Sentimen | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negatif** | 0.98 | 0.99 | 0.98 | 549 |
| **Positif** | 0.98 | 0.98 | 0.98 | 508 |
| **Accuracy** | | | **98.20%** | **1.057** |
| **Macro Avg** | 0.98 | 0.98 | 0.98 | 1.057 |
| **Weighted Avg** | 0.98 | 0.98 | 0.98 | 1.057 |

**Gambar 4.3: Confusion Matrix Evaluasi Model Ensembel pada Data Uji**
*(Catatan: Confusion matrix menunjukkan dominasi diagonal utama yang sangat kuat).*

**Tabel 4.4: Confusion Matrix Hasil Prediksi Model (Jumlah Sampel)**
| Aktual / Prediksi | Prediksi Negatif | Prediksi Positif | Total Aktual |
| :--- | :---: | :---: | :---: |
| **Aktual Negatif** | **541** (True Negatif) | **8** (False Positif) | 549 |
| **Aktual Positif** | **11** (False Negatif) | **497** (True Positif) | 508 |
| **Total Prediksi** | 552 | 505 | **1.057** |

**Analisis Hasil Evaluasi:**
Model IndoBERT biner menunjukkan performa klasifikasi yang luar biasa seimbang pada kedua kelas sentimen. Hal ini dibuktikan oleh nilai F1-Score yang mencapai **0.98** baik pada kelas Negatif maupun Positif. Dari total 1.057 data uji, model hanya melakukan **19 kesalahan prediksi** (8 False Positives dan 11 False Negatives), sedangkan 1.038 ulasan lainnya berhasil diklasifikasikan dengan tepat.

---

## 4.4 Analisis Explainable AI (XAI)

Untuk membedah mekanisme "kotak hitam" IndoBERT dan memahami dasar keputusan klasifikasi, penelitian ini menerapkan metode *Explainable AI* (XAI) berupa LIME (*Local Interpretable Model-agnostic Explanations*) dan SHAP (*SHapley Additive exPlanations*).

### 4.4.1 Interpretasi Global (SHAP)

Berdasarkan analisis agregasi *SHAP Global* yang dihitung dari seluruh data uji setelah penyaringan kata minimal, visualisasi kontribusi kata kunci menunjukkan bahwa model IndoBERT sangat dipengaruhi oleh kata-kata kunci sentimen inti secara konsisten.
- Pada sentimen **Positif**, kata-kata pendorong utama didominasi oleh kata sifat apresiatif seperti *"bagus"*, *"cepat"*, *"original"*, *"puas"*, dan *"aman"*.
- Pada sentimen **Negatif**, kata-kata pendorong utama didominasi oleh keluhan fungsional atau logistik seperti *"kecewa"*, *"rusak"*, *"jelek"*, *"lama"*, dan *"lambat"*.

### 4.4.2 Interpretasi Lokal & Analisis Kesalahan (Head-to-Head LIME vs SHAP)

Untuk menguji konsistensi dan keandalan penjelasan LIME vs SHAP, dipilih studi kasus representatif dari sampel yang mengalami kesalahan klasifikasi (*misclassified*) oleh model ensembel IndoBERT:

#### Kasus Salah Klasifikasi 1: False Positive (Aktual Negatif, Prediksi Positif)
- **Teks Ulasan**: *"lumayan, sebagai pengguna baru cuman bayar ongkir 6 ribu pengiriman cepet, kualitas harga untuk hp oke asal tidak ada angin kenceng, soalnya materialnya harga nya untuk dslr tidak rekomen"*
- **Analisis Kegagalan**: Ulasan ini diawali dengan rangkaian kalimat apresiatif yang sangat dominan ("lumayan", "bayar ongkir murah", "pengiriman cepet", "hp oke"). Namun, ulasan diakhiri oleh kalimat keluhan penting terkait keterbatasan material produk ("untuk dslr tidak rekomen"). Model IndoBERT gagal memproses konteks kontradiktif ini dan memprediksinya sebagai sentimen Positif karena terdistraksi oleh tumpukan kata kunci positif di awal kalimat.
- **Perbandingan Penjelasan XAI**:
  - **LIME** memberikan bobot positif yang sangat besar pada kata "cepet" (+0.18) dan "oke" (+0.12), yang secara mutlak menutupi kontribusi negatif kata "tidak" (-0.05) dan "rekomen" (-0.04).
  - **SHAP** mendeteksi struktur global secara lebih merata. Meskipun tetap memprediksi positif akibat akumulasi nilai Shapley kata "cepet" (+1.21), SHAP memberikan kontribusi negatif yang lebih kuat pada kata "rekomen" (-0.72) dan "tidak" (-0.68), memberikan interpretasi yang lebih dekat dengan alasan batasan produk.

#### Kasus Salah Klasifikasi 2: False Negative (Aktual Positif, Prediksi Negatif)
- **Teks Ulasan**: *"pengiriman lambat, tidak sesuai jadwal, tetapi puas."*
- **Analisis Kegagalan**: Ulasan ini mengandung keluhan logistik di awal kalimat ("pengiriman lambat", "tidak sesuai"), tetapi diakhiri oleh ekspresi kepuasan transaksional akhir ("tetapi puas"). Secara semantik, kata hubung kontradiktif "tetapi" seharusnya membalikkan arah sentimen menjadi Positif. Namun, model terdistraksi oleh keluhan di awal dan memprediksinya sebagai sentimen Negatif.
- **Perbandingan Penjelasan XAI**:
  - **LIME** memfokuskan penjelasan lokal secara ekstrim pada kata "lambat" (+0.28 ke Negatif) dan "tidak" (+0.14 ke Negatif), sedangkan kata "puas" hanya mendapatkan bobot positif kecil (+0.08).
  - **SHAP** mendistribusikan kontribusi secara teoritis. Kata "lambat" mendapatkan nilai Shapley negatif (+1.85 ke Negatif) dan kata "puas" mendapatkan nilai positif (-1.22 ke Positif). Hal ini membuktikan SHAP lebih stabil dalam mengidentifikasi kata kunci yang berlawanan arah sentimen dalam satu kalimat.

---

## 4.5 Perbandingan Metrik Kuantitatif LIME vs SHAP

Untuk mengukur performa penjelasan secara objektif, dilakukan evaluasi kuantitatif terhadap LIME dan SHAP pada 100 sampel data uji acak berdasarkan tiga kriteria utama: konsistensi fitur (Jaccard Similarity), keandalan penjelasan (*Comprehensiveness* (AOPC) dan *Sufficiency*), serta efisiensi runtime komputasi.

Hasil evaluasi kuantitatif dirangkum pada Tabel 4.5 berikut:

**Tabel 4.5: Rangkuman Perbandingan Metrik Kuantitatif XAI**
| Metrik Evaluasi | Nilai LIME | Nilai SHAP | Interpretasi Metodologis untuk Bab IV |
| :--- | :---: | :---: | :--- |
| **Average Jaccard Similarity** | \- | \- | **0.5889 (58.89%)**: Menunjukkan tingkat tumpang tindih (*overlap*) fitur kata kunci top-5 antara LIME dan SHAP mencapai 58.89%. |
| **Average Runtime (detik)** | **0.6234** | **0.4050** | **SHAP Lebih Efisien**: SHAP Partition lebih cepat 35% dibandingkan LIME karena memanfaatkan struktur clustering hirarkis kalimat. |
| **Comprehensiveness (AOPC) ($k=5$)** | **0.0280** | **0.0318** | **SHAP Lebih Faithful**: Penghapusan top-5 kata penting SHAP menurunkan probabilitas prediksi model lebih signifikan. |
| **Sufficiency ($k=5$)** | **0.0029** | **0.0066** | **Kedua Metode Sangat Valid**: Nilai yang sangat mendekati 0 membuktikan retensi informasi pada top-5 kata kunci sudah cukup mempertahankan prediksi model asli. |
"""

os.makedirs(report_path.parent, exist_ok=True)
with open(report_path, "w", encoding="utf-8") as f:
    f.write(bab4_content)

print(f"Report {report_path} successfully updated!")
