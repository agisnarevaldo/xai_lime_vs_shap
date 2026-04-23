# 📄 **4.6 Analisis Perbandingan LIME vs SHAP**

Setelah melakukan interpretasi model menggunakan dua metode *Explainable AI* (XAI) yang berbeda, yaitu LIME dan SHAP, tahap selanjutnya adalah melakukan analisis perbandingan fungsional antara keduanya. Analisis ini bertujuan untuk mengevaluasi konsistensi, keandalan, serta efektivitas masing-masing metode dalam memberikan transparansi terhadap keputusan model IndoBERT.

---

## **4.6.1 Metodologi dan Pendekatan**

Perbedaan mendasar antara LIME dan SHAP terletak pada landasan teoretis yang digunakan untuk menghasilkan penjelasan. LIME (*Local Interpretable Model-agnostic Explanations*) bekerja dengan pendekatan *Local Surrogate*, di mana metode ini menciptakan derau (*noise*) di sekitar sampel ulasan tertentu untuk melatih model linier sederhana yang mampu menangkap perilaku model IndoBERT secara lokal. 

Di sisi lain, SHAP (*SHapley Additive exPlanations*) dibangun di atas landasan teori permainan kooperatif (*Cooperative Game Theory*). SHAP menghitung Nilai Shapley untuk setiap fitur (kata), yang merepresentasikan kontribusi rata-rata fitur tersebut terhadap perubahan prediksi dari nilai rata-rata (*base value*). Pendekatan SHAP memberikan jaminan teoretis mengenai konsistensi dan alokasi kontribusi fitur yang lebih adil dibandingkan LIME.

---

## **4.6.2 Konsistensi Hasil Interpretasi Lokal**

Berdasarkan pengujian pada beberapa sampel ulasan yang sama, kedua metode menunjukkan tingkat konsistensi yang tinggi dalam mengidentifikasi kata kunci yang paling berpengaruh. Hal ini dapat dilihat pada **Gambar 4.9** yang menyajikan perbandingan *side-by-side* antara interpretasi LIME dan SHAP untuk sebuah ulasan positif.

![Perbandingan LIME vs SHAP](file:///d:/xai_lime_vs_shap/outputs/figures/fig4_10_comparison_lime_shap_positive.png)

**Gambar 4.9 Perbandingan Visual Interpretasi LIME dan SHAP pada Sampel Positif**

Pada ulasan tersebut (*"Barang original bergaransi segel, pengiriman cepat..."*), kedua metode secara konsisten menyoroti kata **"original"**, **"cepat"**, dan **"recommended"** sebagai fitur dengan kontribusi tertinggi terhadap prediksi sentimen positif. Konsistensi ini membuktikan bahwa meskipun menggunakan algoritma yang berbeda, kedua metode XAI mampu memvalidasi bahwa model IndoBERT telah mempelajari pola linguistik yang benar dan logis, bukan sekadar mengandalkan korelasi semu pada dataset.

---

## **4.6.3 Cakupan Analisis: Lokal vs Global**

Perbedaan paling signifikan yang ditemukan dalam penelitian ini adalah cakupan analisis yang mampu dihasilkan oleh masing-masing metode.

1.  **LIME (Fokus Lokal)**: Sangat efektif untuk menjelaskan ulasan individu secara spesifik. Visualisasi LIME yang menggunakan *highlighting* pada teks asli memberikan transparansi yang sangat intuitif bagi pengguna akhir untuk memahami alasan di balik label ulasan tertentu.
2.  **SHAP (Lokal dan Global)**: Unggul dalam memberikan pandangan menyeluruh melalui *Global Feature Importance*. Dengan menggunakan *Beeswarm Plot* (Gambar 4.8), SHAP mampu menunjukkan dampak setiap kata terhadap seluruh dataset uji. Analisis global ini sangat krusial dalam penelitian ini karena memungkinkan peneliti untuk mengidentifikasi kata-kata pendorong utama sentimen secara agregat (seperti "cepat", "puas", "kecewa", dan "lambat").

---

## **4.6.4 Performa dan Efisensi Komputasi**

Dalam aspek waktu eksekusi, ditemukan perbedaan performa yang cukup kontras. LIME cenderung jauh lebih cepat dalam menghasilkan penjelasan untuk satu ulasan karena hanya memproses area di sekitar ulasan tersebut. Sebaliknya, perhitungan Nilai Shapley pada SHAP memerlukan waktu komputasi yang lebih intensif, terutama ketika menghasilkan penjelasan global yang melibatkan ribuan permutasi fitur ulasan. Namun, untuk dataset dengan ukuran sedang (seperti pada penelitian ini), waktu komputasi SHAP masih berada dalam batas yang dapat diterima.

---

## **4.6.5 Kesimpulan Komprehensif Perbandingan**

Secara ringkas, kedua metode ini bersifat saling melengkapi (*complementary*). LIME memberikan kejelasan visual yang sangat baik untuk interpretasi mendalam terhadap kasus-kasus spesifik yang membingungkan bagi model (seperti ulasan netral). Sementara itu, SHAP memberikan validasi statistik dan teoretis yang kuat serta mampu merangkum pola umum yang dipelajari model dari seluruh dataset.

Perbandingan komprehensif dari kedua fitur tersebut dirangkum dalam **Tabel 4.10**.

**Tabel 4.10 Ringkasan Perbandingan Karakteristik LIME dan SHAP**

| Fitur Perbandingan | LIME | SHAP |
| :--- | :--- | :--- |
| **Landasan Teori** | *Local Surrogate Models* (Linear) | *Cooperative Game Theory* (Shapley Values) |
| **Cakupan Penjelasan** | Hanya Lokal (per sampel) | Lokal dan Global (seluruh dataset) |
| **Keandalan/Stabilitas** | Cenderung variatif (stokastik) | Sangat stabil dan konsisten secara teoretis |
| **Visualisasi** | Sangat intuitif (teks highlighting) | Mendalam (*Force plot, Beeswarm, Waterfall*) |
| **Waktu Komputasi** | Cepat (efisien untuk ulasan tunggal) | Cukup Intensif (lambat untuk skala besar) |
| **Agnostik Model** | Ya (semua jenis model) | Ya (semua jenis model) |

Penggunaan kedua metode ini secara bersamaan terbukti berhasil memberikan transparansi penuh terhadap model IndoBERT, sekaligus memastikan bahwa interpretasi yang dihasilkan tidak bersifat bias dan memiliki dasar yang kuat baik secara lokal maupun global.
