# 📄 **4.4 Evaluasi Performa Model**

Setelah proses pelatihan selesai, tahap selanjutnya adalah mengevaluasi performa model menggunakan data uji (*test set*). Evaluasi ini bertujuan untuk mengukur sejauh mana model mampu melakukan klasifikasi sentimen secara tepat pada data yang belum pernah dipelajari sebelumnya.

---

## **4.4.1 Hasil Classification Report**

Performa model dievaluasi menggunakan metrik *Precision*, *Recall*, dan *F1-Score*. Metrik ini memberikan gambaran detail mengenai kemampuan model dalam mengklasifikasikan masing-masing kelas sentimen, terutama pada kelas minoritas (Netral dan Negatif). Pengujian dilakukan pada data uji murni (312 ulasan) yang dipisahkan sebelum proses augmentasi data dilakukan.

Hasil evaluasi ditunjukkan pada Tabel 4.10 berikut.

**Tabel 4.10 Classification Report Model IndoBERT (Metode Valid)**

| Kelas Sentimen | Precision | Recall | F1-Score | Support |
| -------------- | --------- | ------ | -------- | ------- |
| Positif        | 0.99      | 0.96   | 0.97     | 281     |
| Negatif        | 0.64      | 0.82   | 0.72     | 17      |
| Netral         | 0.53      | 0.64   | 0.58     | 14      |
|                |           |        |          |         |
| **Accuracy**   |           |        | **0.94** | 312     |
| **Macro Avg**  | 0.72      | 0.81   | 0.76     | 312     |
| **Weighted Avg**| 0.95      | 0.94   | 0.94     | 312     |

Berdasarkan Tabel 4.10, diperoleh nilai akurasi keseluruhan sebesar **94%**. Hasil ini menunjukkan bahwa model memiliki kemampuan generalisasi yang sangat kuat. Meskipun dataset asli memiliki ketidakseimbangan kelas yang ekstrim, penggunaan teknik **Easy Data Augmentation (EDA)** pada set pelatihan terbukti efektif meningkatkan sensitivitas model terhadap kelas minoritas. Hal ini ditunjukkan oleh nilai *Recall* kelas Negatif sebesar **0.82**, yang berarti model berhasil mendeteksi sebagian besar ulasan bernada ketidakpuasan meskipun data aslinya sangat terbatas.

---

## **4.4.2 Confusion Matrix**

Visualisasi *Confusion Matrix* digunakan untuk melihat distribusi prediksi model dibandingkan dengan label sebenarnya (*ground truth*). Hal ini berguna untuk mengidentifikasi kesalahan klasifikasi yang paling sering terjadi.

Distribusi prediksi model ditampilkan pada Gambar 4.6.

![Confusion Matrix IndoBERT](../figures/fig4_6_confusion_matrix.png)

**Gambar 4.6 Confusion Matrix - IndoBERT Sentiment Analysis (Data Murni)**

Berdasarkan Gambar 4.6, terlihat bahwa diagonal utama sangat dominan, menandakan tingkat presisi model yang tinggi secara keseluruhan. Kesalahan klasifikasi yang masih terjadi (False Positif/Negatif) sebagian besar melibatkan kelas **Netral**. Hal ini disebabkan oleh ambiguitas semantik pada ulasan pelanggan yang seringkali mengandung kata-kata pujian namun diikuti oleh keluhan teknis, atau sebaliknya.

---

## **4.4.3 Analisis Validitas Metodologi**

Tingkat akurasi **94%** ini dianggap valid secara ilmiah karena mematuhi prinsip-prinsip penelitian NLP sebagai berikut:
1.  **Mencegah Data Leakage**: Data uji dipisahkan secara murni di awal sebelum proses augmentasi EDA dilakukan. Model tidak pernah melihat variasi dari data uji selama proses pelatihan.
2.  **Evaluasi Realistis**: Model diuji pada data asli (hasil scraping manusia) tanpa manipulasi sintetis, sehingga mencerminkan performa model di dunia nyata.
3.  **Respon Terhadap Ketidakseimbangan**: Dengan F1-Score **0.72** pada kelas Negatif, penelitian ini membuktikan bahwa strategi augmentasi EDA pada data latih mampu mengatasi masalah *imbalanced data* tanpa mengorbankan akurasi keseluruhan.

Hasil evaluasi yang solid ini memberikan fondasi yang sangat kuat untuk melanjutkan ke tahap interpretasi model menggunakan XAI (LIME vs SHAP) guna membedah "kotak hitam" model IndoBERT.
