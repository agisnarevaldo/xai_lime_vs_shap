## Plan: Scraping iPhone 17 dari Tokopedia

Melakukan web scraping 20 produk iPhone 17 dari Tokopedia menggunakan Playwright dengan stealth plugin untuk mengatasi proteksi anti-bot. Data akan disimpan dalam format CSV menggunakan pandas.

### Data yang Akan Di-scrape

| No | Field | Deskripsi |
|----|-------|-----------|
| 1 | `product_name` | Nama lengkap produk |
| 2 | `price` | Harga produk (dalam Rupiah) |
| 3 | `store_name` | Nama toko penjual |
| 4 | `store_location` | Lokasi toko |
| 5 | `rating` | Rating produk (1-5) |
| 6 | `review_count` | Jumlah review produk |
| 7 | `sold_count` | Jumlah produk terjual |
| 8 | `variants` | Varian tersedia (warna, kapasitas) |
| 9 | `stock_status` | Status ketersediaan stok |
| 10 | `description` | Deskripsi produk |
| 11 | `specifications` | Spesifikasi teknis |
| 12 | `images` | URL gambar produk |
| 13 | `url` | Link produk original |
| 14 | `scraped_at` | Timestamp scraping |

### Steps

1. **Setup environment dan dependencies** - Install `playwright`, `playwright-stealth`, `pandas`, dan `fake-useragent` di [scrape.ipynb](scrape.ipynb)

2. **Baca URL dari file sumber** - Load 20 URL dari [link.txt](link.txt) menggunakan Python file handling

3. **Konfigurasi browser Playwright** - Setup browser headful (non-headless) dengan stealth plugin, locale Indonesia, dan viewport realistic untuk bypass anti-bot Tokopedia

4. **Implementasi fungsi scraping** - Buat async function untuk extract semua 14 field data dari setiap halaman produk dengan teknik:
   - CSS selector untuk elemen visible
   - XPath untuk struktur kompleks
   - JavaScript evaluation untuk data dinamis
   - Scroll halaman untuk trigger lazy-load content

5. **Validasi kelengkapan data** - Cek setiap field terisi, log warning jika ada field kosong, dan retry jika data tidak lengkap

6. **Eksekusi scraping dengan delay** - Loop semua URL dengan random delay 5-10 detik antar request untuk menghindari rate limiting

7. **Hitung statistik review** - Agregasi total review dari semua produk, hitung rata-rata rating, dan identifikasi produk dengan review terbanyak/tersedikit

8. **Export data ke CSV** - Simpan hasil scraping ke pandas DataFrame lalu export ke file CSV untuk analisis

### Validasi Data Maksimal

- [ ] Semua 20 URL berhasil di-scrape
- [ ] Setiap produk memiliki minimal: nama, harga, toko, rating, review_count
- [ ] Tidak ada field kosong tanpa alasan valid
- [ ] Data review_count adalah angka valid (bukan string)
- [ ] Timestamp scraping tercatat untuk setiap record

### Further Considerations

1. **Headless vs Headful browser?** Rekomendasi: gunakan headful (non-headless) karena Tokopedia lebih mudah mendeteksi headless browser

2. **Perlu proxy rotating?** Untuk 20 URL dengan delay yang cukup, proxy tidak wajib. Namun jika sering terblokir, pertimbangkan residential proxy Indonesia

3. **Error handling strategy?** Implementasi retry mechanism (max 3x) jika halaman gagal load atau CAPTCHA muncul

4. **Data completeness check?** Implementasi validasi sebelum save - jika field penting kosong, retry scraping halaman tersebut
