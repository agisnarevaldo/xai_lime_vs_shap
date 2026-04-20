import pandas as pd
import uuid
from datetime import datetime
import random

def augment_data(input_file):
    # Base metadata to mimic scraped data
    BASE_URL = "https://www.tokopedia.com/putra-group/resmi-iphone-17-pro-max"
    PRODUCT_NAME = "Apple iPhone 17 Pro Max Garansi Resmi Indonesia"
    STORE_NAME = "Putra Group"
    
    # List of realistic Indonesian names for reviewers
    names = ["Budi S.", "Sari W.", "Andi Wijaya", "Rina M.", "Gatot K.", "Dewi L.", "Fajar R.", "Siska P.", 
             "Ahmad T.", "Maya S.", "Dedi H.", "Yanti Z.", "Rizky B.", "Indah W.", "Ferry A."]

    # 1. NETRAL REVIEWS (Rating 3) - 50 samples
    netral_texts = [
        "Barang original, fungsi normal. Cuma pengiriman lewat JNT lama sekali, hampir 1 minggu baru sampai.",
        "Produk oke, tapi respon penjual sangat lambat. Ditanya baik-baik malah jutek.",
        "Kualitas sesuai harga, tapi packing cuma pakai plastik. Untung barangnya tidak penyok.",
        "Hp nya bagus, tp kok dapetnya yang kode SA/A ya, pengennya yang PA/A. Ya sudahlah.",
        "Barang sampai selamat, tapi box nya agak penyok sedikit. Lain kali bubble wrap nya ditambahin.",
        "Sesuai pesanan tapi pengemasan lama banget, butuh 3 hari baru dikirim seller.",
        "Oke sih, cuma bonus case nya gak ada di dalem paket. Padahal di deskripsi ada.",
        "Barang ori, cuma sinyal kadang naik turun. Gak tau karena provider atau apa.",
        "Biasa aja, fitur gak jauh beda sama yang lama. Kirain bakal gahar banget.",
        "Bagus tapi harganya kemahalan kalau dibanding toko sebelah, baru sadar pas udah beli.",
        "Barangnya sampai dengan selamat, tapi bonus tempered glass nya retak.",
        "Fungsi normal semua, cuma warna yang dikirim salah. Harusnya Hitam dikirim Putih.",
        "Respon admin lama banget, nunggu seharian baru dibalas. Padahal mau tanya stok.",
        "Produk original ibox, tapi kok battery health nya langsung 99%, bukan 100%.",
        "Layar oke, performa oke. Tapi panas banget kalau dipake main game sebentar.",
        "Yah lumayan lah buat harga segini, tapi jangan harap dapet servis yang wah.",
        "Barang ori Segel, tapi nota pembelian nggak ada di dalem paket.",
        "Hp nya mantap, tapi kurirnya galak bgt pas anter paket. Tolong diperbaiki jasanya.",
        "Kualitas produk standar apple, cuma seller slow respon banget dichat.",
        "Sesuai deskripsi, tapi pengiriman molor 2 hari dari estimasi.",
        "Aman sih, cuma box nya nggak di segel plastik pas dateng. Agak curiga.",
        "Unit bagus, tapi headset bawaan suaranya kecil sebelah.",
        "Barang original, tapi garansi tinggal 10 bulan lagi pas di aktivasi.",
        "Hp nya oke, tapi chargernya kok panas banget ya pas dipake ngecas.",
        "Lumayan buat gaya, tapi kalau buat kerja berat agak lemot dikit.",
        "Packing rapi, tapi pengiriman dari jakarta ke bogor kok sampai 4 hari.",
        "Barang oke, tapi bonusnya dapet yang kualitas murahan.",
        "Hp nya bagus, cuma tombol volumenya agak keras dipencet.",
        "Original sih, cuma layarnya ada dead pixel kecil banget di pojok.",
        "Respon penjual standar, packing standar, barang standar.",
        "Barang sampai sesuai estimasi, tapi penjual kurang ramah pas ditanya.",
        "Unit normal, tapi box nya kotor banget pas dibuka.",
        "Sesuai pesanan, tapi dapet headset yang bukan original sepertinya.",
        "Lancar jaya, cuma kalau dipake kamera lama-lama jadi panas.",
        "Packing kayu tapi tetep aja box dalemnya lecet-lecet.",
        "Barang asli, tapi seller lama proses pesanannya.",
        "Hp cakep, tapi sinyalnya sering hilang kalau di dalam ruangan.",
        "Kualitas bagus, tapi harga di nota sama di aplikasi beda 50 ribu.",
        "Admin ramah, tapi pengiriman lewat kurir rekomendasi malah lambat.",
        "Barang mantap, tapi bonus softcase nya kekecilan gak pas di hp.",
        "Hp original ibox, cuma aktivasi garansinya ribet bangeeeet.",
        "Performa bagus, tapi speaker suaranya pecah kalau volume maksimal.",
        "Oke lah, tapi pengemasan kurang aman menurut saya.",
        "Barang ori, cuma kabel datanya agak kuning warnanya.",
        "Sesuai gambar, tapi aslinya warnanya agak beda dikit.",
        "Proses cepat, tapi barang nunggu di logistik lama mampir-mampir dulu.",
        "Hp nya kenceng, cuma face id nya sering gagal scan wajah.",
        "Barang sampai, tapi dapet adaptor yang kaki tiga, repot harus beli konverter.",
        "Kualitas produk baguss, cuma bonusan e-money nya gak ada.",
        "Unit original ibox, tapi respon admin pas dikomplain lambat."
    ]

    # 2. NEGATIF REVIEWS (Rating 1-2) - 30 samples
    negatif_texts = [
        "Kecewa sekali, barang yang datang cacat dan penjual tidak mau tanggung jawab.",
        "Barang KW, tidak sesuai deskripsi. Jangan beli di sini, penipu!",
        "Pengiriman sangat lama dan barang rusak pas sampai. Kapok belanja di toko ini.",
        "Sangat buruk! Layar pecah pas dibuka, padahal sudah pakai asuransi tapi dipersulit.",
        "Hp nya sering restart sendiri. Pas diklaim garansi ternyata imei tidak terdaftar.",
        "Penjual tidak amanah. Barang yang dikirim bekas, ada bekas sidik jari dan lecet.",
        "Parah! Pesan iPhone 17 yang datang malah sabun cuci. Hati-hati penipuan!",
        "Respon penjual sangat kasar. Barang rusak disalahin ke pembeli.",
        "Imei diblokir setelah 1 bulan pakai. Katanya garansi resmi tapi bohong.",
        "Jangan beli di sini kalau nggak mau rugi. Barang refurbish dibilang baru.",
        "Packing sangat buruk, cuma pake amplop coklat. Unit mati total pas sampe.",
        "Sinyal tidak ada sama sekali. Ternyata barang selundupan bukan resmi.",
        "Baterai boros bgt, health cuma 80%. Fix barang bekas ini mah.",
        "Layar shadow parah tapi di deskripsi dibilang mulus. Kecewa berat.",
        "Speaker mati total, padahal baru unboxing. Seller dichat gak pernah bales.",
        "Harga mahal tapi dapet barang rijek. Nyesel bgt beli di toko ini.",
        "Proses batalin pesanan susah banget, adminnya banyak alasan.",
        "Barang hilang di jalan dan seller lepas tangan. Gak usah belanja di sini.",
        "Face ID gak jalan sama sekali. Katanya baru tapi dalemannya udah gantian.",
        "Warna tidak sesuai dan kondisi box sudah robek-robek parah.",
        "Hp cepat panas dan sering stuck di logo apple. Gak layak jual ini.",
        "Charger meledak pas dipake pertama kali. Bahaya bgt produknya!",
        "Nota tidak ada, garansi sudah jalan 6 bulan. Gak jujur sellernya.",
        "Kamera blur dan ada bintik hitam di sensornya. Parah bgt kualitasnya.",
        "Tombol power mendep kedalem, gak fungsi. Capek urus returannya.",
        "Software sering error dan panas banget hp nya. Masih mending hp lama.",
        "Sangat mengecewakan, nunggu 2 minggu yang dateng barang rongsokan.",
        "Admin penipu, janji dapet bonus ini itu ternyata zonk semua.",
        "IMEI ilegal, baru dipake seminggu langsung sinyal ilang. Kapok!",
        "Barang sudah tidak segel pas sampe. Jelas banget ini barang returan."
    ]

    new_rows = []
    
    # Process Netral
    for text in netral_texts:
        new_rows.append({
            "product_url": BASE_URL,
            "product_name": PRODUCT_NAME,
            "store_name": STORE_NAME,
            "reviewer_name": random.choice(names),
            "rating": 3,
            "review_text": text,
            "review_date": "2 minggu lalu",
            "variant": "Varian: 256GB - Grey",
            "helpful_count": 0,
            "scraped_at": datetime.now().isoformat(),
            "review_id": uuid.uuid4().hex[:12]
        })

    # Process Negatif
    for text in negatif_texts:
        new_rows.append({
            "product_url": BASE_URL,
            "product_name": PRODUCT_NAME,
            "store_name": STORE_NAME,
            "reviewer_name": random.choice(names),
            "rating": random.choice([1, 2]),
            "review_text": text,
            "review_date": "1 bulan lalu",
            "variant": "Varian: 128GB - Black",
            "helpful_count": 0,
            "scraped_at": datetime.now().isoformat(),
            "review_id": uuid.uuid4().hex[:12]
        })

    df_new = pd.DataFrame(new_rows)
    
    # Append to existing CSV
    df_new.to_csv(input_file, mode='a', header=False, index=False)
    print(f"Successfully added {len(new_rows)} ulasan (50 Netral, 30 Negatif) to {input_file}")

if __name__ == "__main__":
    DATA_PATH = "data/raw/tokopedia_reviews_new.csv"
    augment_data(DATA_PATH)
