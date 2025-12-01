## 💻 Tugas 1: Implementasi Smoothing dan Blurring Real-time

Tugas ini berfokus pada penerapan teknik dasar pengolahan citra (**Smoothing/Blurring** dan **Sharpening**) langsung pada latar belakang animasi VTuber secara *real-time*.

### Konsep Utama
Inti dari operasi ini adalah **Konvolusi**—menggeser **kernel** (matriks filter) di atas piksel citra untuk menghasilkan efek tertentu. Kami menggunakan `cv2.filter2D()` untuk mengimplementasikan ketiga filter, termasuk Gaussian Blur yang wajib menggunakan kernel kustom.

### 🛠️ Mode Filter yang Diimplementasikan

| Tombol | Mode Filter | Efek & Fungsi | Implementasi Kernel |
| :---: | :--- | :--- | :--- |
| **0** | **Normal** | Menampilkan background tanpa filter. | N/A |
| **1** | **Average Blurring 5x5** | Blurring sederhana dengan kernel $5 \times 5$ yang terbobot rata. | `cv2.filter2D()` |
| **Q** | **Average Blurring 9x9** | Blurring sederhana dengan kernel $9 \times 9$. | `cv2.filter2D()` |
| **2** | **Gaussian Blurring** | Blurring yang halus dan natural, menggunakan kernel $9 \times 9$ berbasis Distribusi Gaussian. | `cv2.filter2D()` dengan `cv2.getGaussianKernel()` |
| **3** | **Sharpening** | Meningkatkan ketajaman dan detail (kebalikan dari blurring). | **Kernel Kustom: [0, -1, 0] / [-1, 5, -1] / [0, -1, 0] dengan `cv2.filter2D()`** |


## 🎨 Tugas 2: Interaksi Berbasis Deteksi Warna HSV (RGB)

Tugas ini mengimplementasikan interaksi dinamis pada aplikasi VTuber dengan mendeteksi objek berwarna **Merah, Hijau, dan Biru** di dunia nyata menggunakan ruang warna **HSV (Hue, Saturation, Value)**.

### Konsep Utama
Deteksi warna dilakukan di ruang HSV karena stabilitasnya terhadap perubahan intensitas cahaya. Untuk setiap warna, dibuat *mask* biner, dibersihkan dengan operasi Morfologi (`Opening` dan `Closing`), dan kemudian dihitung konturnya (`cv2.findContours`). Jika kontur objek terdeteksi dengan ukuran yang cukup, aksi pemicu akan diaktifkan.

### 🔄 Aksi Pemicu (Color Reaction)

Aksi pemicu diimplementasikan pada **latar belakang *gradient* avatar** (Canvas Kanan). Ketika sebuah warna terdeteksi, latar belakang akan berubah menjadi warna yang kontras untuk memberikan visual umpan balik yang jelas.

| Warna Terdeteksi | Aksi Pemicu (Perubahan Background) | Teks Status |
| :---: | :--- | :--- |
| **Merah** | Background berubah menjadi Biru (kontras). | OBJEK MERAH TERDETEKSI! |
| **Hijau** | Background berubah menjadi Merah (kontras). | OBJEK HIJAU TERDETEKSI! |
| **Biru** | Background berubah menjadi Hijau (kontras). | OBJEK BIRU TERDETEKSI! |
| **Tidak Ada** | Background kembali ke warna *gradient* default. | Normal |

---
