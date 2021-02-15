# Basic Feature Discovering for Machine Learning
## Apakah feature engineering dapat meningkatkan akurasi model?
Project ini bertujuan untuk menjawab pertanyaan tersebut dengan membandingkan akurasi model antara dataset yang telah ditambahkan Feature Engineering (FE) dan tanpa FE.  

### Library yang digunakan yaitu:
- Pandas untuk proses dataframe dan csv
- Matplotlib untuk plotting grafik
- Seaborn untuk plotting grafik
- Sklearn untuk machine learning model

### Dataset
Dataset ini berisi data dari semua orang yang ikut di dalam Kapal Titanic. Kolom Survived dijadikan variabel target dan semua kolom lain digunakan sebagai penentu apakah penumpang selamat atau tidak.

### Data cleansing and Correlation
Mengisi Missing Value

![](https://github.com/irfanarga/Basic-Feature-Discovering-for-Machine-Learning/blob/master/Missing%20value.png)

Ada beberapa data yang kosong di kolom Embarked, Age dan Cabin

- Missing value pada kolom Age, karena berkaitan dengan kelas penumpang, maka dapat diisi dengan nilai tengah umur seseorang di dalam kelas tersebut berdasarkan jenis kelaminnya.
- Missing value pada kolom Embarked diisi dengan "S" karena kebanyakan orang di Kapal Titanic berangkat dari Southampton.
- Missing value pada kolom Fare, karena sangat berkaitan dengan Kelas Penumpang, Jumlah Parent/Children, dan Jumlah Sibling/Spouse maka dapat diisi dengan nilai tengah dari tarif seseorang dalam grup tersebut. 

Korelasi

![](https://github.com/irfanarga/Basic-Feature-Discovering-for-Machine-Learning/blob/master/Correlation.png)

Nilai korelasi mendekati 1 untuk korelasi positif dan -1 untuk korelasi negatif/terbalik. Pada data ini dapat dilihat bahwa variabel Survived sangat besar korelasinya dengan Pclass dan Fare. Sedangkan Age sangat berkaitan dengan Pclass, Sibling Spouse (SibSp), Parent Children (Parch). Dapat diasumsikan bahwa kebanyakan orang yang selamat adalah orang dengan PClass atas dan Tuanya umur seseorang dapat dikatakan dia akan membawa saudara/orang tua/anak/pasangan. Dan Fare (harga) tentu saja berkaitan dengan Pclass (kelas penumpang) seorang penumpang. 

### Feature Engineering
Fitur pertama yang dibuat adalah Family_Size, sesuai namanya ini adalah gabungan dari Parent, Children, Sibling, dan Spouse. Lalu kita tambahkan 1 dengan asumsi menghitung diri orang itu juga.

Fitur kedua yang dibuat adalah menggabungkan Family_Size dengan groupnya masing-masing tergantung jumlahnya.

Kategorinya adalah seperti berikut:
- Family Size 1 = Alone
- Family Size 2, 3, and 4 = Small
- Family Size 5 and 6 = Medium
- Family Size 7, 8 and 11 = Large

Fitur ketiga yang dibuat adalah Ticket_Frequency dengan nilai gabungan dari Ticket yang sama.

Fitur ketiga adalah Title yang berisi jabatan seseorang. Paling umum adalah Mr, Mrs, dan Miss.

Fitur keempat adalah Is_Maried yang berisi status pernikahan jenis kelamin perempuan. Dapat diketahui dari statusnya yaitu Mrs.

## Hasil model Random Forest Classifier dengan perhitungan akurasi cross_val_score
Dataset dengan FE

![](https://github.com/irfanarga/Basic-Feature-Discovering-for-Machine-Learning/blob/master/Model%20accuration%20with%20FE.png)

Dataset tanpa FE

![](https://github.com/irfanarga/Basic-Feature-Discovering-for-Machine-Learning/blob/master/Model%20accuration%20without%20FE.png)

