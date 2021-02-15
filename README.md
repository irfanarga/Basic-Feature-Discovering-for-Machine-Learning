# Basic Feature Discovering for Machine Learning
Project ini bertujuan untuk membandingkan akurasi model antara dataset yang telah ditambahkan Feature Engineering (FE) dan tanpa FE.  

### Library yang digunakan yaitu:
- Pandas untuk proses dataframe dan csv
- Matplotlib untuk plotting grafik
- Seaborn untuk plotting grafik
- Sklearn untuk machine learning model

### Dataset
Dataset ini berisi data dari semua orang yang ikut di dalam Kapal Titanic. Kolom Survived dijadikan variabel target dan semua kolom lain digunakan sebagai penentu apakah penumpang selamat atau tidak.

### Data cleansing and Correlation
Filling Missing Value

![](https://github.com/irfanarga/Basic-Feature-Discovering-for-Machine-Learning/blob/master/Missing%20value.png)

Ada beberapa data yang kosong di kolom Embarked, Age dan Cabin

- Missing value in the Age data is filled with the median value of Age data based on passenger class (Pclass) and Sex
- For Embarked, most of the people from Titanic depart from Southampton/S, so we can fill it with S.
- Missing value in the Fare data is filled with the median value of Fare data based on passenger class (Pclass), Parch, and SibSp

Correlation

![](https://github.com/irfanarga/Basic-Feature-Discovering-for-Machine-Learning/blob/master/Correlation.png)

Nilai korelasi mendekati 1 untuk korelasi positif dan -1 untuk korelasi negatif/terbalik. Pada data ini dapat dilihat bahwa variabel Survived sangat besar korelasinya dengan Pclass dan Fare. Sedangkan Age sangat berkaitan dengan Pclass, Sibling Spouse (SibSp), Parent Children (Parch). Dapat diasumsikan bahwa kebanyakan orang yang selamat adalah orang dengan PClass atas dan Tuanya umur seseorang dapat dikatakan dia akan membawa saudara/orang tua/anak/pasangan. Dan Fare (harga) tentu saja berkaitan dengan Pclass (kelas penumpang) seorang penumpang. 

### Feature Engineering
The first feature/column created is Family_Size, which is a combination of Parent, Children, Sibling, and Spouse. Then add 1 assuming that person counts himself too.

The second feature/column that is created is to combine Family_Size with its respective groups depending on the number.

The categories are follows:
- Family Size 1 = Alone
- Family Size 2, 3, and 4 = Small
- Family Size 5 and 6 = Medium
- Family Size 7, 8 and 11 = Large

The third feature/column created is Ticket_Frequency with the combined value of the same Ticket.
