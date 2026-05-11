# Tugas Pengantar Data Science

## Identitas

- **Nama:** Nabil Fakhrezy
- **NIM:** 240401010286
- **Program Studi:** Informatika

---

## Link Google Colab

### Pertemuan 2
https://colab.research.google.com/drive/1f_xkcKtYu8aEAJ9owc-flA1IXRfWm2tH?usp=sharing

### Pertemuan 3
https://colab.research.google.com/drive/1WYNopoxmZvrx3gIMZRbnf_85U8Duo0oj?usp=sharing

### Pertemuan 4
https://colab.research.google.com/drive/1kFyxf9l2RvjxqJ3uZK-BZWRMBASANQIN?usp=sharing

---

## Pertemuan 2

Pada pertemuan ini dilakukan analisis dataset Titanic menggunakan Python, NumPy, dan Pandas.

### Q1. Jumlah Baris dan Kolom

**Pertanyaan:** Berapa jumlah baris dan kolom pada dataset Titanic? Kolom apa saja yang tersedia?

**Kode:**
```python
print("Jumlah baris dan kolom:", df.shape)
print("Kolom yang tersedia:")
print(df.columns.tolist())
```

**Jawaban:** Dataset Titanic memiliki 891 baris dan 12 kolom. Kolom yang tersedia yaitu PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, dan Embarked.

### Q2. Missing Values

**Pertanyaan:** Kolom mana saja yang memiliki missing values? Berapa jumlah dan proporsinya?

**Kode:**
```python
missing = df.isnull().sum()
missing_percent = (missing / len(df) * 100).round(2)
hasil_missing = pd.DataFrame({"Jumlah Missing": missing, "Persentase (%)": missing_percent})
hasil_missing[hasil_missing["Jumlah Missing"] > 0]
```

**Jawaban:** Kolom yang memiliki missing values adalah Age, Cabin, dan Embarked. Missing values paling banyak terdapat pada kolom Cabin.

### Q3. Persentase Penumpang Selamat

**Pertanyaan:** Berapa persen penumpang yang selamat?

**Kode:**
```python
persen_selamat = df["Survived"].mean() * 100
print(f"Persentase penumpang yang selamat: {persen_selamat:.2f}%")
```

**Jawaban:** Persentase penumpang yang selamat adalah sekitar 38,38%.

### Q4. Penumpang Wanita Kelas 1

**Pertanyaan:** Filter penumpang wanita dari kelas 1. Berapa jumlahnya dan berapa rata-rata usianya?

**Kode:**
```python
wanita_kelas1 = df[(df["Sex"] == "female") & (df["Pclass"] == 1)]
jumlah_wanita_kelas1 = len(wanita_kelas1)
rata_usia_wanita_kelas1 = wanita_kelas1["Age"].mean()
print("Jumlah penumpang wanita kelas 1:", jumlah_wanita_kelas1)
print(f"Rata-rata usia penumpang wanita kelas 1: {rata_usia_wanita_kelas1:.2f}")
```

**Jawaban:** Jumlah penumpang wanita kelas 1 adalah 94 orang, dengan rata-rata usia sekitar 34,61 tahun.

### Q5. Tingkat Keselamatan per Kelas

**Pertanyaan:** Hitung tingkat keselamatan per kelas penumpang. Kelas mana yang paling tinggi?

**Kode:**
```python
tingkat_selamat_kelas = df.groupby("Pclass")["Survived"].mean() * 100
tingkat_selamat_kelas = tingkat_selamat_kelas.round(2)
print(tingkat_selamat_kelas)
```

**Jawaban:** Kelas penumpang dengan tingkat keselamatan tertinggi adalah kelas 1.

### Q6. Analisis Tambahan

**Pertanyaan:** Buat analisis tambahan berdasarkan variabel lain yang dianggap menarik.

**Kode:**
```python
survival_by_gender = df.groupby("Sex")["Survived"].mean() * 100
survival_by_gender = survival_by_gender.round(2)
print(survival_by_gender)
```

**Jawaban:** Penumpang perempuan memiliki tingkat keselamatan lebih tinggi dibandingkan penumpang laki-laki.

---

## Pertemuan 3

Pada pertemuan ini dilakukan proses data cleaning pada dataset housing_dirty.csv.

### 1. Load Dataset

**Kode:**
```python
!gdown "https://drive.google.com/uc?id=1LfQWProB0VjWN5q8bKuRIgn-stULfIRo" -O housing_dirty.csv
df = pd.read_csv("housing_dirty.csv")
print("Shape awal:", df.shape)
df.head()
```

**Jawaban:** Dataset berhasil dibaca dan memiliki 130 baris serta 7 kolom.

### 2. Eksplorasi Awal Dataset

**Kode:**
```python
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

**Jawaban:** Dataset masih memiliki missing values pada beberapa kolom dan terdapat nilai yang tidak wajar.

### 3. Menghapus Data Duplikat

**Kode:**
```python
df = df.drop_duplicates()
print("Shape setelah hapus duplikat:", df.shape)
print("Jumlah duplikat setelah dibersihkan:", df.duplicated().sum())
```

**Jawaban:** Data duplikat dihapus agar tidak mempengaruhi hasil analisis.

### 4. Normalisasi Format String

**Kode:**
```python
df["kota"] = df["kota"].astype(str).str.strip().str.title()
df["kondisi"] = df["kondisi"].astype(str).str.strip().str.lower()
df.head()
```

**Jawaban:** Format teks dibuat seragam agar lebih rapi.

### 5. Imputasi Missing Values

**Kode:**
```python
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include="object").columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])
print(df.isnull().sum())
```

**Jawaban:** Missing values pada kolom numerik diisi dengan median, sedangkan kolom kategorik diisi dengan modus.

### 6. Menangani Outlier dengan IQR

**Kode:**
```python
def capping_iqr(data, kolom):
    Q1 = data[kolom].quantile(0.25)
    Q3 = data[kolom].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data[kolom] = data[kolom].clip(lower, upper)
    return data
```

**Jawaban:** Outlier dibatasi menggunakan metode IQR.

### 7. Validasi Akhir dan Export

**Kode:**
```python
print("Total missing values:", df.isnull().sum().sum())
print("Total duplikat:", df.duplicated().sum())
df.to_csv("housing_clean.csv", index=False)
```

**Jawaban:** Data akhir divalidasi, kemudian disimpan menjadi housing_clean.csv.

### 8. Akses REST API Publik

**Kode:**
```python
import requests
from pandas import json_normalize
url_api = "https://jsonplaceholder.typicode.com/users"
response = requests.get(url_api, timeout=10)
data = response.json()
df_api = json_normalize(data, sep="_")
display(df_api[["id", "name", "email", "address_city"]])
```

**Jawaban:** Data dari API berhasil diambil dalam format JSON dan diubah menjadi DataFrame Pandas.

---

## Pertemuan 4

Pada pertemuan ini dilakukan eksplorasi statistik dasar dan analisis data menggunakan dataset Iris.

### 1. Load dan Inspect Dataset

**Kode:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = sns.load_dataset('iris')
print('Shape:', df.shape)
print(df.dtypes)
display(df.head())
display(df.describe().round(3))
```

**Jawaban:** Dataset Iris memiliki 150 baris dan 5 kolom. Empat kolom berupa numerik dan satu kolom berupa kategori species.

### 2. Statistik Deskriptif Lengkap

**Kode:**
```python
for col_name in df.select_dtypes(include='number').columns:
    col = df[col_name]
    print(f'\\n=== {col_name} ===')
    print(f'Mean      : {col.mean():.3f}')
    print(f'Median    : {col.median():.3f}')
    print(f'Std Dev   : {col.std():.3f}')
    print(f'Varians   : {col.var():.3f}')
    print(f'Skewness  : {col.skew():.3f}')
    print(f'Kurtosis  : {col.kurt():.3f}')
```

**Jawaban:** Statistik deskriptif digunakan untuk melihat pemusatan, penyebaran, kemiringan, dan bentuk distribusi data pada setiap kolom numerik.

### 3. Histogram dan KDE Sepal Length

**Kode:**
```python
sns.histplot(df['sepal_length'], kde=True, bins=20)
plt.axvline(df['sepal_length'].mean(), linestyle='--', label='Mean')
plt.axvline(df['sepal_length'].median(), linestyle='--', label='Median')
plt.legend()
plt.show()
```

**Jawaban:** Distribusi sepal_length cukup mendekati simetris karena nilai mean dan median tidak terlalu jauh.

### 4. Boxplot dan Violin Plot per Spesies

**Kode:**
```python
sns.boxplot(data=df, x='species', y='petal_length')
plt.show()
sns.violinplot(data=df, x='species', y='petal_length', inner='box')
plt.show()
```

**Jawaban:** Spesies setosa memiliki petal_length paling kecil, sedangkan virginica cenderung memiliki nilai paling besar.

### 5. Matriks Korelasi Pearson

**Kode:**
```python
corr = df.drop('species', axis=1).corr(method='pearson')
display(corr.round(3))
```

**Jawaban:** Korelasi tertinggi terdapat pada pasangan petal_length dan petal_width.

### 6. Scatter Plot dan Heatmap

**Kode:**
```python
sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species')
sns.regplot(data=df, x='petal_length', y='petal_width', scatter=False)
plt.show()
sns.heatmap(corr, annot=True, fmt='.2f', vmin=-1, vmax=1)
plt.show()
```

**Jawaban:** Scatter plot menunjukkan hubungan positif yang kuat antara petal_length dan petal_width, dan heatmap memperkuat hasil tersebut.

## Kesimpulan

Dari praktikum ini, dapat disimpulkan bahwa proses pengolahan dan analisis data perlu dilakukan secara bertahap, mulai dari membaca dataset, mengecek data, membersihkan data, menghitung statistik deskriptif, sampai membuat visualisasi untuk memahami pola data.
