=================================================================
                    SISTEM TEMU KEMBALI INFORMASI
                    Information Retrieval System
=================================================================

## PENJELASAN SEDERHANA UNTUK ORANG AWAM

### Apa itu Sistem Temu Kembali Informasi?
Bayangkan Anda memiliki perpustakaan dengan ribuan buku, dan Anda ingin mencari buku yang membahas topik tertentu. Sistem ini bekerja seperti pustakawan pintar yang dapat menemukan buku-buku yang paling relevan dengan kata kunci yang Anda berikan.

### Bagaimana Cara Kerjanya?
1. **Input**: Anda memasukkan kata kunci pencarian (misalnya: "machine learning")
2. **Proses**: Sistem membandingkan kata kunci Anda dengan semua dokumen yang ada
3. **Output**: Sistem memberikan daftar dokumen yang paling relevan, diurutkan berdasarkan tingkat kemiripan

### Dua Metode Pencarian yang Digunakan:

**1. Bag-of-Words (BoW) - "Kantong Kata"**
Seperti menghitung berapa kali kata tertentu muncul dalam sebuah dokumen. Semakin sering kata pencarian Anda muncul dalam dokumen, semakin relevan dokumen tersebut.

Contoh sederhana:
- Pencarian: "kucing"
- Dokumen A: menyebut "kucing" 5 kali
- Dokumen B: menyebut "kucing" 2 kali
- Hasil: Dokumen A lebih relevan

**2. TF-IDF (Term Frequency-Inverse Document Frequency) - "Frekuensi Kata Cerdas"**
Metode yang lebih pintar. Tidak hanya menghitung seberapa sering kata muncul, tetapi juga mempertimbangkan seberapa unik kata tersebut. Kata yang jarang muncul di seluruh koleksi dokumen dianggap lebih penting.

Contoh:
- Kata "kucing" muncul di banyak dokumen → bobot rendah
- Kata "quantum" hanya muncul di beberapa dokumen → bobot tinggi

### Sistem Evaluasi - Mengukur Seberapa Baik Pencarian
Seperti nilai rapor untuk sistem pencarian:

- **Presisi**: Dari 10 dokumen yang ditemukan, berapa yang benar-benar relevan?
- **Recall**: Dari semua dokumen relevan yang ada, berapa persen yang berhasil ditemukan?
- **F1-Score**: Nilai gabungan yang menyeimbangkan presisi dan recall

=================================================================

## PENJELASAN TEKNIS

### Arsitektur Sistem

```
streamlit_app.py          # User Interface
       ↓
ir_system.py             # Main Controller
       ↓
ir_logic/
├── preprocessing.py      # Text Preprocessing
├── vectorization.py     # BoW & TF-IDF Implementation
├── evaluation.py        # Metrics Calculation
└── caching.py          # Performance Optimization
```

### 1. PREPROCESSING (preprocessing.py)

**Fungsi**: Membersihkan dan mempersiapkan teks untuk diproses

**Tahapan**:
1. **HTML Tag Removal**: Menghapus tag HTML menggunakan regex
2. **Text Cleaning**: Menghapus karakter non-alfanumerik
3. **Lowercasing**: Mengubah semua huruf menjadi kecil
4. **Tokenization**: Memecah teks menjadi kata-kata individual
5. **Stop Words Removal**: Menghapus kata-kata umum ("dan", "atau", "the", "is")

**Implementasi**:
```python
def preprocess_text(text, language='auto', use_spacy=True):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase and tokenize
    tokens = text.lower().split()
    # Remove stop words
    processed_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(processed_tokens)
```

### 2. VECTORIZATION (vectorization.py)

#### A. Bag-of-Words (BoW)

**Konsep Matematika**:
- Representasi dokumen sebagai vektor frekuensi kata
- Dimensi vektor = jumlah kata unik dalam korpus
- Nilai = frekuensi kemunculan kata dalam dokumen

**Implementasi**:
```python
from sklearn.feature_extraction.text import CountVectorizer

def vectorize_bow(documents):
    vectorizer = CountVectorizer()
    doc_matrix = vectorizer.fit_transform(documents)
    return vectorizer, doc_matrix
```

**Contoh**:
```
Korpus: ["kucing lucu", "anjing lucu", "kucing tidur"]
Vocabulary: ["kucing", "lucu", "anjing", "tidur"]

Vektor:
Dok1: [1, 1, 0, 0]  # "kucing lucu"
Dok2: [0, 1, 1, 0]  # "anjing lucu"
Dok3: [1, 0, 0, 1]  # "kucing tidur"
```

#### B. TF-IDF (Term Frequency-Inverse Document Frequency)

**Formula Matematika**:
```
TF-IDF(t,d) = TF(t,d) × IDF(t)

TF(t,d) = (Jumlah kemunculan term t dalam dokumen d) / (Total term dalam dokumen d)

IDF(t) = log(Total dokumen / Dokumen yang mengandung term t)
```

**Implementasi**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_tfidf(documents):
    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform(documents)
    return vectorizer, doc_matrix
```

**Keunggulan TF-IDF**:
- Mengurangi bobot kata-kata umum (high frequency, low importance)
- Meningkatkan bobot kata-kata unik (low frequency, high importance)
- Lebih akurat dalam menentukan relevansi dokumen

### 3. SEARCH ALGORITHM (vectorization.py)

**Cosine Similarity**:
Mengukur sudut antara dua vektor dalam ruang multidimensi

**Formula**:
```
cosine_similarity(A,B) = (A·B) / (||A|| × ||B||)

Dimana:
- A·B = dot product
- ||A|| = magnitude vektor A
- ||B|| = magnitude vektor B
```

**Implementasi**:
```python
from sklearn.metrics.pairwise import cosine_similarity

def search(query, vectorizer, doc_vectors, df, search_column, top_k=10):
    # Transform query ke vektor
    query_vector = vectorizer.transform([query])
    
    # Hitung similarity dengan semua dokumen
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    
    # Ambil top-k dokumen dengan similarity tertinggi
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for i in top_indices:
        if similarities[i] > 0:  # Hanya dokumen dengan similarity > 0
            results.append({
                'original_index': int(i),
                'title': f"Dokumen {i}",
                'snippet': df.iloc[i][search_column][:200] + "...",
                'score': float(similarities[i])
            })
    
    return results
```

### 4. EVALUATION METRICS (evaluation.py)

#### A. Dengan Ground Truth (Ideal)

**Precision**:
```
Precision = True Positives / (True Positives + False Positives)
         = Dokumen relevan yang ditemukan / Total dokumen yang ditemukan
```

**Recall**:
```
Recall = True Positives / (True Positives + False Negatives)
       = Dokumen relevan yang ditemukan / Total dokumen relevan yang ada
```

**F1-Score**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

#### B. Heuristic Evaluation (Tanpa Ground Truth)

**Implementasi dalam Proyek**:
```python
def calculate_metrics(results, ground_truth_indices=None):
    if ground_truth_indices is None:
        # Heuristic approach
        scores = np.array([r['score'] for r in results])
        
        # Gunakan persentil ke-60 sebagai threshold
        if len(scores) >= 3:
            score_threshold = np.percentile(scores, 60)
        else:
            score_threshold = np.median(scores)
        
        # Hitung pseudo true positives
        pseudo_true_positives = sum(1 for score in scores if score >= score_threshold)
        
        # Fallback: ambil 30% teratas jika tidak ada yang memenuhi threshold
        if pseudo_true_positives == 0 and len(scores) > 0:
            top_30_percent = max(1, int(len(scores) * 0.3))
            sorted_scores = np.sort(scores)[::-1]
            min_relevant_score = sorted_scores[top_30_percent - 1]
            pseudo_true_positives = sum(1 for score in scores if score >= min_relevant_score)
        
        # Hitung precision
        precision = pseudo_true_positives / len(results) if results else 0
        
        # Hitung pseudo-recall berdasarkan distribusi score
        max_score = np.max(scores)
        min_score = np.min(scores)
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
            pseudo_recall = np.mean(normalized_scores)
        else:
            pseudo_recall = 0.5 if len(scores) > 0 else 0.0
        
        # Hitung pseudo F1-score
        if precision + pseudo_recall > 0:
            pseudo_f1 = (2 * precision * pseudo_recall) / (precision + pseudo_recall)
        else:
            pseudo_f1 = 0.0
        
        return {
            'precision': precision,
            'recall': pseudo_recall,
            'f1_score': pseudo_f1
        }
```

### 5. CACHING SYSTEM (caching.py)

**Tujuan**: Optimasi performa dengan menyimpan hasil komputasi yang mahal

**Implementasi**:
```python
cache = {}

# Cache key berdasarkan: filepath + search_column + model_type + language
cache_key = f"{filepath}_{search_column}_{model_type}_{language}"

# Simpan: dataframe, vectorizer, document matrix
if cache_key not in cache:
    cache[cache_key] = {
        'df': processed_dataframe,
        'vectorizer': fitted_vectorizer,
        'doc_matrix': document_vectors
    }
```

### 6. PERFORMANCE CONSIDERATIONS

**Time Complexity**:
- Preprocessing: O(n×m) dimana n=jumlah dokumen, m=rata-rata panjang dokumen
- Vectorization: O(n×v) dimana v=ukuran vocabulary
- Search: O(v) untuk query transformation + O(n) untuk similarity calculation

**Space Complexity**:
- BoW/TF-IDF Matrix: O(n×v) - bisa sangat besar untuk korpus besar
- Caching: Mengurangi rekomputasi tetapi meningkatkan memory usage

**Optimizations**:
1. **Sparse Matrix**: sklearn menggunakan sparse matrix untuk efisiensi memory
2. **Caching**: Menghindari rekomputasi vectorization
3. **Top-k Search**: Hanya mengembalikan k dokumen teratas

=================================================================

## CARA PENGGUNAAN

1. **Instalasi Dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Menjalankan Aplikasi**:
   ```
   streamlit run streamlit_app.py
   ```

3. **Upload Dataset**: Format CSV atau JSON dengan kolom teks

4. **Konfigurasi**:
   - Pilih kolom untuk pencarian
   - Pilih model (TF-IDF atau BoW)
   - Atur bahasa preprocessing

5. **Pencarian**: Masukkan query dan lihat hasil beserta metrik evaluasi

=================================================================

## STRUKTUR FILE

```
IR/
├── streamlit_app.py      # Main UI application
├── ir_system.py          # Core system controller
├── requirements.txt      # Python dependencies
├── test_system.py        # Automated testing
├── README.txt           # This documentation
├── ir_logic/
│   ├── __init__.py
│   ├── preprocessing.py  # Text preprocessing functions
│   ├── vectorization.py  # BoW & TF-IDF implementation
│   ├── evaluation.py     # Metrics calculation
│   └── caching.py       # Performance caching
└── uploads/             # Directory for uploaded datasets
    ├── gojek.csv
    └── politik_merge.csv
```

=================================================================

Dibuat dengan ❤️ untuk memudahkan pencarian informasi dalam dokumen teks.
Untuk pertanyaan teknis, silakan periksa kode sumber atau jalankan test_system.py
untuk memverifikasi fungsionalitas sistem.