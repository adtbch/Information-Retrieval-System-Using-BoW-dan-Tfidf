import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def vectorize_bow(documents):
    """
    Membuat representasi Bag-of-Words (BoW) dari dokumen.

    Parameters:
    -----------
    documents : list of str
        Daftar dokumen (teks yang sudah diproses).

    Returns:
    --------
    CountVectorizer
        Objek vectorizer yang sudah di-fit.
    scipy.sparse.csr_matrix
        Matriks BoW dari dokumen.
    """
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(documents)
    return vectorizer, bow_matrix

def vectorize_tfidf(documents):
    """
    Membuat representasi TF-IDF dari dokumen.

    Parameters:
    -----------
    documents : list of str
        Daftar dokumen (teks yang sudah diproses).

    Returns:
    --------
    TfidfVectorizer
        Objek vectorizer yang sudah di-fit.
    scipy.sparse.csr_matrix
        Matriks TF-IDF dari dokumen.
    """
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError as e:
        if 'empty vocabulary' in str(e):
            raise ValueError(
                "Gagal membuat matriks TF-IDF: Kosakata kosong. "
                "Ini kemungkinan terjadi karena semua dokumen hanya berisi 'stop words' "
                "atau karakter yang diabaikan setelah pra-pemrosesan. "
                "Coba periksa kembali data Anda atau sesuaikan langkah pra-pemrosesan."
            )
        else:
            raise e
    return vectorizer, tfidf_matrix

def search(query, vectorizer, doc_vectors, df, search_column):
    """
    Mencari dokumen yang relevan dengan query menggunakan cosine similarity.

    Parameters:
    -----------
    query : str
        Query yang sudah diproses.
    vectorizer : CountVectorizer or TfidfVectorizer
        Objek vectorizer yang sudah di-fit.
    doc_vectors : scipy.sparse.csr_matrix
        Matriks vektor dari seluruh dokumen.
    df : pandas.DataFrame
        DataFrame yang berisi data dokumen.
    search_column : str
        Nama kolom di DataFrame yang berisi teks asli untuk ditampilkan.

    Returns:
    --------
    list
        Daftar hasil pencarian.
    """
    # Ubah query menjadi vektor
    query_vector = vectorizer.transform([query])

    # Hitung cosine similarity
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()

    # Dapatkan 10 hasil teratas
    top_indices = similarities.argsort()[-10:][::-1]

    results = []
    for i in top_indices:
        if similarities[i] > 0.01:  # Filter hasil yang tidak relevan
            results.append({
                'original_index': int(i),
                'title': df.iloc[i].get('title', f"Dokumen {i}"),
                'snippet': df.iloc[i][search_column][:150] + '...',
                'score': float(similarities[i])
            })

    return results