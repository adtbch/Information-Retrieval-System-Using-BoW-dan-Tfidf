import pandas as pd
from ir_logic.preprocessing import preprocess_text
from ir_logic.caching import cache
from ir_logic.vectorization import vectorize_bow, vectorize_tfidf, search
from ir_logic.evaluation import calculate_metrics

def search_documents(filepath, query, search_column, model_type, language='auto', use_spacy=True):
    # Buat kunci cache unik untuk dataset, kolom, dan model
    cache_key = f"{filepath}_{search_column}_{model_type}_{language}"

    # Muat dan proses dataset jika tidak ada di cache
    if 'df' not in cache.get(cache_key, {}):
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath, lines=True)
        else:
            raise ValueError("Unsupported file format")

        if search_column not in df.columns:
            raise ValueError(f"Column '{search_column}' not found in the dataset.")

        df['processed_text'] = df[search_column].apply(lambda x: preprocess_text(x, language=language, use_spacy=use_spacy))

        # Inisialisasi cache untuk kunci ini
        cache[cache_key] = {'df': df}
    else:
        df = cache[cache_key]['df']

    # Vektorisasi dokumen atau muat dari cache
    if 'vectorizer' not in cache[cache_key] or 'doc_matrix' not in cache[cache_key]:
        if model_type == 'bow':
            vectorizer, doc_matrix = vectorize_bow(df['processed_text'])
        elif model_type == 'tfidf':
            vectorizer, doc_matrix = vectorize_tfidf(df['processed_text'])
        else:
            raise ValueError("Invalid model type specified")
        cache[cache_key]['vectorizer'] = vectorizer
        cache[cache_key]['doc_matrix'] = doc_matrix
    else:
        vectorizer = cache[cache_key]['vectorizer']
        doc_matrix = cache[cache_key]['doc_matrix']

    # Pra-pemrosesan dan vektorisasi kueri
    processed_query = preprocess_text(query, language=language, use_spacy=use_spacy)
    query_vector = vectorizer.transform([processed_query])

    # Lakukan pencarian
    results = search(processed_query, vectorizer, doc_matrix, df, search_column)

    # Hitung metrik evaluasi menggunakan pendekatan heuristik (tidak ada ground truth yang tersedia)
    # Berikan None sebagai ground_truth_indices untuk menggunakan evaluasi heuristik
    metrics = calculate_metrics(results, ground_truth_indices=None)

    return {
        'results': results,
        'metrics': metrics
    }