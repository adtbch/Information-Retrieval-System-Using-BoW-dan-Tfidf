import numpy as np

def calculate_metrics(results, ground_truth_indices=None):
    """
    Menghitung presisi, recall, dan F1-score.

    Parameter:
    - results (list of dicts): Hasil pencarian, di mana setiap dict berisi setidaknya 'score'.
    - ground_truth_indices (list of int, opsional): Daftar indeks yang dianggap relevan.
      Jika tidak disediakan, heuristik sederhana digunakan untuk demonstrasi.

    Mengembalikan:
    - dict: Kamus yang berisi presisi, recall, dan F1-score.
    """
    if not results:
        return {'precision': 0, 'recall': 0, 'f1_score': 0}

    retrieved_indices = [r['original_index'] for r in results]

    if ground_truth_indices is not None:
        # Perhitungan presisi, recall, F1 standar dengan ground truth
        true_positives = len(set(retrieved_indices) & set(ground_truth_indices))
        
        precision = true_positives / len(retrieved_indices) if retrieved_indices else 0
        recall = true_positives / len(ground_truth_indices) if ground_truth_indices else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    else:
        # Evaluasi berbasis heuristik ketika tidak ada ground truth yang tersedia
        # Menggunakan pendekatan yang lebih kuat untuk menentukan relevansi
        scores = np.array([r['score'] for r in results])
        if len(scores) == 0:
            return {'precision': 0, 'recall': 0, 'f1_score': 0}
        
        # Gunakan beberapa pendekatan untuk menentukan ambang batas relevansi
        if len(scores) >= 3:
            # Gunakan persentil ke-60 sebagai ambang batas untuk distribusi yang lebih baik
            score_threshold = np.percentile(scores, 60)
        else:
            # Untuk hasil yang sangat sedikit, gunakan median
            score_threshold = np.median(scores)
        
        # Pastikan setidaknya beberapa dokumen dianggap relevan
        # Jika tidak ada dokumen yang melebihi ambang batas, anggap 30% teratas sebagai relevan
        pseudo_true_positives = sum(1 for score in scores if score >= score_threshold)
        if pseudo_true_positives == 0 and len(scores) > 0:
            # Ambil 30% dokumen teratas sebagai relevan
            top_30_percent = max(1, int(len(scores) * 0.3))
            sorted_scores = np.sort(scores)[::-1]  # Urutkan menurun
            min_relevant_score = sorted_scores[top_30_percent - 1]
            pseudo_true_positives = sum(1 for score in scores if score >= min_relevant_score)
        
        # Presisi adalah rasio dokumen "relevan" ini terhadap semua dokumen yang diambil
        precision = pseudo_true_positives / len(retrieved_indices) if retrieved_indices else 0
        
        # Untuk tujuan demonstrasi, hitung pseudo-recall berdasarkan distribusi skor
        # Ini mengasumsikan bahwa dokumen dengan skor lebih tinggi lebih mungkin relevan
        max_score = np.max(scores)
        min_score = np.min(scores)
        if max_score > min_score:
            # Normalisasi skor dan gunakan rata-rata sebagai indikator pseudo-recall
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
            'recall': pseudo_recall,  # Pseudo-recall berdasarkan distribusi skor
            'f1_score': pseudo_f1     # Pseudo F1-score
        }

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }