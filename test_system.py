#!/usr/bin/env python3
"""
Script pengujian untuk memverifikasi bahwa semua komponen sistem IR bekerja dengan benar.
"""

import pandas as pd
import os
from ir_system import search_documents

def test_ir_system():
    """
    Menguji sistem IR dengan data sampel untuk memastikan tidak ada error.
    """
    print("=== Pengujian Sistem Information Retrieval ===")
    
    # Buat data sampel untuk pengujian
    sample_data = {
        'title': [
            'Teknologi Artificial Intelligence',
            'Machine Learning dalam Bisnis',
            'Deep Learning dan Neural Networks',
            'Data Science untuk Pemula',
            'Python Programming Tutorial'
        ],
        'content': [
            'Artificial intelligence adalah teknologi yang memungkinkan mesin untuk belajar dan berpikir seperti manusia.',
            'Machine learning membantu bisnis dalam menganalisis data dan membuat prediksi yang akurat.',
            'Deep learning menggunakan neural networks untuk memecahkan masalah kompleks dalam computer vision.',
            'Data science menggabungkan statistik, programming, dan domain knowledge untuk mengekstrak insight.',
            'Python adalah bahasa programming yang populer untuk data science dan machine learning.'
        ]
    }
    
    # Simpan data sampel ke file CSV
    test_file = 'test_data.csv'
    df = pd.DataFrame(sample_data)
    df.to_csv(test_file, index=False)
    print(f"✓ Data sampel disimpan ke {test_file}")
    
    try:
        # Test 1: TF-IDF dengan query sederhana
        print("\n--- Test 1: TF-IDF dengan query 'machine learning' ---")
        result_tfidf = search_documents(
            filepath=test_file,
            query='machine learning',
            search_column='content',
            model_type='tfidf',
            language='en',
            use_spacy=False
        )
        
        print(f"✓ TF-IDF: Ditemukan {len(result_tfidf['results'])} hasil")
        print(f"✓ Metrik TF-IDF: {result_tfidf['metrics']}")
        
        # Verifikasi struktur hasil
        if result_tfidf['results']:
            first_result = result_tfidf['results'][0]
            required_fields = ['original_index', 'title', 'snippet', 'score']
            for field in required_fields:
                if field not in first_result:
                    raise ValueError(f"Field '{field}' tidak ditemukan dalam hasil")
            print(f"✓ Struktur hasil TF-IDF valid: {list(first_result.keys())}")
        
        # Test 2: Bag-of-Words dengan query berbeda
        print("\n--- Test 2: BoW dengan query 'python programming' ---")
        result_bow = search_documents(
            filepath=test_file,
            query='python programming',
            search_column='content',
            model_type='bow',
            language='en',
            use_spacy=False
        )
        
        print(f"✓ BoW: Ditemukan {len(result_bow['results'])} hasil")
        print(f"✓ Metrik BoW: {result_bow['metrics']}")
        
        # Test 3: Query dengan bahasa Indonesia
        print("\n--- Test 3: TF-IDF dengan bahasa Indonesia ---")
        result_id = search_documents(
            filepath=test_file,
            query='teknologi kecerdasan buatan',
            search_column='content',
            model_type='tfidf',
            language='id',
            use_spacy=True
        )
        
        print(f"✓ Bahasa ID: Ditemukan {len(result_id['results'])} hasil")
        print(f"✓ Metrik Bahasa ID: {result_id['metrics']}")
        
        # Test 4: Query yang tidak menghasilkan hasil
        print("\n--- Test 4: Query yang tidak relevan ---")
        result_empty = search_documents(
            filepath=test_file,
            query='quantum computing blockchain cryptocurrency',
            search_column='content',
            model_type='tfidf',
            language='en',
            use_spacy=False
        )
        
        print(f"✓ Query tidak relevan: Ditemukan {len(result_empty['results'])} hasil")
        print(f"✓ Metrik query kosong: {result_empty['metrics']}")
        
        print("\n=== SEMUA TEST BERHASIL! ===")
        print("Sistem IR berfungsi dengan baik dan siap digunakan.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Bersihkan file test
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\n✓ File test {test_file} telah dihapus")
    
    return True

if __name__ == "__main__":
    success = test_ir_system()
    if success:
        print("\n🎉 Sistem siap untuk digunakan!")
    else:
        print("\n⚠️  Masih ada masalah yang perlu diperbaiki.")