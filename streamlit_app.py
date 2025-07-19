import streamlit as st
import pandas as pd
import os
import glob
from ir_system import search_documents

st.set_page_config(page_title="Information Retrieval", layout="wide")

st.title("Sistem Temu Kembali Informasi")
st.markdown("""Sebuah sistem untuk mencari informasi dari dokumen menggunakan model **Bag-of-Words** dan **TF-IDF**.""")

# --- Konfigurasi Sidebar ---
with st.sidebar:
    st.header("Pengaturan")
    use_spacy = st.toggle("Gunakan Pra-pemrosesan Lanjutan (spaCy)", value=True, help="Mengaktifkan tokenisasi dan lemmatisasi yang lebih canggih jika diperlukan.")
    language = st.selectbox(
        "Pilih Bahasa Teks",
        ['id', 'en', 'auto'],
        format_func=lambda x: {'id': 'Indonesia', 'en': 'Inggris', 'auto': 'Deteksi Otomatis'}.get(x)
    )
    
    # Opsi preprocessing tambahan
    st.subheader("Opsi Preprocessing")
    apply_stemming = st.toggle("Terapkan Stemming", value=True, help="Mengubah kata menjadi bentuk dasarnya (contoh: 'running' menjadi 'run')")
    apply_lemmatization = False
    if language == 'en':
        apply_lemmatization = st.toggle("Terapkan Lemmatization", value=False, help="Mengubah kata menjadi bentuk lemma (memerlukan model spaCy lengkap)")
    
    # Opsi untuk menyimpan hasil preprocessing
    st.subheader("Simpan Hasil Preprocessing")
    save_preprocessing = st.toggle("Simpan Hasil Preprocessing", value=False, help="Menyimpan hasil preprocessing ke file dengan kolom terpisah untuk setiap tahap")
    if save_preprocessing:
        output_format = st.selectbox(
            "Format Output",
            ['csv', 'json'],
            format_func=lambda x: x.upper()
        )

# --- Pemilihan Sumber Data ---
st.header("1. Pilih Sumber Data")
uploads_dir = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

available_files = [f for f in os.listdir(uploads_dir) if f.endswith(('.csv', '.json'))]

data_source_option = st.radio("Pilih sumber:", ["Upload File Baru", "Gunakan File yang Ada"], horizontal=True)

filepath = None
if data_source_option == "Upload File Baru":
    uploaded_file = st.file_uploader("Upload file CSV atau JSON", type=['csv', 'json'])
    if uploaded_file:
        filepath = os.path.join(uploads_dir, uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' berhasil di-upload.")
elif data_source_option == "Gunakan File yang Ada":
    if not available_files:
        st.warning("Tidak ada file di direktori 'uploads'. Silakan upload file terlebih dahulu.")
    else:
        selected_file = st.selectbox("Pilih file dari 'uploads'", available_files)
        if selected_file:
            filepath = os.path.join(uploads_dir, selected_file)

# --- Proses Pencarian (hanya jika file sudah dipilih) ---
if filepath:
    try:
        df_peek = pd.read_csv(filepath, nrows=5) if filepath.endswith('.csv') else pd.read_json(filepath, lines=True, nrows=5)
        columns = df_peek.columns.tolist()

        st.header("2. Konfigurasi Pencarian")
        search_col, model_col = st.columns(2)
        with search_col:
            search_column = st.selectbox("Pilih kolom teks untuk pencarian:", columns)
        with model_col:
            model_type = st.selectbox("Pilih Model:", ['tfidf', 'bow'], format_func=lambda x: {'tfidf': 'TF-IDF', 'bow': 'Bag-of-Words'}.get(x))

        query = st.text_input("Masukkan query pencarian:", key="search_query")

        # Tambahkan tombol untuk menyimpan hasil preprocessing
        if 'save_preprocessing' in locals() and save_preprocessing:
            from ir_logic.preprocessing import save_preprocessing_results
            import pandas as pd
            import os
            import datetime
            
            if st.button("Simpan Hasil Preprocessing", type="secondary"):
                try:
                    # Baca dataset
                    if filepath.endswith('.csv'):
                        df = pd.read_csv(filepath)
                    elif filepath.endswith('.json'):
                        df = pd.read_json(filepath, lines=True)
                    else:
                        st.error("Format file tidak didukung.")
                        
                    # Buat direktori untuk menyimpan hasil preprocessing jika belum ada
                    preprocessing_dir = os.path.join(os.getcwd(), 'preprocessing_results')
                    if not os.path.exists(preprocessing_dir):
                        os.makedirs(preprocessing_dir)
                    
                    # Buat nama file output berdasarkan waktu saat ini
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.basename(filepath).split('.')[0]
                    output_filename = f"{filename}_preprocessed_{timestamp}.{output_format}"
                    output_path = os.path.join(preprocessing_dir, output_filename)
                    
                    # Simpan hasil preprocessing
                    with st.spinner("Menyimpan hasil preprocessing..."):
                        saved_path = save_preprocessing_results(
                            df=df,
                            text_column=search_column,
                            output_path=output_path,
                            language=language,
                            use_spacy=use_spacy,
                            apply_stemming=apply_stemming,
                            apply_lemmatization=apply_lemmatization,
                            output_format=output_format
                        )
                    
                    # Tampilkan pesan sukses dan tautan untuk mengunduh file
                    st.success(f"Hasil preprocessing berhasil disimpan ke: {saved_path}")
                    
                    # Baca file yang telah disimpan untuk ditampilkan dan diunduh
                    if output_format == 'csv':
                        preprocessed_df = pd.read_csv(saved_path)
                        st.dataframe(preprocessed_df.head())
                        
                        # Buat tombol unduh
                        with open(saved_path, 'rb') as f:
                            st.download_button(
                                label="Unduh Hasil Preprocessing",
                                data=f,
                                file_name=output_filename,
                                mime="text/csv"
                            )
                    else:  # json
                        import json
                        with open(saved_path, 'r', encoding='utf-8') as f:
                            preprocessed_data = json.load(f)
                        
                        # Tampilkan informasi preprocessing
                        st.subheader("Informasi Preprocessing")
                        st.json(preprocessed_data['preprocessing_info'])
                        
                        # Tampilkan data hasil preprocessing
                        st.subheader("Data Hasil Preprocessing (5 baris pertama)")
                        preprocessed_df = pd.DataFrame(preprocessed_data['data'][:5])
                        st.dataframe(preprocessed_df)
                        
                        # Buat tombol unduh
                        with open(saved_path, 'rb') as f:
                            st.download_button(
                                label="Unduh Hasil Preprocessing",
                                data=f,
                                file_name=output_filename,
                                mime="application/json"
                            )
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menyimpan hasil preprocessing: {e}")
                    st.exception(e)
        
        if st.button("Cari Dokumen", type="primary"):
            if query:
                with st.spinner(f"Mencari '{query}' menggunakan {model_type.upper()}..."):
                    search_output = search_documents(
                        filepath=filepath,
                        query=query,
                        search_column=search_column,
                        model_type=model_type,
                        language=language,
                        use_spacy=use_spacy,
                        apply_stemming=apply_stemming if 'apply_stemming' in locals() else True,
                        apply_lemmatization=apply_lemmatization if 'apply_lemmatization' in locals() else False
                    )
                
                results = search_output['results']
                metrics = search_output['metrics']

                st.header("3. Hasil Pencarian")
                st.success(f"Ditemukan {len(results)} hasil yang relevan.")

                # --- Tampilan Metrik ---
                st.subheader("Metrik Evaluasi (Heuristik)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Presisi", f"{metrics.get('precision', 0):.2f}", help="Dari dokumen yang diambil, berapa persen yang relevan?")
                col2.metric("Recall", f"{metrics.get('recall', 0):.2f}", help="Dari semua dokumen relevan, berapa persen yang berhasil diambil? (Placeholder)")
                col3.metric("F1-Score", f"{metrics.get('f1_score', 0):.2f}", help="Keseimbangan antara Presisi dan Recall. (Placeholder)")
                st.caption("Catatan: Recall dan F1-Score adalah placeholder karena tidak ada ground truth.")

                # --- Tampilan Hasil ---
                st.subheader("Dokumen Ditemukan")
                for result in results:
                    with st.container(border=True):
                        st.markdown(f"**Skor Relevansi:** `{result['score']:.4f}`")
                        # Menampilkan semua kolom dari hasil, kecuali yang tidak perlu
                        display_data = {k: v for k, v in result.items() if k not in ['score', 'original_index', 'processed_text']}
                        st.json(display_data)
            else:
                st.warning("Mohon masukkan query untuk memulai pencarian.")

        # --- Tab untuk melihat detail dataset ---
        with st.expander("Lihat Detail Dataset"):
            st.dataframe(df_peek)
            if st.button("Tampilkan lebih banyak"):
                 df_full = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_json(filepath, lines=True)
                 st.dataframe(df_full)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
        st.exception(e) # Menampilkan traceback untuk debugging