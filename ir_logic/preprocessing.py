import re
import nltk
import spacy
import pandas as pd # Tambahkan import pandas di sini
import os # Tambahkan import os di sini
import json # Tambahkan import json di sini

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Dictionary untuk menyimpan model spaCy berdasarkan bahasa
nlp_models = {}

# Coba muat model spaCy untuk bahasa Indonesia
try:
    nlp_models['id'] = spacy.load('id_core_news_sm')
    print("Model spaCy untuk bahasa Indonesia berhasil dimuat.")
except (OSError, ImportError):
    try:
        nlp_models['id'] = spacy.blank('id')
        print("Menggunakan model spaCy blank untuk bahasa Indonesia. Lemmatization mungkin tidak optimal.")
        print("Untuk hasil terbaik, install model bahasa Indonesia dengan: python -m spacy download id_core_news_sm")
    except (OSError, ImportError):
        nlp_models['id'] = None
        print("Model spaCy untuk bahasa Indonesia tidak tersedia. Lemmatization tidak akan berfungsi.")
        print("Untuk mengaktifkan lemmatization, install spaCy dan model bahasa dengan: pip install spacy dan python -m spacy download id_core_news_sm")

# Coba muat model spaCy untuk bahasa Inggris
try:
    nlp_models['en'] = spacy.load('en_core_web_sm')
    print("Model spaCy untuk bahasa Inggris berhasil dimuat.")
except (OSError, ImportError):
    try:
        nlp_models['en'] = spacy.blank('en')
        print("Menggunakan model spaCy blank untuk bahasa Inggris. Lemmatization mungkin tidak optimal.")
        print("Untuk hasil terbaik, install model bahasa Inggris dengan: python -m spacy download en_core_web_sm")
    except (OSError, ImportError):
        nlp_models['en'] = None
        print("Model spaCy untuk bahasa Inggris tidak tersedia. Lemmatization tidak akan berfungsi.")
        print("Untuk mengaktifkan lemmatization, install spaCy dan model bahasa dengan: pip install spacy dan python -m spacy download en_core_web_sm")

# Untuk kompatibilitas dengan kode lama
nlp = nlp_models.get('id')

# Unduh data NLTK yang diperlukan
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Coba mendapatkan stopwords bahasa Indonesia
try:
    id_stop_words = set(stopwords.words('indonesian'))
except LookupError:
    try:
        nltk.download('stopwords')
        id_stop_words = set(stopwords.words('indonesian'))
    except:
        id_stop_words = set([
            'ada', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhir', 
            'akhiri', 'akhirnya', 'aku', 'akulah', 'amat', 'amatlah', 'anda', 'andalah', 'antar', 'antara',
            'antaranya', 'apa', 'apaan', 'apabila', 'apakah', 'apalagi', 'apatah', 'artinya', 'asal', 'asalkan',
            'atas', 'atau', 'ataukah', 'ataupun', 'awal', 'awalnya', 'bagai', 'bagaikan', 'bagaimana', 'bagaimanakah',
            'bagaimanapun', 'bagi', 'bagian', 'bahkan', 'bahwa', 'bahwasanya', 'baik', 'bakal', 'bakalan', 'balik',
            'banyak', 'bapak', 'baru', 'bawah', 'beberapa', 'begini', 'beginian', 'beginikah', 'beginilah', 'begitu',
            'begitukah', 'begitulah', 'begitupun', 'bekerja', 'belakang', 'belakangan', 'belum', 'belumlah', 'benar',
            'benarkah', 'benarlah', 'berada', 'berakhir', 'berakhirlah', 'berakhirnya', 'berapa', 'berapakah',
            'berapalah', 'berapapun', 'berarti', 'berawal', 'berbagai', 'berdatangan', 'beri', 'berikan', 'berikut',
            'berikutnya', 'berjumlah', 'berkali-kali', 'berkata', 'berkehendak', 'berkeinginan', 'berkenaan',
            'berlainan', 'berlalu', 'berlangsung', 'berlebihan', 'bermacam', 'bermacam-macam', 'bermaksud',
            'bermula', 'bersama', 'bersama-sama', 'bersiap', 'bersiap-siap', 'bertanya', 'bertanya-tanya', 'berturut',
            'berturut-turut', 'bertutur', 'berujar', 'berupa', 'besar', 'betul', 'betulkah', 'biasa', 'biasanya',
            'bila', 'bilakah', 'bisa', 'bisakah', 'boleh', 'bolehkah', 'bolehlah', 'buat', 'bukan', 'bukankah',
            'bukanlah', 'bukannya', 'bulan', 'bung', 'cara', 'caranya', 'cukup', 'cukupkah', 'cukuplah', 'cuma',
            'dahulu', 'dalam', 'dan', 'dapat', 'dari', 'daripada', 'datang', 'dekat', 'demi', 'demikian',
            'demikianlah', 'dengan', 'depan', 'di', 'dia', 'diakhiri', 'diakhirinya', 'dialah', 'diantara',
            'diantaranya', 'diberi', 'diberikan', 'diberikannya', 'dibuat', 'dibuatnya', 'didapat',
            'didatangkan', 'digunakan', 'diibaratkan', 'diibaratkannya', 'diingat', 'diingatkan', 'diinginkan',
            'dijawab', 'dijelaskan', 'dijelaskannya', 'dikarenakan', 'dikatakan', 'dikatakannya', 'dikerjakan',
            'diketahui', 'diketahuinya', 'dikira', 'dilakukan', 'dilalui', 'dilihat', 'dimaksud', 'dimaksudkan',
            'dimaksudkannya', 'dimaksudnya', 'diminta', 'dimintai', 'dimisalkan', 'dimulai', 'dimulailah',
            'dimulainya', 'dimungkinkan', 'dini', 'dipastikan', 'diperbuat', 'diperbuatnya', 'dipergunakan',
            'diperkirakan', 'diperlihatkan', 'diperlukan', 'diperlukannya', 'dipersoalkan', 'dipertanyakan',
            'dipunyai', 'diri', 'dirinya', 'disampaikan', 'disebut', 'disebutkan', 'disebutkannya', 'disini',
            'disinilah', 'ditambahkan', 'ditandaskan', 'ditanya', 'ditanyai', 'ditanyakan', 'ditegaskan',
            'ditujukan', 'ditunjuk', 'ditunjuki', 'ditunjukkan', 'ditunjukkannya', 'ditunjuknya', 'dituturkan',
            'dituturkannya', 'diucapkan', 'diucapkannya', 'diungkapkan', 'dong', 'dua', 'dulu', 'empat', 'enggak',
            'enggaknya', 'entah', 'entahlah', 'guna', 'gunakan', 'hal', 'hampir', 'hanya', 'hanyalah', 'hari',
            'harus', 'haruslah', 'harusnya', 'hendak', 'hendaklah', 'hendaknya', 'hingga', 'ia', 'ialah', 'ibarat',
            'ibaratkan', 'ibaratnya', 'ibu', 'ikut', 'ingat', 'ingat-ingat', 'ingin', 'inginkah', 'inginkan', 'ini',
            'inikah', 'inilah', 'itu', 'itukah', 'itulah', 'jadi', 'jadilah', 'jadinya', 'jangan', 'jangankan',
            'janganlah', 'jauh', 'jawab', 'jawaban', 'jawabnya', 'jelas', 'jelaskan', 'jelaslah', 'jelasnya',
            'jika', 'jikalau', 'juga', 'jumlah', 'jumlahnya', 'justru', 'kala', 'kalau', 'kalaulah', 'kalaupun',
            'kalian', 'kami', 'kamilah', 'kamu', 'kamulah', 'kan', 'kapan', 'kapankah', 'kapanpun', 'karena',
            'karenanya', 'kasus', 'kata', 'katakan', 'katakanlah', 'katanya', 'ke', 'keadaan', 'kebetulan',
            'kecil', 'kedua', 'keduanya', 'keinginan', 'kelamaan', 'kelihatan', 'kelihatannya', 'kelima',
            'keluar', 'kembali', 'kemudian', 'kemungkinan', 'kemungkinannya', 'kenapa', 'kepada', 'kepadanya',
            'kesamaan', 'keseluruhan', 'keseluruhannya', 'keterlaluan', 'ketika', 'khususnya', 'kini', 'kinilah',
            'kira', 'kira-kira', 'kiranya', 'kita', 'kitalah', 'kok', 'kurang', 'lagi', 'lagian', 'lah', 'lain',
            'lainnya', 'lalu', 'lama', 'lamanya', 'lanjut', 'lanjutnya', 'lebih', 'lewat', 'lima', 'luar', 'macam',
            'maka', 'makanya', 'makin', 'malah', 'malahan', 'mampu', 'mampukah', 'mana', 'manakala', 'manalagi',
            'masa', 'masalah', 'masalahnya', 'masih', 'masihkah', 'masing', 'masing-masing', 'mau', 'maupun',
            'melainkan', 'melakukan', 'melalui', 'melihat', 'melihatnya', 'memang', 'memastikan', 'memberi',
            'memberikan', 'membuat', 'memerlukan', 'memihak', 'meminta', 'memintakan', 'memisalkan', 'memperbuat',
            'mempergunakan', 'memperkirakan', 'memperlihatkan', 'mempersiapkan', 'mempersoalkan', 'mempertanyakan',
            'mempunyai', 'memulai', 'memungkinkan', 'menaiki', 'menambahkan', 'menandaskan', 'menanti', 'menanti-nanti',
            'menantikan', 'menanya', 'menanyai', 'menanyakan', 'mendapat', 'mendapatkan', 'mendatang', 'mendatangi',
            'mendatangkan', 'menegaskan', 'mengakhiri', 'mengapa', 'mengatakan', 'mengatakannya', 'mengenai',
            'mengerjakan', 'mengetahui', 'menggunakan', 'menghendaki', 'mengibaratkan', 'mengibaratkannya',
            'mengingat', 'mengingatkan', 'menginginkan', 'mengira', 'mengucapkan', 'mengucapkannya', 'mengungkapkan',
            'menjadi', 'menjawab', 'menjelaskan', 'menuju', 'menunjuk', 'menunjuki', 'menunjukkan', 'menunjuknya',
            'menurut', 'menuturkan', 'menyampaikan', 'menyangkut', 'menyatakan', 'menyebutkan', 'menyeluruh',
            'menyiapkan', 'merasa', 'mereka', 'merekalah', 'merupakan', 'meski', 'meskipun', 'meyakini', 'meyakinkan',
            'minta', 'mirip', 'misal', 'misalkan', 'misalnya', 'mula', 'mulai', 'mulailah', 'mulanya', 'mungkin',
            'mungkinkah', 'nah', 'naik', 'namun', 'nanti', 'nantinya', 'nyaris', 'nyatanya', 'oleh', 'olehnya',
            'pada', 'padahal', 'padanya', 'pak', 'paling', 'panjang', 'pantas', 'para', 'pasti', 'pastilah',
            'penting', 'pentingnya', 'per', 'percuma', 'perlu', 'perlukah', 'perlunya', 'pernah', 'persoalan',
            'pertama', 'pertama-tama', 'pertanyaan', 'pertanyakan', 'pihak', 'pihaknya', 'pukul', 'pula', 'pun',
            'punya', 'rasa', 'rasanya', 'rata', 'rupanya', 'saat', 'saatnya', 'saja', 'sajalah', 'saling', 'sama',
            'sama-sama', 'sambil', 'sampai', 'sampai-sampai', 'sampaikan', 'sana', 'sangat', 'sangatlah', 'satu',
            'saya', 'sayalah', 'se', 'sebab', 'sebabnya', 'sebagai', 'sebagaimana', 'sebagainya', 'sebagian',
            'sebaik', 'sebaik-baiknya', 'sebaiknya', 'sebaliknya', 'sebanyak', 'sebegini', 'sebegitu', 'sebelum',
            'sebelumnya', 'sebenarnya', 'seberapa', 'sebesar', 'sebetulnya', 'sebisanya', 'sebuah', 'sebut',
            'sebutlah', 'sebutnya', 'secara', 'secukupnya', 'sedang', 'sedangkan', 'sedemikian', 'sedikit',
            'sedikitnya', 'seenaknya', 'segala', 'segalanya', 'segera', 'seharusnya', 'sehingga', 'seingat',
            'sejak', 'sejauh', 'sejenak', 'sejumlah', 'sekadar', 'sekadarnya', 'sekali', 'sekali-kali', 'sekalian',
            'sekaligus', 'sekalipun', 'sekarang', 'sekarang', 'sekecil', 'seketika', 'sekiranya', 'sekitar',
            'sekitarnya', 'sekurang-kurangnya', 'sekurangnya', 'sela', 'selain', 'selaku', 'selalu', 'selama',
            'selama-lamanya', 'selamanya', 'selanjutnya', 'seluruh', 'seluruhnya', 'semacam', 'semakin', 'semampu',
            'semampunya', 'semasa', 'semasih', 'semata', 'semata-mata', 'semaunya', 'sementara', 'semisal', 'semisalnya',
            'sempat', 'semua', 'semuanya', 'semula', 'sendiri', 'sendirian', 'sendirinya', 'seolah', 'seolah-olah',
            'seorang', 'sepanjang', 'sepantasnya', 'sepantasnyalah', 'seperlunya', 'seperti', 'sepertinya', 'sepihak',
            'sering', 'seringnya', 'serta', 'serupa', 'sesaat', 'sesama', 'sesampai', 'sesegera', 'sesekali',
            'seseorang', 'sesuatu', 'sesuatunya', 'sesudah', 'sesudahnya', 'setelah', 'setempat', 'setengah',
            'seterusnya', 'setiap', 'setiba', 'setibanya', 'setidak-tidaknya', 'setidaknya', 'setinggi', 'seusai',
            'sewaktu', 'siap', 'siapa', 'siapakah', 'siapapun', 'sini', 'sinilah', 'soal', 'soalnya', 'suatu',
            'sudah', 'sudahkah', 'sudahlah', 'supaya', 'tadi', 'tadinya', 'tahu', 'tahun', 'tak', 'tambah',
            'tambahnya', 'tampak', 'tampaknya', 'tandas', 'tandasnya', 'tanpa', 'tanya', 'tanyakan', 'tanyanya',
            'tapi', 'tegas', 'tegasnya', 'telah', 'tempat', 'tengah', 'tentang', 'tentu', 'tentulah', 'tentunya',
            'tepat', 'terakhir', 'terasa', 'terbanyak', 'terdahulu', 'terdapat', 'terdiri', 'terhadap', 'terhadapnya',
            'teringat', 'teringat-ingat', 'terjadi', 'terjadilah', 'terjadinya', 'terkira', 'terlalu', 'terlebih',
            'terlihat', 'termasuk', 'ternyata', 'tersampaikan', 'tersebut', 'tersebutlah', 'tertentu', 'tertuju',
            'terus', 'terutama', 'tetap', 'tetapi', 'tiap', 'tiba', 'tiba-tiba', 'tidak', 'tidakkah', 'tidaklah',
            'tiga', 'tinggi', 'toh', 'tunjuk', 'turut', 'tutur', 'tuturnya', 'ucap', 'ucapnya', 'ujar', 'ujarnya',
            'umum', 'umumnya', 'ungkap', 'ungkapnya', 'untuk', 'usah', 'usai', 'waduh', 'wah', 'wahai', 'waktu',
            'waktunya', 'walau', 'walaupun', 'wong', 'yaitu', 'yakin', 'yakni', 'yang'
        ])

stemmer = PorterStemmer()
# Gabungkan stopwords bahasa Inggris dan Indonesia
stop_words = set(stopwords.words('english')).union(id_stop_words)

def preprocess_text(text, language='id', use_spacy=True, apply_stemming=True, apply_lemmatization=False, return_details=False):
    """Melakukan pra-pemrosesan teks untuk information retrieval.

    Parameter:
    -----------
    text : str
        Teks yang akan diproses.
    language : str, optional
        Bahasa dari teks. Pilihan: 'en', 'id'. Default adalah 'id'.
    use_spacy : bool, optional
        Apakah menggunakan spaCy untuk pemrosesan. Default adalah True.
    apply_stemming : bool, optional
        Apakah menerapkan stemming pada token. Default adalah True.
    apply_lemmatization : bool, optional
        Apakah menerapkan lemmatization pada token. Default adalah False.
        Catatan: Jika apply_lemmatization=True, apply_stemming akan diabaikan.
    return_details : bool, optional
        Jika True, mengembalikan dictionary dengan detail setiap tahap preprocessing.
        Jika False, hanya mengembalikan teks akhir yang telah diproses.

    Mengembalikan:
    --------
    str atau dict
        Jika return_details=False: Teks yang telah diproses.
        Jika return_details=True: Dictionary dengan detail setiap tahap preprocessing.
    """
    # Define a complete default dictionary for return_details=True
    default_details = {
        "original_text": "",
        "cleaned_text": "",
        "tokenized_text": [],
        "filtered_text": [],
        "stemmed_text": "",
        "stemmed_tokens": [],
        "lemmatized_text": "",
        "lemmatized_tokens": [],
        "final_text": "",
        "preprocessing_info": {
            "language": language,
            "use_spacy": use_spacy,
            "apply_stemming": apply_stemming,
            "apply_lemmatization": apply_lemmatization
        }
    }

    if not isinstance(text, str):
        return "" if not return_details else default_details

    # Simpan teks asli
    original_text = text

    # Hapus tag HTML dan karakter non-alfanumerik
    cleaned_text = re.sub(r'<.*?>', '', text)
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    cleaned_text = cleaned_text.lower()

    # Pilih model spaCy berdasarkan bahasa
    current_nlp = nlp_models.get(language)
    
    # Pilih stopwords berdasarkan bahasa yang dipilih
    current_stop_words = id_stop_words if language == 'id' else set(stopwords.words('english'))

    # Variabel untuk menyimpan token di setiap tahap
    tokenized_tokens = []
    filtered_tokens = []
    stemmed_tokens = []
    lemmatized_tokens = []
    final_tokens = []

    # Gunakan pendekatan tokenisasi NLTK untuk tokenisasi dasar
    tokenized_tokens = word_tokenize(cleaned_text)
    
    # Hapus stop words
    filtered_tokens = [w for w in tokenized_tokens if w not in current_stop_words]
    
    # Simpan versi token yang difilter (tanpa stemming/lemmatization)
    filtered_only_tokens = filtered_tokens.copy()
    
    # Terapkan stemming jika diminta
    if apply_stemming:
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    else:
        stemmed_tokens = filtered_tokens.copy()
    
    # Terapkan lemmatization jika diminta dan spaCy tersedia
    if apply_lemmatization and current_nlp is not None and use_spacy:
        try:
            # Proses teks dengan spaCy
            doc = current_nlp(cleaned_text)
            # Ambil lemma untuk token yang tidak dalam stopwords
            lemmatized_tokens = [token.lemma_ for token in doc if token.text.lower() not in current_stop_words]
            
            # Jika tidak ada token yang diproses atau model blank tidak mendukung lemmatization
            if not lemmatized_tokens or all(token == '' for token in lemmatized_tokens):
                raise ValueError("Lemmatization tidak menghasilkan output yang valid")
        except Exception as e:
            print(f"Peringatan: Lemmatization gagal ({str(e)}). Menggunakan metode alternatif.")
            # Fallback ke filtered tokens
            lemmatized_tokens = filtered_tokens.copy()
    else:
        lemmatized_tokens = filtered_tokens.copy()
    
    # Tentukan token final berdasarkan prioritas
    if apply_lemmatization and lemmatized_tokens != filtered_tokens:
        # Jika lemmatization berhasil, gunakan hasil lemmatization
        final_tokens = lemmatized_tokens
    elif apply_stemming:
        # Jika lemmatization tidak diminta atau gagal, dan stemming diminta, gunakan hasil stemming
        final_tokens = stemmed_tokens
    else:
        # Jika tidak ada yang diminta atau keduanya gagal, gunakan token yang difilter
        final_tokens = filtered_tokens.copy()

    # Gabungkan token menjadi teks akhir
    final_text = ' '.join(final_tokens)

    # Gabungkan token menjadi teks untuk setiap tahap
    stemmed_text = ' '.join(stemmed_tokens)
    lemmatized_text = ' '.join(lemmatized_tokens)
    filtered_only_text = ' '.join(filtered_only_tokens)
    
    # Kembalikan hasil sesuai dengan parameter return_details
    if return_details:
        return {
            "original_text": original_text,
            "cleaned_text": cleaned_text,
            "tokenized_text": tokenized_tokens,
            "filtered_text": filtered_only_tokens,
            "stemmed_text": stemmed_text,
            "stemmed_tokens": stemmed_tokens,
            "lemmatized_text": lemmatized_text,
            "lemmatized_tokens": lemmatized_tokens,
            "final_text": final_text,
            "preprocessing_info": {
                "language": language,
                "use_spacy": use_spacy,
                "apply_stemming": apply_stemming,
                "apply_lemmatization": apply_lemmatization
            }
        }
    else:
        return final_text


def contoh_penggunaan():
    """Contoh penggunaan fungsi preprocess_text dengan berbagai opsi.
    
    Catatan:
    -------
    Untuk lemmatization yang optimal, pastikan model spaCy yang sesuai telah diinstal:
    - Untuk bahasa Indonesia: python -m spacy download id_core_news_sm
    - Untuk bahasa Inggris: python -m spacy download en_core_web_sm
    
    Jika model tidak tersedia, sistem akan menggunakan model blank yang mungkin tidak
    mendukung lemmatization dengan baik, atau akan kembali ke metode tokenisasi NLTK.
    """
    # Contoh teks dalam bahasa Indonesia
    teks_id = "Saya sedang belajar pemrosesan bahasa alami menggunakan Python. Ini sangat menyenangkan!"
    
    # Contoh teks dalam bahasa Inggris
    teks_en = "I am learning natural language processing using Python. It's very exciting!"
    
    print("\n===== CONTOH PENGGUNAAN PREPROCESS_TEXT =====\n")
    
    # Preprocessing dasar (tanpa stemming atau lemmatization)
    print("[Bahasa Indonesia] Preprocessing dasar:")
    hasil_id_dasar = preprocess_text(teks_id, language='id', apply_stemming=False, apply_lemmatization=False)
    print(f"Input: {teks_id}")
    print(f"Output: {hasil_id_dasar}\n")
    
    # Preprocessing dengan stemming
    print("[Bahasa Indonesia] Preprocessing dengan stemming:")
    hasil_id_stem = preprocess_text(teks_id, language='id', apply_stemming=True, apply_lemmatization=False)
    print(f"Input: {teks_id}")
    print(f"Output: {hasil_id_stem}\n")
    
    # Preprocessing dengan lemmatization
    print("[Bahasa Indonesia] Preprocessing dengan lemmatization:")
    hasil_id_lemma = preprocess_text(teks_id, language='id', apply_stemming=False, apply_lemmatization=True)
    print(f"Input: {teks_id}")
    print(f"Output: {hasil_id_lemma}")
    print("Catatan: Jika output kosong atau tidak optimal, pastikan model spaCy 'id_core_news_sm' telah diinstal.\n")
    
    # Preprocessing dasar untuk bahasa Inggris
    print("[Bahasa Inggris] Preprocessing dasar:")
    hasil_en_dasar = preprocess_text(teks_en, language='en', apply_stemming=False, apply_lemmatization=False)
    print(f"Input: {teks_en}")
    print(f"Output: {hasil_en_dasar}\n")
    
    # Preprocessing dengan stemming untuk bahasa Inggris
    print("[Bahasa Inggris] Preprocessing dengan stemming:")
    hasil_en_stem = preprocess_text(teks_en, language='en', apply_stemming=True, apply_lemmatization=False)
    print(f"Input: {teks_en}")
    print(f"Output: {hasil_en_stem}\n")
    
    # Preprocessing dengan lemmatization untuk bahasa Inggris
    print("[Bahasa Inggris] Preprocessing dengan lemmatization:")
    hasil_en_lemma = preprocess_text(teks_en, language='en', apply_stemming=False, apply_lemmatization=True)
    print(f"Input: {teks_en}")
    print(f"Output: {hasil_en_lemma}")
    print("Catatan: Jika output kosong atau tidak optimal, pastikan model spaCy 'en_core_web_sm' telah diinstal.\n")
    
    # Perbandingan stemming vs lemmatization untuk bahasa Indonesia
    print("[Bahasa Indonesia] Perbandingan stemming vs lemmatization:")
    teks_id_perbandingan = "Saya membaca buku-buku pembelajaran dan mempelajari materi pemrograman"
    hasil_id_stem_perbandingan = preprocess_text(teks_id_perbandingan, language='id', apply_stemming=True, apply_lemmatization=False)
    hasil_id_lemma_perbandingan = preprocess_text(teks_id_perbandingan, language='id', apply_stemming=False, apply_lemmatization=True)
    print(f"Input: {teks_id_perbandingan}")
    print(f"Stemming: {hasil_id_stem_perbandingan}")
    print(f"Lemmatization: {hasil_id_lemma_perbandingan}\n")
    
    # Perbandingan stemming vs lemmatization untuk bahasa Inggris
    print("[Bahasa Inggris] Perbandingan stemming vs lemmatization:")
    teks_en_perbandingan = "I am running and jumping while studying programming languages"
    hasil_en_stem_perbandingan = preprocess_text(teks_en_perbandingan, language='en', apply_stemming=True, apply_lemmatization=False)
    hasil_en_lemma_perbandingan = preprocess_text(teks_en_perbandingan, language='en', apply_stemming=False, apply_lemmatization=True)
    print(f"Input: {teks_en_perbandingan}")
    print(f"Stemming: {hasil_en_stem_perbandingan}")
    print(f"Lemmatization: {hasil_en_lemma_perbandingan}\n")
    
    print("===== SELESAI =====\n")


def save_preprocessing_results(df, text_column, output_path, language='id', use_spacy=True, apply_stemming=True, apply_lemmatization=False, output_format='csv'):
    """
    Menyimpan hasil preprocessing dari DataFrame ke file dengan kolom terpisah untuk setiap tahap preprocessing.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame yang berisi data yang akan diproses.
    text_column : str
        Nama kolom yang berisi teks yang akan diproses.
    output_path : str
        Path untuk menyimpan file hasil preprocessing.
    language : str, optional
        Bahasa dari teks. Pilihan: 'en', 'id'. Default adalah 'id'.
    use_spacy : bool, optional
        Apakah menggunakan spaCy untuk pemrosesan. Default adalah True.
    apply_stemming : bool, optional
        Apakah menerapkan stemming pada token. Default adalah True.
    apply_lemmatization : bool, optional
        Apakah menerapkan lemmatization pada token. Default adalah False.
    output_format : str, optional
        Format output file. Pilihan: 'csv', 'json'. Default adalah 'csv'.
    
    Returns:
    --------
    str
        Path ke file hasil preprocessing yang telah disimpan.
    """
    # Pastikan df adalah DataFrame yang valid dan text_column ada
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' harus berupa pandas.DataFrame.")
    if text_column not in df.columns:
        raise ValueError(f"Kolom '{text_column}' tidak ditemukan dalam DataFrame.")

    # Buat DataFrame baru untuk menyimpan hasil preprocessing
    results_df = df.copy() # Salin DataFrame asli untuk menjaga kolom lain
    
    # Lakukan preprocessing dan tambahkan kolom hasil preprocessing
    preprocessing_results = []
    for text_value in df[text_column]: # Gunakan nama variabel yang berbeda untuk menghindari konflik dengan parameter
        result = preprocess_text(text_value, language=language, use_spacy=use_spacy, 
                                 apply_stemming=apply_stemming, apply_lemmatization=apply_lemmatization,
                                 return_details=True)
        preprocessing_results.append(result)
    
    # Tambahkan kolom hasil preprocessing ke DataFrame baru
    results_df['original_text'] = [result['original_text'] for result in preprocessing_results]
    results_df['cleaned_text'] = [result['cleaned_text'] for result in preprocessing_results]
    results_df['tokenized_text'] = [' '.join(result['tokenized_text']) for result in preprocessing_results]
    results_df['filtered_text'] = [' '.join(result['filtered_text']) for result in preprocessing_results]
    
    # Tambahkan kolom stemming dan lemmatization
    results_df['stemmed_text'] = [result['stemmed_text'] for result in preprocessing_results]
    results_df['lemmatized_text'] = [result['lemmatized_text'] for result in preprocessing_results]
    
    # Tambahkan kolom final text (hasil akhir preprocessing)
    results_df['final_text'] = [result['final_text'] for result in preprocessing_results]
    
    # Tambahkan kolom untuk menunjukkan metode yang digunakan
    results_df['preprocessing_method'] = [
        'Lemmatization' if result['preprocessing_info']['apply_lemmatization'] else
        'Stemming' if result['preprocessing_info']['apply_stemming'] else
        'Filtering Only'
        for result in preprocessing_results
    ]
    
    # Tambahkan informasi preprocessing sebagai metadata
    preprocessing_info = preprocessing_results[0]['preprocessing_info'] if preprocessing_results else {}
    
    # Simpan hasil preprocessing ke file
    if output_format.lower() == 'csv':
        results_df.to_csv(output_path, index=False)
    elif output_format.lower() == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json_data = {
                'preprocessing_info': preprocessing_info,
                'data': results_df.to_dict(orient='records')
            }
            json.dump(json_data, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Format output '{output_format}' tidak didukung. Gunakan 'csv' atau 'json'.")
    
    return output_path

# Jalankan contoh jika file ini dijalankan langsung
if __name__ == "__main__":
    contoh_penggunaan()

    # Contoh penggunaan save_preprocessing_results (membutuhkan DataFrame)
    # Anda bisa membuat DataFrame dummy untuk pengujian:
    dummy_data = {
        'id': [1, 2, 3],
        'text_data': [
            "Ini adalah contoh teks Bahasa Indonesia yang akan diproses.",
            "Another example of English text for natural language processing.",
            None # Contoh data non-string
        ],
        'category': ['A', 'B', 'A']
    }
    dummy_df = pd.DataFrame(dummy_data)
    
    print("\n===== CONTOH PENGGUNAAN SAVE_PREPROCESSING_RESULTS =====\n")

    # Simpan hasil preprocessing ke CSV
    try:
        saved_csv_path = save_preprocessing_results(
            df=dummy_df,
            text_column='text_data',
            output_path='hasil_preprocessing.csv',
            language='id',
            apply_stemming=True,
            apply_lemmatization=False,
            output_format='csv'
        )
        print(f"Hasil preprocessing disimpan di: {saved_csv_path}")
    except Exception as e:
        print(f"Terjadi kesalahan saat menyimpan CSV: {e}")

    # Simpan hasil preprocessing ke JSON dengan lemmatization
    try:
        saved_json_path = save_preprocessing_results(
            df=dummy_df,
            text_column='text_data',
            output_path='hasil_preprocessing.json',
            language='en',
            use_spacy=True,
            apply_stemming=False,
            apply_lemmatization=True,
            output_format='json'
        )
        print(f"Hasil preprocessing disimpan di: {saved_json_path}")
    except Exception as e:
        print(f"Terjadi kesalahan saat menyimpan JSON: {e}")

    print("\n===== SELESAI PENGUJIAN SAVE_PREPROCESSING_RESULTS =====\n")