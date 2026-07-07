import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import os

# Pengaturan Konfigurasi Halaman Utama
st.set_page_config(
    page_title="Repository Smart Mapping",
    page_icon="🌐",
    layout="wide"
)

PATH_DATASET = "156202.CSV"

@st.cache_data
def load_local_database():
    """Memuat database secara pintar menggunakan cache Streamlit agar hemat RAM"""
    if not os.path.exists(PATH_DATASET):
        return None, f"Error: File '{PATH_DATASET}' tidak ditemukan di repositori GitHub Anda."
    
    try:
        # Menggunakan encoding standar untuk ekspor ProQuest/Scopus
        df = pd.read_csv(PATH_DATASET, encoding='utf-8', errors='replace')
        df.columns = [c.strip().lower() for c in df.columns]
        return df, "Sukses"
    except Exception as e:
        try:
            df = pd.read_csv(PATH_DATASET, encoding='latin1')
            df.columns = [c.strip().lower() for c in df.columns]
            return df, "Sukses"
        except Exception as e_fallback:
            return None, f"Gagal membaca file CSV: {str(e_fallback)}"

# Tampilan Header Aplikasi
st.title("🌐 Repository Smart Mapping (Scopus AI-Style)")
st.markdown("Jelajahi jaringan hubungan antar-artikel ilmiah di dalam pangkalan data koleksi perpustakaan secara visual.")

# Load data di latar belakang
df_global, status_load = load_local_database()

if df_global is None:
    st.error(status_load)
else:
    # Membagi layout menjadi 2 kolom (Kiri untuk Input, Kanan untuk Output)
    kolom_kiri, kolom_kanan = st.columns([1, 2], gap="large")
    
    with kolom_kiri:
        st.subheader("🔍 Pusat Pencarian Literatur")
        kata_kunci = st.text_input(
            "Masukkan Kata Kunci / Frasa:",
            placeholder="Contoh: psychological capital, knowledge transfer..."
        )
        
        st.markdown("**Contoh Kata Kunci Sampel Berdasarkan Dataset:**")
        # Contoh tombol cepat untuk memudahkan pengujian
        sample_queries = ["psychological capital", "knowledge transfer", "belonging", "indigenous"]
        for q in sample_queries:
            if st.button(f"🔑 {q}", key=q):
                kata_kunci = q
                st.rerun()

    with kolom_kanan:
        if kata_kunci.strip():
            df = df_global.copy()
            
            # Identifikasi nama kolom otomatis dari berkas Anda
            kolom_judul = 'title' if 'title' in df.columns else None
            kolom_abstrak = 'abstract' if 'abstract' in df.columns else None
            kolom_penulis = 'author' if 'author' in df.columns else None
            kolom_tahun = 'pubdate' if 'pubdate' in df.columns else ('alphadate' if 'alphadate' in df.columns else 'tahun')
            
            if not kolom_judul:
                st.error("Error Struktur: Kolom 'Title' tidak ditemukan dalam berkas CSV.")
            else:
                # Menggabungkan Judul dan Abstrak untuk pencarian kontekstual semantik
                df[kolom_judul] = df[kolom_judul].fillna('')
                if kolom_abstrak:
                    df[kolom_abstrak] = df[kolom_abstrak].fillna('')
                    df['fitur_teks'] = df[kolom_judul] + " " + df[kolom_abstrak]
                else:
                    df['fitur_teks'] = df[kolom_judul]

                # Perhitungan Vektor Kedekatan Teks (TF-IDF & Cosine Similarity)
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(df['fitur_teks'].astype(str))
                query_vec = vectorizer.transform([kata_kunci])
                
                skor = cosine_similarity(query_vec, tfidf_matrix).flatten()
                df['skor_relevansi'] = skor
                
                # Mengambil Top 7 artikel paling relevan
                df_hasil = df[df['skor_relevansi'] > 0].sort_values(by='skor_relevansi', ascending=False).head(7)
                
                if df_hasil.empty:
                    st.warning(f"❌ Tidak ditemukan artikel yang cocok dengan kata kunci: '{kata_kunci}'. Coba kata kunci lainnya.")
                else:
                    st.subheader("📊 Klaster Visual & Daftar Literatur")
                    
                    # --- TAB DATA UTAMA ---
                    tab_grafik, tab_daftar = st.tabs(["🌐 Network Graph Visual", "📚 Daftar Artikel"])
                    
                    with tab_grafik:
                        # Membuat Grafik Jaringan Menggunakan NetworkX Python murni
                        G = nx.Graph()
                        node_pusat = f"Topik:\n'{kata_kunci}'"
                        G.add_node(node_pusat, type='query')
                        
                        for idx, row in df_hasil.iterrows():
                            judul_bersih = row[kolom_judul].strip()
                            judul_pendek = judul_bersih[:25] + '...' if len(judul_bersih) > 25 else judul_bersih
                            G.add_node(judul_pendek, type='artikel')
                            G.add_edge(node_pusat, judul_pendek, weight=float(row['skor_relevansi']) * 6)
                            
                        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
                        pos = nx.spring_layout(G, k=0.55, seed=42)
                        
                        color_map = []
                        for node in G:
                            if G.nodes[node].get('type') == 'query':
                                color_map.append('#E74C3C') # Merah untuk poros pencarian
                            else:
                                color_map.append('#2980B9') # Biru untuk node artikel ilmiah
                                
                        weights = [G[u][v]['weight'] for u, v in G.edges()]
                        nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=1600, alpha=0.9, ax=ax)
                        nx.draw_networkx_edges(G, pos, width=weights, edge_color='#BDC3C7', ax=ax)
                        nx.draw_networkx_labels(G, pos, font_size=7, font_family='sans-serif', font_weight='bold', ax=ax)
                        
                        ax.set_title(f"Scopus AI-Style Literature Mapping Network\nKata Kunci: '{kata_kunci}'", fontsize=10, fontweight='bold', pad=15)
                        plt.axis('off')
                        st.pyplot(fig)
                        
                    with tab_daftar:
                        # Menampilkan list artikel rekomendasi berbentuk markdown bersih
                        for idx, row in df_hasil.iterrows():
                            penulis = row[kolom_penulis] if kolom_penulis in df.columns and str(row[kolom_penulis]) != 'nan' else "Anonim"
                            tahun_raw = str(row[kolom_tahun]) if kolom_tahun in df.columns else ""
                            tahun = tahun_raw.split('-')[0] if '-' in tahun_raw else tahun_raw[:4]
                            
                            st.markdown(f"### 📄 {row[kolom_judul]}")
                            st.markdown(f"**Penulis:** {penulis} | **Tahun:** {tahun} | **Skor Relevansi:** `{float(row['skor_relevansi']):.2f}`")
                            if kolom_abstrak and str(row[kolom_abstrak]) != 'nan':
                                with st.expander("Lihat Abstrak Lengkap"):
                                    st.write(row[kolom_abstrak])
                            st.markdown("---")
        else:
            st.info("💡 Silakan ketik kata kunci di kolom sebelah kiri atau klik tombol pintas sampel yang tersedia untuk memetakan dokumen.")