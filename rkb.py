import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from io import BytesIO
from PIL import Image
import streamlit as st
from sklearn.metrics.pairwise import rbf_kernel

# Fungsi untuk membaca gambar dan mengkonversi ke format RGB
def read_image(image_path):
    image = cv2.imdecode(np.frombuffer(image_path.read(), np.uint8), 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

# Fungsi untuk menghitung frekuensi warna dominan tertentu dengan RBF
def calculate_color_frequencies(image, gamma=0.001):
    pixels = image.reshape(-1, 3)
    color_labels = {
        (255, 0, 0): 'Merah',
        (0, 255, 0): 'Hijau',
        (0, 0, 255): 'Biru',
        (255, 255, 0): 'Kuning'
    }
    colors = np.array(list(color_labels.keys()))
    
    # Hitung kernel RBF
    rbf_values = rbf_kernel(pixels, colors, gamma=gamma)
    
    closest_colors = np.argmax(rbf_values, axis=1)
    color_counts = {tuple(color): 0 for color in colors}
    
    for color_idx in closest_colors:
        color = tuple(colors[color_idx])
        color_counts[color] += 1
    
    return color_counts, color_labels

# Fungsi untuk menggambar lingkaran pada lokasi warna dominan
def draw_dominant_color_circle(image, color_counts, color_labels):
    output_image = image.copy()
    height, width, _ = image.shape
    
    for color, count in color_counts.items():
        if count > 0:
            # Konversi warna dari (R, G, B) ke (B, G, R) untuk OpenCV
            color_bgr = tuple(int(c) for c in color[::-1])
            mask = np.all(image == color, axis=-1)
            locations = np.column_stack(np.where(mask))
            
            # Pilih lokasi acak dari lokasi warna dominan jika jumlahnya lebih dari 10
            if len(locations) > 10:
                locations = locations[np.random.choice(locations.shape[0], 10, replace=False)]
            
            for loc in locations:
                center = (loc[1], loc[0])  # Perhatikan urutan (x, y)
                radius = min(height, width) // 20  # Batasi radius lingkaran
                cv2.circle(output_image, center, radius, color_bgr, thickness=2)
    
    return output_image

# Fungsi untuk membuat time series data
def create_time_series(dates, color_counts, color_labels):
    data = []
    for date in dates:
        for color, count in color_counts.items():
            data.append({
                'Date': date,
                'Color': color_labels[color],
                'Count': count
            })
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Aplikasi Streamlit
st.title("Deteksi Warna Dominan Gambar")

# Memuat gambar dari perangkat pengguna
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = read_image(uploaded_file)
    color_counts, color_labels = calculate_color_frequencies(image)
    
    image_with_circles = draw_dominant_color_circle(image, color_counts, color_labels)

    image_pil = Image.fromarray(image_with_circles)
    buffer = BytesIO()
    image_pil.save(buffer, format="JPEG")
    st.image(buffer.getvalue(), caption='Gambar', use_column_width=True)

    st.write("Frekuensi Warna Dominan:")
    color_data = pd.DataFrame({
        'Warna': [color_labels[color] for color in color_counts.keys()],
        'Frekuensi': [count for count in color_counts.values()]
    })
    
    st.dataframe(color_data)
    
    # Tampilkan diagram batang
    st.bar_chart(color_data.set_index('Warna'))

    # Buat time series data
    dates = [datetime.now()]
    time_series_df = create_time_series(dates, color_counts, color_labels)
    
    st.write("Data Time Series Warna Dominan")
    st.dataframe(time_series_df)

    csv = time_series_df.to_csv().encode('utf-8')
    st.download_button(label="Download CSV", data=csv, file_name='color_time_series.csv', mime='text/csv')
