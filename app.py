import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

# Configuración de la página web
st.set_page_config(page_title="Reconocimiento Facial con PCA", page_icon="👤", layout="centered")

st.title("Reconocimiento Facial y Reconstrucción con PCA")
st.write("Proyecto de Álgebra Lineal: Espacios vectoriales aplicados a rostros.")

# Configuración de resolución
IMG_HEIGHT, IMG_WIDHT = 128, 128
channels = 3

# Sección de subida de archivos en la web
st.sidebar.header("1. Cargar Dataset")
uploaded_mias = st.sidebar.file_uploader("Sube tus fotos (Tú)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
uploaded_otras = st.sidebar.file_uploader("Sube fotos de otra persona", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_mias and uploaded_otras:
    images_list = []
    y_labels_list = []
    
    # Procesar fotos tuyas (Etiqueta 1)
    for file in uploaded_mias:
        img = Image.open(file).convert('RGB').resize((IMG_WIDHT, IMG_HEIGHT))
        images_list.append(np.array(img, dtype=np.float32) / 255.0)
        y_labels_list.append(1)
        
    # Procesar fotos de otros (Etiqueta 0)
    for file in uploaded_otras:
        img = Image.open(file).convert('RGB').resize((IMG_WIDHT, IMG_HEIGHT))
        images_list.append(np.array(img, dtype=np.float32) / 255.0)
        y_labels_list.append(0)
        
    X_images = np.array(images_list)
    y_labels = np.array(y_labels_list)
    n_samples = X_images.shape[0]
    X = X_images.reshape(n_samples, -1)
    
    # Aplicar PCA y KNN
    max_comp = min(n_samples - 1, 30) if n_samples > 1 else 1
    if max_comp < 1: max_comp = 1
    
    pca_full = PCA(n_components=max_comp, svd_solver='full').fit(X)
    X_transformed = pca_full.transform(X)
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_transformed, y_labels)
    
    st.sidebar.success(f"¡Modelo entrenado con {n_samples} imágenes!")
    
    # --- SECCIÓN 2: EL DESLIZADOR INTERACTIVO ---
    st.header("2. Reconstrucción de Imágenes con PCA")
    st.write("Mueve el deslizador para ver cómo la imagen recupera nitidez al sumar componentes principales.")
    
    indice_foto = st.slider("Selecciona el ID de la foto", 0, n_samples - 1, 0)
    num_componentes = st.slider("Número de Componentes Principales", 1, max_comp, min(5, max_comp))
    
    # Reconstrucción matemática
    reconstruida_plana = np.dot(X_transformed[indice_foto, :num_componentes], 
                                pca_full.components_[:num_componentes, :]) + pca_full.mean_
    reconstruida = reconstruida_plana.reshape((IMG_HEIGHT, IMG_WIDHT, channels))
    reconstruida = np.clip(reconstruida, 0, 1)
    
    # Mostrar imágenes lado a lado
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Original #{indice_foto}")
        st.image(X_images[indice_foto], use_container_width=True)
        
    with col2:
        st.subheader(f"Reconstruida ({num_componentes} comp.)")
        st.image(reconstruida, use_container_width=True)
        
# --- SECCIÓN 3: IDENTIFICADOR DE ROSTROS ---
st.header("3. Identificador de Identidad")
st.write(f"Evaluando la foto actual seleccionada (ID #{indice_foto}):")

# Mostramos un indicador de qué categoría REAL tiene esta foto según lo que subiste
if indice_foto < len(uploaded_mias):
    st.info("📌 Esta foto pertenece originalmente al grupo: **Tú (Dataset propio)**")
else:
    st.warning("📌 Esta foto pertenece originalmente al grupo: **Otra persona**")

if st.button("Verificar Identidad con KNN"):
    vector_prueba = X[indice_foto].reshape(1, -1)
    vector_reducido = pca_full.transform(vector_prueba)
    prediccion = knn.predict(vector_reducido)
    
    if prediccion[0] == 1:
        st.success("Resultado del Modelo: ¡SÍ ERES TÚ! ✅")
    else:
        st.error("Resultado del Modelo: NO ERES TÚ ❌")
