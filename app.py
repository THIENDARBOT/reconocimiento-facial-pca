import streamlit as st
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

# Configuración de la página web
st.set_page_config(page_title="Reconocimiento Facial con PCA", page_icon="👤", layout="centered")

st.title("Reconocimiento Facial y Probabilidad con PCA")
st.write("Sistema inteligente de reconocimiento basado en espacios vectoriales y álgebra lineal.")

# Configuración de resolución
IMG_HEIGHT, IMG_WIDHT = 256, 256
channels = 3

# Ruta de la carpeta donde están precargadas las 3 personas en tu repositorio de GitHub
CARPETA_DATASET = "personas"

@st.cache_resource
def cargar_dataset_precargado():
    images_list = []
    y_labels_list = []
    nombres_clases = {}
    
    if not os.path.exists(CARPETA_DATASET):
        return None, None, None, f"No se encontró la carpeta '{CARPETA_DATASET}' en el repositorio."
    
    # Leer las subcarpetas (cada subcarpeta es una persona)
    subcarpetas = sorted([d for d in os.listdir(CARPETA_DATASET) if os.path.isdir(os.path.join(CARPETA_DATASET, d))])
    
    if len(subcarpetas) < 3:
        return None, None, None, f"Se necesitan al menos 3 carpetas de personas dentro de '{CARPETA_DATASET}'."
    
    for idx, nombre_persona in enumerate(subcarpetas[:3]):  Tomamos exactamente 3 personas
        nombres_clases[idx] = nombre_persona
        ruta_persona = os.path.join(CARPETA_DATASET, nombre_persona)
        
        for filename in os.listdir(ruta_persona):
            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(ruta_persona, filename)
                try:
                    img = Image.open(img_path).convert('RGB').resize((IMG_WIDHT, IMG_HEIGHT))
                    images_list.append(np.array(img, dtype=np.float32) / 255.0)
                    y_labels_list.append(idx)
                except Exception as e:
                    pass
                    
    if len(images_list) == 0:
        return None, None, None, "No hay imágenes válidas dentro de las subcarpetas."
        
    X_images = np.array(images_list)
    y_labels = np.array(y_labels_list)
    n_samples = X_images.shape[0]
    X = X_images.reshape(n_samples, -1)
    
    # Entrenar PCA y KNN
    max_comp = min(n_samples - 1, 30) if n_samples > 1 else 1
    if max_comp < 1: max_comp = 1
    
    pca_full = PCA(n_components=max_comp, svd_solver='full').fit(X)
    X_transformed = pca_full.transform(X)
    
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn.fit(X_transformed, y_labels)
    
    return pca_full, knn, nombres_clases, X_images.shape[0]

# Cargar el modelo con las imágenes fijas del repositorio
pca_full, knn, nombres_clases, resultado_carga = cargar_dataset_precargado()

if isinstance(resultado_carga, str):
    st.error(f"⚠️ Error de configuración: {resultado_carga}")
    st.info(f"Asegúrate de crear una carpeta llamada '{CARPETA_DATASET}' en tu repositorio de GitHub y dentro pon 3 carpetas con los nombres de las personas y sus fotos.")
else:
    st.success(f"✅ ¡Dataset precargado exitosamente! El sistema reconoce a: **{list(nombres_clases.values())}** ({resultado_carga} imágenes en total).")
    
    # --- INTERFAZ PARA EL USUARIO FINAL ---
    st.header("Verificación de Identidad")
    st.write("Sube una foto tuya e ingresa tu nombre para que el sistema calcule la probabilidad de coincidencia.")

    nombre_usuario = st.text_input("Ingresa tu nombre:")
    foto_usuario = st.file_uploader("Sube tu foto de prueba", type=['png', 'jpg', 'jpeg'])

    if foto_usuario is not None:
        img_usuario = Image.open(foto_usuario).convert('RGB').resize((IMG_WIDHT, IMG_HEIGHT))
        st.image(img_usuario, caption="Foto ingresada por el usuario", width=250)

        if st.button("Analizar y Comparar con PCA"):
            if nombre_usuario.strip() == "":
                st.warning("Por favor, ingresa tu nombre antes de analizar.")
            else:
                # Procesar la foto ingresada
                vector_nuevo = np.array(img_usuario, dtype=np.float32).flatten() / 255.0
                vector_reducido = pca_full.transform([vector_nuevo])
                
                # Obtener probabilidades y predicción
                probabilidades = knn.predict_proba(vector_reducido)[0]
                clase_predicha = knn.predict(vector_reducido)[0]
                
                nombre_mas_cercano = nombres_clases[clase_predicha]
                porcentaje_maximo = probabilidades[clase_predicha] * 100

                st.subheader("Resultados del Análisis Matemático:")
                st.write(f"🔍 **Persona más cercana detectada por PCA/KNN:** {nombre_mas_cercano} ({porcentaje_maximo:.2f}% de similitud global)")

                # Buscar si el nombre ingresado por el usuario coincide con alguna clase registrada
                clase_usuario_id = None
                for k, v in nombres_clases.items():
                    if v.strip().lower() == nombre_usuario.strip().lower():
                        clase_usuario_id = k
                        break
                
                if clase_usuario_id is not None:
                    prob_usuario = probabilidades[clase_usuario_id] * 100
                    st.info(f"📊 Probabilidad de coincidencia específica para **{nombre_usuario}**: **{prob_usuario:.2f}%**")
                    
                    if prob_usuario > 50:
                        st.success(f"¡Identidad confirmada! El sistema valida que eres **{nombre_usuario}** ✅")
                    else:
                        st.warning("Los rasgos difieren considerablemente de este perfil. Revisa el nombre ingresado ⚠️")
                else:
                    st.warning(f"⚠️ El nombre ingresado ('{nombre_usuario}') no se encuentra registrado en el dataset base del sistema (Personas válidas: {list(nombres_clases.values())}).")
