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
    
    for idx, nombre_persona in enumerate(subcarpetas[:3]):
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
    
    knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')
    knn.fit(X_transformed, y_labels)
    
    return pca_full, knn, nombres_clases, X_images.shape[0], y_labels

# Cargar el modelo con las imágenes fijas del repositorio
pca_full, knn, nombres_clases, resultado_carga, y_labels = cargar_dataset_precargado()

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
                
                # Obtener distancias reales de los vecinos más cercanos
                distancias, indices = knn.kneighbors(vector_reducido)
                vecinos_etiquetas = y_labels[indices[0]]
                
                # Distancia mínima al vecino más cercano (menor distancia = mayor parecido real)
                distancia_minima = distancias[0][0]
                
                # Definimos un umbral de distancia máxima tolerada para considerar que un rostro pertenece al dataset.
                # (Puedes ajustar este valor si notas que es muy estricto o muy relajado).
                UMBRAL_DISTANCIA = 2500.0  # Si la distancia excede esto, el rostro es ajeno.
                
                # Calcular probabilidad basada en distancias inversas suavizadas
                eps = 1e-5
                distancias_inv = 1.0 / (distancias[0] + eps)
                clases_unicas = np.unique(y_labels)
                pesos_clases = {c: 0.0 for c in clases_unicas}
                
                for etiqueta, dist_inv in zip(vecinos_etiquetas, distancias_inv):
                    pesos_clases[etiqueta] += dist_inv
                    
                total_peso = sum(pesos_clases.values())
                probabilidades_suaves = {c: (p / total_peso) * 100 for c, p in pesos_clases.items()}
                
                clase_predicha = max(probabilidades_suaves, key=probabilidades_suaves.get)
                nombre_mas_cercano = nombres_clases[clase_predicha]
                
                # Si la distancia supera el umbral, forzamos a que el porcentaje caiga drásticamente
                if distancia_minima > UMBRAL_DISTANCIA:
                    porcentaje_maximo = max(5.0, 100.0 - (distancia_minima / 50.0))
                else:
                    porcentaje_maximo = probabilidades_suaves[clase_predicha]

                st.subheader("Resultados del Análisis Matemático:")
                st.write(f"🔍 **Persona más cercana detectada por PCA/KNN:** {nombre_mas_cercano} ({porcentaje_maximo:.2f}% de similitud)")

                # Buscar si el nombre ingresado coincide con alguna clase registrada
                clase_usuario_id = None
                for k, v in nombres_clases.items():
                    if v.strip().lower() == nombre_usuario.strip().lower():
                        clase_usuario_id = k
                        break
                
                if clase_usuario_id is not None:
                    if distancia_minima > UMBRAL_DISTANCIA:
                        prob_usuario = max(2.0, 40.0 - (distancia_minima / 100.0))
                    else:
                        prob_usuario = probabilidades_suaves[clase_usuario_id]
                        
                    st.info(f"📊 Probabilidad de coincidencia específica para **{nombre_usuario}**: **{prob_usuario:.2f}%**")
                    
                    # Validamos tanto la coincidencia de nombre como que la distancia sea razonable
                    if clase_predicha == clase_usuario_id and prob_usuario >= 60 and distancia_minima <= UMBRAL_DISTANCIA:
                        st.success(f"¡Identidad validada con buen margen! El sistema apunta a que eres **{nombre_usuario}** ✅")
                    else:
                        st.error("❌ **Rechazado:** La foto ingresada no coincide geométricamente con los rasgos de este perfil (similitud muy baja o rostro desconocido).")
                else:
                    st.warning(f"⚠️ El nombre ingresado ('{nombre_usuario}') no se encuentra registrado en el dataset base (Personas válidas: {list(nombres_clases.values())}).")
