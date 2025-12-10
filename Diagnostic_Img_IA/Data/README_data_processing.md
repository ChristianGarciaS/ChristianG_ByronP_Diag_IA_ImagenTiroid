# README – Descripción global de los datos y procesamiento

Este repositorio contiene dos componentes principales:
1) Un archivo comprimido con los **datos originales** (imágenes ecográficas de tiroides), y  
2) Un notebook en formato Jupyter (`.ipynb`) que documenta la **etapa de procesamiento de imágenes** utilizada para preparar el dataset.

---

## 1. Raw Data – Dataset de imágenes ecográficas (`p_1_image.zip`)

### Contenido
El archivo ZIP contiene el dataset original tal como fue publicado, incluyendo:

- Imágenes ecográficas de tiroides  
- Dos categorías:
  - **Benigno**
  - **Maligno**

### Características importantes

- Imágenes provenientes de estudios ecográficos reales.
- Etiquetas diagnósticas ya organizadas en carpetas por clase.
- No se modificó ni alteró el dataset original.

### Uso esperado

- Entrenamiento de modelos de clasificación.
- Evaluación de técnicas de preprocesamiento.
- Análisis estadístico de características visuales.

---

## 2. Procesamiento de Imágenes – Notebook (`Etapa_de_Procesamiento_Imagenes.ipynb`)

El notebook documenta la **ETAPA 2: Procesamiento de Imágenes**, donde se preparan los datos antes del modelado.

### Contenido del notebook

---

### 2.1. Montaje y entorno

- Montaje de Google Drive.
- Instalación de dependencias:
  - TensorFlow
  - scikit-image
  - OpenCV
  - imutils
  - ReportLab
  - NumPy, Pandas, Matplotlib y Seaborn

---

### 2.2. Parámetros globales

- Tamaño objetivo de las imágenes: **224×224 px**
- Batch size
- Clases del problema: `benign` y `malignant`

---

### 2.3. Funciones implementadas

1. **Carga de imágenes (`load_img`)**
2. **Extracción de características (`compute_image_features`)**
   - Estadísticas de intensidad
   - Textura
   - Gradientes y bordes (Sobel, Gaussian)
   - Histogramas
   - Kurtosis y skewness
3. **Visualización respetando proporción (`draw_image_keep_aspect`)**
4. Conversión a tensores y normalización

---

### 2.4. Exploración del dataset (EDA)

Incluye:

- Visualización de imágenes por clase
- Histogramas de intensidad
- Estadísticas por categoría
- Matrices de correlación de características
- Identificación de patrones

---

### 2.5. Creación de conjuntos de datos

Mediante `image_dataset_from_directory` se generan:

- **train_ds** (entrenamiento)
- **val_ds** (validación)

Incluye:

- Shuffling  
- Cache + Prefetch  
- Normalización  
- Cálculo de **class weights**

---

## 3. Relación entre ambos archivos

| Archivo | Propósito |
|--------|-----------|
| `p_1_image.zip` | Dataset en estado original. Base del proyecto. |
| `Etapa_de_Procesamiento_Imagenes.ipynb` | Documento que registra el flujo de preprocesamiento y preparación de datos. |


