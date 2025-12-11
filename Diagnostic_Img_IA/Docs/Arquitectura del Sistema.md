**ARQUITECTURA DEL SISTEMA**

**Proyecto:** **DIAGNÓSTICO AVANZADO DE IMÁGENES APLICANDO TÉCNICAS DE       INTELIGENCIA ARTIFICIAL**

**1. Tipo de modelo seleccionado y justificación**

El sistema implementa una **arquitectura híbrida**, combinando:

1. **Deep Learning (CNN con Transfer Learning)**
1. **Machine Learning clásico (Random Forest sobre embeddings)**

**Justificación de la selección**

- Las **redes neuronales convolucionales (CNN)** son el estado del arte para el análisis de imágenes médicas, ya que aprenden automáticamente patrones espaciales y texturas relevantes.
- Se utilizó **EfficientNetB0**, un modelo preentrenado en ImageNet, por su excelente relación entre **precisión y costo computacional**.
- El **Random Forest** se emplea como clasificador complementario, aprovechando **embeddings profundos** extraídos por la CNN, lo que mejora la robustez del sistema.
- La combinación de ambos modelos permite:
  - Comparar enfoques (deep vs clásico).
  - Reducir dependencia de un solo modelo.
  - Calcular una **probabilidad combinada** más estable para el diagnóstico final.

**2. Arquitectura detallada del modelo**

**2.1 Arquitectura CNN (Transfer Learning – EfficientNetB0)**

**Entrada:**

- Imagen RGB de tamaño **224×224×3**

**Bloques principales:**

1. **Preprocesamiento**
   1. Normalización mediante efficientnet.preprocess\_input
1. **Backbone preentrenado**
   1. EfficientNetB0 (include\_top=False)
   1. Pesos: ImageNet
   1. Inicialmente congelado (base frozen)
1. **Capas personalizadas**
   1. GlobalAveragePooling2D
   1. Dense (128 neuronas, activación ReLU)
   1. Dropout (0.3 – 0.4)
   1. Dense final (2 neuronas, activación Softmax)

**Función de pérdida:**

- categorical\_crossentropy

**Optimizador:**

- Adam (learning rate inicial: 1e-4, reducido a 5e-5 en fine-tuning)

**Métrica principal:**

- Accuracy

**2.2 Fine-tuning del modelo CNN**

- Se desbloquean las **últimas ~20 capas** del backbone EfficientNet.
- Se reentrena el modelo con un learning rate reducido.
- Se utilizan callbacks:
  - **EarlyStopping**
  - **ReduceLROnPlateau**

Esto permite especializar el modelo en el dominio médico sin perder el conocimiento general aprendido.

**2.3 Arquitectura Random Forest sobre embeddings**

**Entrada:**

- Embeddings extraídos de EfficientNetB0:
- GlobalAveragePooling → vector 1D

**Modelo:**

- RandomForestClassifier (Scikit-learn)

**Parámetros base:**

- n\_estimators: 100
- random\_state: 42

**Optimización:**

- GridSearchCV
  - n\_estimators: [100, 200]
  - max\_depth: [None, 10, 20]
  - Validación cruzada (cv=3)

Este modelo permite interpretar los embeddings profundos con un clasificador no neuronal, complementando la CNN.






**3. Diagrama de flujo del sistema completo**

![Interfaz de usuario gráfica, Texto, Aplicación&#x0A;&#x0A;El contenido generado por IA puede ser incorrecto.](Aspose.Words.f14b2b75-ca6f-4a13-b626-78dc5d0c17f5.001.png)

**4. Pipeline de datos (input → output)**

**Paso 1: Entrada de datos**

- Imagen ecográfica cargada desde:
- Dataset (benign/, malignant/)
- Imagen subida por el usuario (diagnóstico individual).

**Paso 2: Preprocesamiento**

- Conversión a RGB si aplica.
- Redimensionamiento a 224×224.
- Normalización según EfficientNet.

**Paso 3: Análisis exploratorio (EDA)**

- Extracción de características:
  - Intensidad, contraste, entropía, textura, bordes.
- Visualizaciones estadísticas y correlacionales.

**Paso 4: Inferencia de modelos**

- CNN fine-tuned → probabilidad por clase.
- Random Forest → probabilidad por clase.



**Paso 5: Fusión de resultados**

- Promedio simple de probabilidad de malignidad.
- Obtención del diagnóstico final asistido.

**Paso 6: Salida**

- Predicción en consola.
- Visualización en notebook.
- Generación de informe profesional **PDF A4 horizontal** con:
  - Imagen analizada.
  - Resultados cuantitativos.
  - Recomendación clínica.
  - Disclaimer ético.

**5. Tecnologías y librerías utilizadas**

**Lenguaje y entorno**

- **Python 3.x**
- **Google Colab**

**Deep Learning**

- TensorFlow 2.x
- Keras (integrado)
- EfficientNetB0 (transfer learning)

**Procesamiento de imágenes**

- OpenCV (opencv-python-headless)
- Scikit-image
- PIL

**Machine Learning**

- Scikit-learn
  - RandomForestClassifier
  - GridSearchCV

**Análisis y visualización**

- NumPy
- Pandas
- Matplotlib
- Seaborn

**Generación de reportes**

- ReportLab (PDF A4)
- IPython Display

**Gestión de datos**

- Google Drive API (Colab)


**Conclusión arquitectónica**

La arquitectura propuesta combina **robustez, eficiencia y escalabilidad**, integrando modelos profundos con clasificadores clásicos, análisis exploratorio exhaustivo y una salida profesional orientada al usuario final. 

**Esta estructura facilita futuras extensiones del sistema hacia otros estudios médicos o entornos clínicos.**


