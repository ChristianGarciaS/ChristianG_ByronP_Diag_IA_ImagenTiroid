**PLANIFICACIÓN**

**1. Definición del problema y objetivos**

**Definición del problema**

El análisis de imágenes ecográficas de tiroides para la detección de nódulos benignos y malignos depende en gran medida de la experiencia del especialista, lo cual introduce variabilidad subjetiva, riesgos de error diagnóstico y tiempos prolongados de evaluación. Además, el creciente volumen de estudios médicos genera una carga adicional sobre el personal de salud, especialmente en contextos con recursos limitados.

El problema que aborda este proyecto es la **necesidad de un sistema automatizado de apoyo al diagnóstico**, capaz de analizar imágenes tiroideas mediante técnicas de Inteligencia Artificial y proporcionar una estimación objetiva de malignidad, basada en patrones aprendidos automáticamente a partir de datos históricos.

**Objetivo general**

Desarrollar un sistema de **diagnóstico asistido por Inteligencia Artificial** que analice imágenes tiroideas y clasifique los casos como **benignos o malignos**, empleando técnicas de visión por computador, aprendizaje profundo y modelos de clasificación supervisada.

**Objetivos específicos**

- Analizar exploratoriamente un conjunto de imágenes tiroideas etiquetadas (EDA).
- Extraer características visuales relevantes mediante técnicas clásicas y embeddings profundos.
- Entrenar un modelo de Deep Learning basado en **transfer learning (EfficientNetB0)**.
- Complementar el modelo profundo con un clasificador **Random Forest** sobre embeddings.
- Evaluar y comparar el desempeño de los modelos antes y después de la optimización.
- Generar un informe diagnóstico automatizado en formato profesional (PDF A4).


**2. Justificación de la relevancia del proyecto**

Este proyecto es relevante porque integra **Inteligencia Artificial aplicada al sector salud**, específicamente al diagnóstico por imágenes, un área de alto impacto social y científico. El uso de modelos de aprendizaje profundo permite identificar patrones complejos que no siempre son evidentes para el ojo humano, aumentando la consistencia y trazabilidad del análisis.

Desde el punto de vista académico, el proyecto combina:

- Procesamiento digital de imágenes médicas.
- Análisis exploratorio de datos (EDA) con métricas estadísticas y visuales.
- Modelos de Deep Learning con transferencia de aprendizaje.
- Modelos clásicos de machine learning sobre representaciones profundas.
- Generación automática de informes clínicos.

Además, el sistema propuesto actúa como **herramienta de apoyo**, no como sustituto del especialista, alineándose con las buenas prácticas éticas en el uso de IA en medicina.

**3. Alcance del proyecto**

**Incluye**

- Uso de un dataset estructurado de imágenes tiroideas clasificadas en **benignas** y **malignas**.
- Preprocesamiento de imágenes (redimensionamiento, normalización, conversión de canales).
- Análisis exploratorio de datos (EDA), incluyendo:
  - Distribución de clases.
  - Estadísticas descriptivas.
  - Análisis de contraste e intensidad.
  - Matriz de correlación de características.
- Entrenamiento de:
  - Un modelo CNN con **EfficientNetB0** (transfer learning + fine-tuning).
  - Un clasificador **Random Forest** sobre embeddings profundos.
- Evaluación mediante métricas cuantitativas (accuracy, matriz de confusión).
- Generación automática de informes diagnósticos en PDF.

**No incluye**

- Diagnóstico médico definitivo ni validación clínica oficial.
- Uso del sistema en entornos hospitalarios productivos.
- Estudios prospectivos con pacientes reales.
- Certificación sanitaria del modelo.
- Interpretación histopatológica o reemplazo del criterio médico.

**4. Cronograma de desarrollo (planificado vs real)**

|**Fase**|**Actividad**|**Planificado**|**Real**|
| - | - | - | - |
|1|Definición del problema y revisión de datos|Semana 1|Semana 1|
|2|Carga y análisis exploratorio de imágenes (EDA)|Semana 2|Semana 2|
|3|Extracción de características y visualización|Semana 3|Semana 3|
|4|Entrenamiento de modelo CNN base|Semana 4|Semana 4|
|5|Optimización (fine-tuning + RF)|Semana 5|Semana 5|
|6|Evaluación y comparación de modelos|Semana 6|Semana 6|
|7|Generación de informes y documentación|Semana 7|Semana 7|

Las desviaciones entre lo planificado y lo real fueron mínimas y estuvieron asociadas principalmente al ajuste de hiperparámetros y validación de resultados.

**5. Recursos necesarios**

**Datos**

- Dataset de imágenes tiroideas etiquetadas:
  - Carpeta benign/
  - Carpeta malignant/
- Imágenes en formatos .png, .jpg o .jpeg

**Hardware**

- Computador personal con:
  - CPU multinúcleo.
  - Memoria RAM ≥ 8 GB.
- GPU disponible (Google Colab) para acelerar el entrenamiento del modelo CNN.

**Software**

- Google Colab
- Python 3.x
- Librerías principales:
  - TensorFlow / Keras
  - NumPy, Pandas
  - Scikit-learn
  - OpenCV
  - Scikit-image
  - Matplotlib, Seaborn
  - ReportLab (generación de PDF)


**6. Riesgos identificados y mitigación**

|**Riesgo**|**Impacto**|**Estrategia de mitigación**|
| :-: | :-: | :-: |
|Dataset desbalanceado|Sesgo del modelo|Uso de pesos de clase (class\_weight)|
|Overfitting del CNN|Poca generalización|Early Stopping y Fine-tuning controlado|
|Variabilidad en calidad de imágenes|Errores de clasificación|Preprocesamiento y normalización|
|Limitaciones de hardware|Entrenamientos lentos|Uso de GPU en Colab|
|Mala interpretación de resultados|Conclusiones incorrectas|Análisis comparativo y métricas claras|
|Uso indebido del modelo|Riesgo ético|Inclusión de disclaimer clínico|


