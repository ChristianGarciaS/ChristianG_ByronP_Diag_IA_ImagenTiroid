**ANALISIS DE DATOS.**

**Descripción detallada del dataset**

El dataset utilizado en este proyecto está compuesto por imágenes médicas de tiroides, organizadas en una estructura jerárquica de carpetas, donde cada subcarpeta representa una clase diagnóstica:

p\_1\_image/

` `├── benign/

` `└── malignant/

Cada imagen corresponde a un estudio ecográfico tiroideo y se encuentra previamente etiquetada como **benigna** o **maligna**, permitiendo el uso de técnicas de aprendizaje supervisado.

**Características del dataset:**

- Tipo de datos: Imágenes médicas (ecografía).
- Formatos soportados: PNG, JPG, JPEG.
- Dimensiones originales variables.
- Clases:
  - benign: nódulos benignos.
  - malignant: nódulos malignos.
- Canalización: imágenes RGB (convertidas desde escala de grises cuando aplica).

La carga de datos se realiza directamente desde Google Drive, lo que facilita la reproducibilidad y persistencia del experimento.

-----
**2. Estadísticas descriptivas**

Como parte del Análisis Exploratorio de Datos (EDA), se extrajeron un conjunto de **10 características numéricas simples** por imagen, diseñadas para describir propiedades visuales relevantes:

|**Característica**|**Descripción**|
| - | - |
|mean\_r, mean\_g, mean\_b|Intensidad promedio por canal RGB|
|brightness|Brillo medio de la imagen|
|std|Desviación estándar de intensidades|
|entropy|Entropía de Shannon|
|edge\_sum|Suma de bordes (Canny/Sobel)|
|lbp\_var|Varianza del histograma LBP|
|contrast|Contraste global|
|aspect|Relación de aspecto (alto/ancho)|

Se calcularon estadísticas descriptivas (media, desviación, percentiles, mínimos y máximos) para cada característica, permitiendo comprender la distribución y variabilidad de los datos antes del entrenamiento de modelos.

**3. Visualizaciones del EDA**

Durante el análisis exploratorio se generaron múltiples visualizaciones relevantes. Las más importantes incluyen:

**3.1 Ejemplos de imágenes por clase**

Se mostraron imágenes representativas de ambas clases para verificar la calidad visual del dataset y confirmar la correcta asignación de etiquetas.

**Objetivo:** Validación visual del dataset.

**3.2 Distribución de clases**

Gráfico de barras con el número de imágenes en cada clase (benigno vs maligno).

![](/imagenes/img_doc2_1.png)

**Conclusión:**\
Se observa un **desbalance moderado entre clases**, lo cual justificó el uso posterior de pesos de clase durante el entrenamiento.



**3.3 Análisis de contraste por clase**

Diagrama de cajas (boxplot) de la desviación estándar de intensidades por clase.

![](/imagenes/img_doc2_2.png)

**Observación:**\
Las imágenes malignas presentan, en promedio, una mayor variabilidad de intensidad, lo que puede asociarse a mayor complejidad estructural.

**3.4 Histogramas de intensidad por clase**

Histogramas de frecuencia de intensidades de píxeles para imágenes benignas y malignas.

![](/imagenes/img_doc2_3.png)

**Conclusión:**\
Se identifican diferencias sutiles en la distribución de intensidades, lo que sugiere que la información de textura es relevante para la clasificación.

**3.5 Matriz de correlación de características**

Mapa de calor (heatmap) con coeficientes de correlación entre las características extraídas.

![](/imagenes/img_doc2_4.png)

**Hallazgos clave:**

- Alta correlación entre brillo, desviación estándar y contraste.
- Baja correlación entre relación de aspecto y características de textura.
- Potencial redundancia entre algunas características de intensidad.

**4. Identificación de patrones, correlaciones y outliers**

A partir del EDA se identificaron los siguientes patrones:

- Las imágenes malignas tienden a presentar mayor contraste y variabilidad.
- Algunas características presentan correlaciones altas, indicando que no todas aportan información independiente.
- Se identificaron valores extremos en métricas como edge\_sum, asociados a imágenes con ruido o alta presencia de bordes.

Los posibles outliers no fueron eliminados explícitamente, ya que representan variabilidad real del dominio médico.


**5. Decisiones de preprocesamiento justificadas**

Las siguientes decisiones de preprocesamiento fueron tomadas en base al análisis de datos:

- Conversión de imágenes a RGB cuando estaban en escala de grises.
- Redimensionamiento uniforme a **224×224 píxeles** para compatibilidad con EfficientNetB0.
- Normalización implícita mediante preprocess\_input del modelo preentrenado.
- Uso de cache() y prefetch() para optimizar la carga de datos durante el entrenamiento.

Estas decisiones buscaron equilibrar **calidad de información**, **eficiencia computacional** y **compatibilidad con modelos preentrenados**.

**6. Manejo de datos faltantes y desbalanceados**

**Datos faltantes**

- En caso de imágenes que no pudieron ser cargadas correctamente, sus características fueron rellenadas con valores nulos (NaN) durante el EDA para no interrumpir el análisis.
- Estas imágenes no afectaron el entrenamiento del modelo CNN, ya que se trabajó directamente sobre archivos válidos.

**Desbalance de clases**

- Se detectó un desbalance entre las clases benignas y malignas.
- Este problema fue mitigado mediante el cálculo de **pesos de clase** (class\_weight) durante el entrenamiento del modelo CNN.
- Para evaluación, se generó un conjunto de validación balanceado 1:1 para la matriz de confusión.

**Resultado del análisis de datos**

El EDA permitió confirmar que el dataset contiene información relevante para la clasificación, justificar las decisiones de preprocesamiento adoptadas y fundamentar el uso de modelos avanzados de deep learning combinados con clasificadores clásicos.







