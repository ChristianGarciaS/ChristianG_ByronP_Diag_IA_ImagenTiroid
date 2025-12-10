#  Exploratory Data Analysis (EDA)

Este reporte resume el análisis exploratorio realizado sobre el dataset de imágenes de ultrasonido tiroideo.  
El objetivo del EDA es comprender la estructura del dataset, detectar desbalances, caracterizar las imágenes y evaluar la consistencia de las features extraídas antes del entrenamiento de modelos de Machine Learning.

---

##  1. Información general del dataset

Durante el proceso de carga se identificó un total de '637 imágenes', distribuidas en dos clases principales:

- 'benign'
- 'malignant'

Estas imágenes fueron empleadas para:
- Extracción de features para análisis estadístico,
- Entrenamiento de modelos base (CNN y RF),
- Generación de embeddings para el modelo Random Forest,
- Creación de los conjuntos 'train' y 'validation'.

---

##  2. Distribución de clases

La distribución global del dataset es la siguiente:

| label     | count |
|-----------|-------|
| malignant | 377   |
| benign    | 260   |

 **Conclusión:**  
Existe un desbalance notable, con aproximadamente '59%' de imágenes malignas y '41%' benignas.  
Este desbalance afecta directamente el comportamiento del modelo, especialmente métricas como **recall** para la clase minoritaria.

---

##  3. Class Weights generados

Para compensar este desbalance durante el entrenamiento con CNN, se generaron los siguientes pesos:

| Clase | Peso |
|-------|-------|
| benign (0) | 1.2439 |
| malignant (1) | 0.8361 |

Estos pesos se integraron en el entrenamiento del modelo para equilibrar la contribución de cada clase en la función de pérdida.

---

##  4. División de datos: Train vs Validation

El dataset se dividió de la siguiente manera:

- **510 imágenes para entrenamiento**
- **127 imágenes para validación**

Con la siguiente distribución interna:

| split       | benign | malignant |
|-------------|--------|-----------|
| train       | 205    | 305       |
| validation  | 55     | 72        |

 *El desbalance se mantiene en ambos subconjuntos, lo cual debe considerarse al interpretar las métricas.*

---

##  5. Features extraídas para análisis estadístico

Se extrajeron un total de '10 features por imagen', útiles para caracterizar brillo, textura, entropía, bordes e intensidad de color.  

Las estadísticas globales obtenidas fueron:

| Feature    | count | mean | std | min | 25% | 50% | 75% | max |
|------------|-------|------|------|------|------|------|------|------|
| mean_r     | 637 | 67.95 | 12.66 | 35.40 | 59.42 | 67.37 | 75.99 | 105.69 |
| mean_g     | 637 | 67.95 | 12.66 | 35.40 | 59.42 | 67.37 | 75.99 | 105.69 |
| mean_b     | 637 | 67.95 | 12.66 | 35.40 | 59.42 | 67.37 | 75.99 | 105.69 |
| brightness | 637 | 0.266 | 0.0496 | 0.138 | 0.233 | 0.264 | 0.298 | 0.414 |
| std        | 637 | 0.145 | 0.0169 | 0.095 | 0.133 | 0.144 | 0.156 | 0.216 |
| entropy    | 637 | 6.73 | 0.203 | 5.50 | 6.61 | 6.75 | 6.86 | 7.37 |
| edge_sum   | 637 | 1,658,677 | 412,007 | 468,690 | 1,362,210 | 1,639,395 | 1,914,285 | 3,012,315 |
| lbp_var    | 637 | 0.00229 | 0.00063 | 0.00137 | 0.00182 | 0.00215 | 0.00266 | 0.00589 |
| contrast   | 637 | 37.11 | 4.33 | 24.25 | 33.99 | 36.91 | 39.80 | 55.31 |
| aspect     | 637 | ≈1.0 | ~0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

 **Conclusiones principales de las features:**

- Los canales RGB presentan valores casi idénticos → las imágenes son prácticamente monocromáticas (típico de ultrasonidos).
- El brillo, contraste, textura y entropía muestran una distribución estable sin outliers severos.
- `aspect` revela que todas las imágenes mantienen proporción cuadrada.
- La variación en `edge_sum` y `lbp_var` sugiere diferencias estructurales relevantes entre clases.

---

##  6. Visualizaciones generadas

Los gráficos producidos durante el EDA se encuentran en `results/figures/` e incluyen:

- distribución de clases (`class_distribution.png`)
- estadísticas visuales de features (`eda_feature_stats.png`)
- ejemplos del dataset
- gráficas exploratorias adicionales

Estos gráficos complementan el análisis cuantitativo presentado arriba.

---

##  7. Resumen del EDA

1. El dataset presenta un 'desbalance significativo', justificando el uso de 'class weights'.
2. Las features extraídas confirman que las imágenes son 'coherentes y estables' entre sí.
3. No se identificaron valores extremos que comprometan el entrenamiento.
4. Las diferencias entre clases parecen residir principalmente en 'textura, bordes y brillo', lo cual sustenta el uso de CNN + RF embeddings como estrategia de modelado.

---

##  Archivos generados por el EDA

Ubicados en:
- `Distribucion de clases.csv`
- `EDA Features.csv`

Ubicados en carpeta:
- `EDA IMAGES`
---

#  Fin del Reporte EDA

Este informe sirve como fundamento para las decisiones de preprocesamiento, balanceo y modelado aplicadas posteriormente en el proyecto.
