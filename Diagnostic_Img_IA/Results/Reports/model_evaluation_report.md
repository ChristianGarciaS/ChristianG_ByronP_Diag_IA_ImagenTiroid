#  Model Evaluation Report

Este reporte resume la evaluación de los modelos utilizados en el proyecto de clasificación de ultrasonidos tiroideos.  
Incluye resultados del **CNN base**, **CNN fine‑tuned**, **Random Forest base** y **Random Forest optimizado**, así como sus métricas clave y análisis comparativo.

---

#  1. Modelos evaluados

El proyecto incluye cuatro modelos principales:

1. **CNN Base (Transfer Learning con EfficientNetB0 + capa densa final)**
2. **CNN Fine‑tuned (ajuste de capas superiores)**
3. **Random Forest Base (entrenado con embeddings del CNN)**
4. **Random Forest Optimizado (GridSearchCV)**

Cada modelo es evaluado con métricas clave del problema:  
**accuracy**, **precision**, **recall** y **f1-score**, además de análisis por clase cuando aplica.

---

#  2. Métricas globales (Accuracy)

| Modelo                   | Accuracy |
|-------------------------|----------|
| CNN Base                | 0.6377   |
| CNN Fine‑tuned          | 0.6377   |
| RF Base                 | 0.6094   |
| RF Optimizado           | 0.6406   |

 **Conclusión inicial:**  
El **Random Forest optimizado** obtiene el mejor desempeño global del proyecto.

---

#  3. Evaluación detallada por modelo

---

##  3.1 CNN Base

El modelo CNN inicial (sin fine‑tuning) obtiene:
- **Accuracy:** 0.6377  
- Entrena 50 epochs con estabilidad moderada.
- Mantiene una val_accuracy alrededor de 0.63–0.64.

 Observación:  
El modelo aprende, pero no logra diferenciar suficientemente entre clases debido a la limitada profundidad entrenable en el transfer learning inicial.

---

##  3.2 CNN Fine‑tuned

Resultados:

- **Accuracy:** 0.6377  
- Se observó mejora de estabilidad en entrenamiento, pero sin incremento real en accuracy.

 Observación:  
A pesar del fine‑tuning, el modelo no supera su versión base.  
Esto se debe a que:
- los features del ultrasonido son sutiles,
- la estructura del dataset es homogénea,
- el fine‑tuning parcial no aporta información nueva suficiente para mejorar la discriminación.

---

##  3.3 Random Forest Base (con embeddings)

El Random Forest entrenado sobre los embeddings de la CNN arroja:

- **Accuracy:** 0.6094

 Observación:  
La capacidad del RF base es comparable al CNN, pero sin superar significativamente los resultados.

---

##  3.4 Random Forest Optimizado (GridSearchCV)

El mejor desempeño del proyecto:

- **Accuracy:** 0.6406  
- **Mejores parámetros:**  
  - `max_depth = 10`  
  - `n_estimators = 200`

###  Clasification Report:

| Clase      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| benign     | 0.65      | 0.25   | 0.36     | 52      |
| malignant  | 0.64      | 0.91   | 0.75     | 76      |
| **accuracy** | —         | —      | **0.64** | 128     |
| macro avg  | 0.64      | 0.58   | 0.56     | 128     |
| weighted avg | 0.64    | 0.64   | 0.59     | 128     |

 Observaciones clave:

- El modelo es **muy fuerte clasificando malignos** (recall = 0.91).
- El rendimiento para benignos es más bajo debido al **desbalance del dataset**.
- RF optimizado captura mejor las variaciones texturales que la CNN.

---

#  4. Matrices de confusión

 Análisis:
- La CNN confunde fuertemente ambas clases por igual.
- El RF optimizado reduce falsos negativos (malignos predichos como benignos).

---

#  5. Comparación final de modelos

| Modelo               | Ventajas | Desventajas |
|---------------------|----------|-------------|
| **CNN Base**        | Usa deep features | Limitada capacidad discriminativa |
| **CNN Fine‑tuned**  | Mejor adaptación | No mejora accuracy |
| **RF Base**         | Rápido, simple | No supera CNN base |
| **RF Optimizado**   | **Mejor accuracy**, mejor recall maligno | Bajo recall benigno |

---

#  6. Modelo recomendado

El **Random Forest optimizado** es el modelo recomendado del proyecto debido a:

- Mejor accuracy general: **0.6406**
- Recall sobresaliente en clase “malignant” (prioritaria clínicamente)
- Aprovecha los embeddings del CNN para lograr un balance entre complejidad y desempeño

---

#  Archivos generados por esta evaluación

Ubicados en `results/metrics/`:

- `Accuracy antes y despues.csv`
- `Accuracy CNN base.csv`
- `Accuracy CNN y RF Optimizado.csv`
- `Accuracy RF Base.csv`
- `Reporte de clasificacion RF Optimizado.csv`

Ubicados en `results/figures/`:

- `rf_confusion_matrix.png`
- `cnn_vs_rf_comparison.png`

---

#  Fin del Model Evaluation Report

Este reporte sintetiza el desempeño de todos los modelos evaluados y justifica la selección del modelo final basado en criterios cuantitativos y clínicos.
