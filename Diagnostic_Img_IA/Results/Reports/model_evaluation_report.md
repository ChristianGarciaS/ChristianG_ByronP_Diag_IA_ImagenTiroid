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
| CNN Fine‑tuned          | 0.6692   |
| RF Base                 | 0.6796   |
| RF Optimizado           | 0.7109   |

 **Conclusión inicial:**  
El **Random Forest optimizado** obtiene el mejor desempeño global del proyecto.

---

#  3. Evaluación detallada por modelo

---

##  3.1 CNN Base

El modelo CNN inicial (sin fine‑tuning) obtiene:
- **Accuracy:** 0.6377  
- Entrena 300 epochs con estabilidad moderada.
- Mantiene una val_accuracy alrededor de 0.63–0.64.

 Observación:  
El modelo aprende, pero no logra diferenciar suficientemente entre clases debido a la limitada profundidad entrenable en el transfer learning inicial.

---

##  3.2 CNN Fine‑tuned

Resultados:

- **Accuracy:** 0.6692  
- Se observó mejora de estabilidad en entrenamiento, con incremento real en accuracy.

 Observación:  
Implementado fine‑tuning, el modelo SI supera su versión base.  
Esto se debe a que:
- los features del ultrasonido son sutiles,
- la estructura del dataset ya no es tan homogénea,
- el fine‑tuning parcial si aportó información nueva suficiente para mejorar la discriminación.

---

##  3.3 Random Forest Base (con embeddings)

El Random Forest entrenado sobre los embeddings de la CNN arroja:

- **Accuracy:** 0.6796 

 Observación:  
La capacidad del RF base es un tanto comparable al CNN, pero supera significativamente los resultados.

---

##  3.4 Random Forest Optimizado (GridSearchCV)

El mejor desempeño del proyecto:

- **Accuracy:** 0.7109  
- **Mejores parámetros:**  
  - `max_depth = 50  
  - `n_estimators = 300`

###  Clasification Report:

| Clase      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
 benign      |      0.80 |   0.38 |     0.52 |      52 |
 malignant   |      0.69 |   0.93 |     0.79 |      76 |
 accuracy    |           |        |     0.71 |     128 |
 macro avg   |      0.74 |   0.66 |     0.66 |     128 |
weighted avg |      0.73 |   0.71 |     0.68 |     128 |

 Observaciones clave:

- El modelo es **muy fuerte clasificando ambas clases ** (Precision = 0.80 - 0.69).
- El rendimiento para benignos es un poco bajo debido al **menor cantidad de imágenes benignas**
- RF optimizado captura mejor las variaciones texturales que la CNN.

---

#  4. Matrices de confusión

 Análisis:
- La CNN optimizado reduce en algo falsos negativos.
- El RF optimizado reduce falsos negativos (malignos predichos como benignos).

---

#  5. Comparación final de modelos

| Modelo               | Ventajas | Desventajas |
|---------------------|----------|-------------|
| **CNN Base**        | Usa deep features | Limitada capacidad discriminativa |
| **CNN Fine‑tuned**  | Mejor adaptación  | Si mejora accuracy |
| **RF Base**         | Rápido, simple    | Si supera CNN base |
| **RF Optimizado**   | **Mejor accuracy**, mejor F1-Score maligno | Medio F1-Score benigno |

---

#  6. Modelo recomendado

El **Random Forest optimizado** es el modelo recomendado del proyecto debido a:

- Mejor accuracy general: **0.7109**
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
