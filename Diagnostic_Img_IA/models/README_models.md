#  Modelos Entrenados 

Esta carpeta contiene los modelos entrenados durante el desarrollo del sistema de clasificación de imágenes ecográficas de tiroides. Incluye el mejor modelo final y versiones anteriores utilizadas para comparación.

```
models/
├── best_model.pkl        # Mejor modelo guardado (RF_opt)
├── model_v1.pkl          # Versión inicial / histórica
└── README.md             # Descripción de modelos y comparación
```

---

# 1. Descripción general de los modelos

Durante el experimento se evaluaron dos enfoques principales:

### **A. Redes Neuronales Convolucionales (CNN)**
- **CNN_base** – Modelo inicial entrenado desde cero.  
- **CNN_finetune** – Versión optimizada aplicando *fine-tuning* sobre capas superiores.

### **B. Random Forest (RF)**
Utilizan características estadísticas extraídas de las imágenes:
- RF_base – Versión estándar.  
- RF_opt – Versión optimizada mediante búsqueda de hiperparámetros.

---

# 2. Resultados comparativos

| Modelo         | Accuracy |
|----------------|----------|
| CNN_base       | 0.622047 |
| CNN_finetune   | 0.622047 |
| RF_base        | 0.609375 |
| RF_opt         | 0.640625 |

---

# 3. Análisis comparativo

##  3.1 No mejora en CNN (0.62 → 0.62)

A pesar de aplicar optimización y *fine-tuning*, la CNN mantuvo la misma accuracy.  
Las razones más probables:

### **1. Dataset pequeño y con poca variabilidad**
Las CNN requieren una gran cantidad de imágenes para aprender representaciones robustas.  
El tamaño reducido del dataset limita la capacidad de aprendizaje → **underfitting persistente**.

### **2. Diferencias visuales mínimas en ecografías**
Los nódulos benignos y malignos presentan patrones muy similares visualmente.  
La CNN no encuentra diferencias suficientes para mejorar.

### **3. Fine-tuning insuficiente**
Aunque se ajustaron hiperparámetros, el modelo no logró aprender nuevas representaciones significativas.

### **4. Sensibilidad al desbalance**
Las CNN son más sensibles al desbalance entre clases que modelos tabulares.

> **Conclusión:** La arquitectura CNN está limitada por la naturaleza del dataset más que por el modelo en sí.

---

##  3.2 Mejora en Random Forest (0.60 → 0.64)

A diferencia de la CNN, RF trabaja con **características tabulares** generadas en el preprocesamiento.

### **1. Las características estadísticas son muy útiles**
El EDA del notebook generó features como:
- intensidad media,  
- varianza,  
- gradientes (Sobel),  
- filtros gaussianos,  
- textura (kurtosis, skewness),  
- histogramas.  

Estas métricas capturan diferencias sutiles entre clases.

### **2. RF funciona muy bien con datasets pequeños**
Los modelos de árbol no requieren tantos datos como una CNN.

### **3. La optimización tuvo impacto significativo**
El ajuste de hiperparámetros mejoró la generalización:
- `n_estimators` más altos,  
- Profundidades óptimas (`max_depth`),  
- Mejor control de sobreajuste.

> **Conclusión:** RF_opt es capaz de encontrar patrones que la CNN no puede capturar con el tamaño actual del dataset.

---

# 4. Modelo seleccionado (`best_model.pkl`)

El modelo seleccionado como mejor desempeño fue:

###  **Random Forest Optimizado (RF_opt)**
- Accuracy: **0.640625**  
- Mejor capacidad de generalización  
- Menor sensibilidad al dataset pequeño  
- Basado en características estadísticamente significativas  

---

# 5. Versiones anteriores

### `model_v1.pkl`
Contiene el modelo RF_base o una versión histórica del pipeline.  
Se mantiene para asegurar trazabilidad del experimento.

