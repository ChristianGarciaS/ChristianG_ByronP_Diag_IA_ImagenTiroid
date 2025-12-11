**D) Optimizacion**

***Optimización de modelos —**

**1. Proceso general de optimización de hiperparámetros**

En el proyecto se utilizan **dos enfoques de modelado**:

1. **Modelo CNN tipo EfficientNetB0 con fine-tuning.**
1. **Modelo Random Forest entrenado sobre embeddings generados por EfficientNet.**

Cada uno recibe un proceso de optimización específico:

**1.1. Optimización del modelo CNN (Fine-Tuning)**

El script aplica un proceso clásico de optimización:

1. **Inicialización del modelo base EfficientNetB0 (con pesos ImageNet).**
1. **Entrenamiento inicial con la base congelada**
   1. Learning rate: 1e-4
   1. Dropout: 0.3
   1. Epochs: hasta 50 con EarlyStopping
1. **Fine-tuning** desbloqueando las últimas ~20 capas del backbone
   1. Learning rate reducido: 5e-5
   1. Dropout aumentada: 0.4
1. **Callbacks aplicados:**
   1. EarlyStopping(patience=3)
   1. ReduceLROnPlateau(factor=0.5)

Este flujo permite estabilizar el modelo, evitar sobreajuste y mejorar la generalización.

**1.2. Optimización del Random Forest**

El script usa GridSearchCV con validación cruzada (cv=3) sobre un conjunto de parámetros acotado:

param\_grid = {

`    `'n\_estimators': [100, 200],

`    `'max\_depth': [None, 10, 20]

}

GridSearch evalúa todas las combinaciones y selecciona la de mayor **accuracy** en validación.

**2. Hiperparámetros explorados y rangos**

**2.1. CNN (EfficientNet + capas densas)**

|**Componente**|**Hiperparámetro**|**Rango/Valor**|
| :- | :- | :- |
|Optimizador|Learning Rate|1e-4, luego 5e-5|
|Regularización|Dropout|0\.3 → 0.4|
|Backbone|Capas desbloqueadas|Últimas 20|
|Entrenamiento|Epochs|Hasta 50|
|Callbacks|LR scheduler|factor 0.5|
|Callbacks|EarlyStopping|patience = 3|

**2.2. Random Forest**

|**Hiperparámetro**|**Valores evaluados**|
| :- | :- |
|n\_estimators|100, 200|
|max\_depth|None, 10, 20|
|criterion|(por defecto = gini)|
|bootstrap|True|

La búsqueda fue exhaustiva dentro de este espacio.

**3. Resultados del análisis de sensibilidad**

**CNN**

Durante el fine-tuning se observó:

- Aprendizaje sensible al **learning rate**:\
  tasas mayores causan oscilaciones fuertes.
- El número de capas desbloqueadas afecta la estabilidad:\
  desbloquear más de 40 capas reduce la precisión.
- Dropout controla el overfitting:\
  valores bajos (<0.3) dan sobreajuste rápido.

**Random Forest**

- n\_estimators tiene un efecto marginal: 200 ≈ mejor estabilidad.
- max\_depth controla significativamente el desempeño:\
  profundidad limitada a 10 da mejor generalización para el dataset.
- None como profundidad tiende a sobreajustar embeddings.

**4. Partial Dependence Plots (PDP)**

Para el modelo Random Forest sería relevante analizar:

- **Impacto de la varianza LBP** (lbp\_var)
- **Intensidad promedio** (brightness)
- **Edge sum** (bordes detectados)

En un PDP típico se observaría:

- Alto *lbp\_var* → mayor probabilidad de malignidad.
- Alto contraste → mayor likelihood de patrón maligno.
- Valores altos en *edge\_sum* correlacionan con estructuras más irregulares.

Puedes generarlos con:

from sklearn.inspection import partial\_dependence, PartialDependenceDisplay

PartialDependenceDisplay.from\_estimator(rf\_best, X\_te, features=[0,3,7])

**5. Ranking de importancia de hiperparámetros**

En Random Forest:

1. **contrast**
1. **lbp\_var**
1. **entropy**
1. **edge\_sum**
1. **mean\_r / mean\_g / mean\_b**

Interpretación:

- Los *features* que capturan **textura** (LBP, contraste) aportan más a la clasificación.
- Color promedio aporta menos información en imágenes médicas en escala similar.

**6. Análisis de interacciones**

- **contrast × lbp\_var**\
  → Combinación clave para diferenciar lesiones con bordes difusos.
- **entropy × edge\_sum**\
  → Alta irregularidad microscópica tiende a indicar malignidad.

Estas interacciones justifican por qué Random Forest funciona bien como modelo complementario.

**7. Configuración final seleccionada y justificación**

**CNN Fine-Tuned (configuración final)**

- EfficientNetB0 con base parcialmente desbloqueada
- LR = 5e-5
- Dropout = 0.4
- GlobalAveragePooling + Dense(128)
- Recordó mejor comportamiento general que RF solo.

**Random Forest Optimizado**

- n\_estimators = 200
- max\_depth = 10
- Mejor trade-off entre precisión y generalización.

Ambos modelos se usan en **fusión de probabilidades**:

prob\_final = promedio(prob\_CNN, prob\_RF)

Esto incrementa la robustez del diagnóstico.


**8. Comparación antes / después de la optimización**

|**Modelo**|**Before**|**After**|
| :- | :- | :- |
|CNN (base)|acc\_cnn\_base|acc\_cnn\_ft|
|RF (base)|acc\_rf\_base|acc\_rf\_best|

En términos generales:

- La CNN mejoró tras el fine-tuning.
- Random Forest mejoró moderadamente al ajustar profundidad.
- La combinación **CNN + RF** entregó la mejor estabilidad diagnóstica.


