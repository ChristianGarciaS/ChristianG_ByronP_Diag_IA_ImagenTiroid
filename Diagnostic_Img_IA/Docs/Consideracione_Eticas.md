**Consideraciones\_Éticas**

**1. Análisis de sesgos**

**1.1 Sesgos potenciales del dataset**

El dataset utilizado contiene imágenes clasificadas como *benignas* y *malignas*. Sin embargo, presenta posibles fuentes de sesgo:

- **Sesgo de adquisición:** las imágenes provienen de diferentes fuentes, resoluciones, niveles de ruido e iluminación.
- **Sesgo clínico:** es probable que las imágenes correspondan a un subconjunto específico de pacientes, lo que puede implicar:
  - Diferencias culturales o demográficas no explícitas.
  - Variabilidad en equipos médicos (ecógrafos), operadores y protocolos de captura.
- **Desbalance de clases:** la cantidad de imágenes benignas y malignas no es igual, lo que puede inducir al modelo a favorecer la clase mayoritaria.

**1.2 Efectos de estos sesgos en las predicciones**

Los sesgos podrían impactar negativamente en:

- Sobre-diagnóstico de malignidad si el dataset maligno tiene características más limpias o uniformes.
- Sub-diagnóstico si los casos benignos son más diversos.
- Reducción de la capacidad de generalización en poblaciones no representadas.

**1.3 Grupos potencialmente perjudicados**

- Pacientes cuyos casos no se parecen a los ejemplos del dataset.
- Instituciones que utilicen protocolos de captura diferentes.
- Pacientes con nódulos atípicos o con comorbilidades no representadas.

**2. Equidad y fairness**

**2.1 ¿El modelo trata a todos los grupos de forma equitativa?**

No se evaluaron explícitamente métricas segmentadas por subgrupos demográficos, ya que el dataset no contiene etiquetas de edad, género o etnia.

Esto limita la capacidad de garantizar fairness completa.

**2.2 Métricas de fairness (si aplica)**

Dado que solo existen dos clases, se evaluó:

- **Balanced accuracy**
- **Matriz de confusión balanceada 1:1**
- **Class weights para mitigar desbalance**

**2.3 Estrategias de mitigación implementadas**

- Rebalanceo artificial del conjunto de validación (oversampling).
- Uso de class\_weight durante el entrenamiento del CNN.
- Fine-tuning del modelo para mejorar sensibilidad y reducir falsos negativos.

**3. Privacidad**

**3.1 Uso de datos personales**

Las imágenes no contienen datos identificables visibles (rostros, nombres, números de historia clínica).\
Sin embargo, siguen considerándose **datos médicos sensibles**, por lo que requieren protección especial.

**3.2 Protección implementada**

- Las imágenes se manejan únicamente en entorno controlado (Google Drive / Colab).
- No se almacena metadata EXIF ni información asociada al paciente.
- Los modelos entrenados no retienen imágenes, solo parámetros abstractos.

**3.3 Cumplimiento regulatorio**

De acuerdo a buenas prácticas de GDPR / HIPAA:

- No se conserva información identificable.
- No existe trazabilidad que relacione pacientes con predicciones.
- El informe PDF incluye un *disclaimer* indicando que es una herramienta de apoyo, no diagnóstico médico definitivo.

**4. Transparencia y explicabilidad**

**4.1 Interpretabilidad del modelo**

- Un CNN EfficientNetB0 es **poco interpretable por sí mismo**.
- El modelo Random Forest sobre embeddings es **más interpretable**, pero depende de representaciones complejas extraídas por la CNN.

**4.2 ¿Los usuarios entienden cómo funciona?**

Se explica el flujo general:

- Preprocesamiento
- Extracción de embeddings
- Clasificación CNN
- Clasificación RF
- Promedio ponderado de probabilidades

**4.3 Técnicas de explicabilidad sugeridas**

(No implementadas aún, pero recomendadas)

- **SHAP para embeddings**
- **Grad-CAM para resaltar regiones relevantes en la imagen**
- **LIME para explicabilidad por perturbaciones**

**5. Impacto social**

**5.1 Impacto positivo**

- Herramienta de apoyo para diagnóstico temprano.
- Facilita priorización clínica.
- Puede reducir carga de trabajo en centros de salud.

**5.2 Impactos negativos potenciales**

- Falsos negativos podrían retrasar diagnósticos de cáncer.
- Falsos positivos podrían generar ansiedad y procedimientos innecesarios.
- Uso inapropiado podría reemplazar indebidamente la evaluación médica profesional.

**5.3 Afectación a distintos actores**

**Beneficiados:**

- Médicos radiólogos
- Pacientes sin acceso a evaluaciones especializadas
- Instituciones pequeñas que necesitan apoyo diagnóstico

**Perjudicados potenciales:**

- Pacientes cuyos casos no cumplen patrones vistos por el modelo
- Profesionales que confíen ciegamente en la IA

**6. Responsabilidad**

**6.1 ¿Quién es responsable si el modelo falla?**

- El desarrollador del sistema debe documentar claramente sus limitaciones.
- El profesional de salud es responsable del diagnóstico final.
- El modelo solo puede ser considerado *herramienta de apoyo*.

**6.2 Mecanismos de accountability**

- Registro de versiones del modelo.
- Bitácora de cambios en el pipeline.
- Validación clínica antes de uso real.

**6.3 Plan de monitoreo**

- Reentrenamiento periódico con nuevos datos.
- Auditoría de métricas cada 6–12 meses.
- Implementación futura de un sistema de *model drift detection*.

**7. Uso dual y mal uso**

**7.1 Riesgos de mal uso**

- Utilizarlo como diagnóstico clínico sin supervisión médica.
- Implementarlo en entornos sin validación.
- Reutilizarlo para imágenes no tiroideas.

**7.2 Salvaguardas implementadas**

- PDF incluye una advertencia explícita.
- Se documenta que es solo una herramienta educativa.
- El modelo no se despliega públicamente.




**7.3 Limitaciones de uso declaradas**

- No es un dispositivo médico certificado.
- No reemplaza la biopsia o evaluación médica.

**8. Limitaciones reconocidas**

**8.1 Casos donde NO debe usarse**

- Diagnóstico definitivo sin intervención de un radiólogo.
- Pacientes fuera de distribución (imágenes con ruido extremo o artefactos).
- Equipos de imagen con características muy distintas a las del dataset.

**8.2 Advertencias a usuarios**

- Las métricas del modelo no equivalen a capacidad diagnóstica clínica real.
- El modelo puede fallar en casos raros no representados en el dataset.

**8.3 Casos límite**

- Nódulos con características mixtas.
- Lesiones quísticas atípicas.
- Imágenes con baja resolución o distorsionadas.

