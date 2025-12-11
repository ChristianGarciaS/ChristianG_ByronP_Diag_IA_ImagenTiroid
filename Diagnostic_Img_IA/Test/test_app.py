# tests/test_app.py
# Prueba la parte final de predicción e integración CNN + RF.

import numpy as np
import pytest

# IMPORTA FUNCIONALIDAD
from your_module import (
    build_transfer_model,
    feature_extractor,
    classes
)

def test_full_pipeline_prediction():
    '''Prueba pipeline completo CNN + RF (sin PDF).'''

    model = build_transfer_model()
    dummy = np.zeros((1, 224, 224, 3), dtype="float32")

    # Predicción CNN
    pred_cnn = model.predict(dummy)
    assert pred_cnn.shape == (1, 2)

    # Embeddings para RF (simulación)
    emb = feature_extractor.predict(dummy)
    assert emb.ndim == 2

    # Simulación de RF predicción (modelo real se entrena aparte)
    fake_rf_probs = np.array([[0.3, 0.7]])

    # Combinación (como en tu app final)
    combined_malign = (pred_cnn[0,1] + fake_rf_probs[0,1]) / 2
    assert 0 <= combined_malign <= 1


def test_classes_exist():
    '''Asegura que existan las clases.'''
    assert isinstance(classes, list)
    assert len(classes) == 2
    assert "benign" in classes
    assert "malignant" in classes
