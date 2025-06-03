# src/explainability.py

import shap
import matplotlib.pyplot as plt
import numpy as np
import os

def crear_explainer(modelo, datos):
    """
    Crea un explainer SHAP para el modelo dado.
    """
    explainer = shap.Explainer(modelo, datos)
    return explainer

def graficar_importancia_global_shap(explainer, datos, nombres_columnas, nombre_archivo=None):
    """
    Genera y muestra el SHAP summary plot (importancia global).
    """
    shap_values = explainer(datos)

    # Visualización
    shap.summary_plot(shap_values, features=datos, feature_names=nombres_columnas, show=False)
    if nombre_archivo:
        plt.savefig(nombre_archivo, bbox_inches="tight")
    plt.show()

def graficar_explicacion_individual_shap(explainer, datos, index=0, nombres_columnas=None, nombre_archivo=None):
    """
    Genera y muestra un SHAP waterfall plot para una observación individual.
    """
    shap_values = explainer(datos)
    
    # Visualización
    shap.plots.waterfall(shap_values[index], max_display=10, show=False)
    if nombre_archivo:
        plt.savefig(nombre_archivo, bbox_inches="tight")
    plt.show()

def asegurar_directorio_salida(path="../outputs/plots"):
    os.makedirs(path, exist_ok=True)

