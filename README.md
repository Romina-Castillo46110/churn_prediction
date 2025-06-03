# 📦 Churn Prediction – Telco Customer Churn

Este proyecto de ciencia de datos busca predecir la probabilidad de que un cliente abandone una compañía de telecomunicaciones (*churn*) utilizando datos históricos y técnicas de machine learning.

---

## 📁 Estructura del Proyecto
```
churn-prediction/
│
├── data/ # Datos en crudo y procesados
│ └── raw/
│
├── models/ # Modelos y preprocesador entrenados (joblib)
│
├── notebooks/ # Notebooks ordenados por etapa
│ ├── 01_preprocessing.ipynb
│ ├── 02_modeling.ipynb
│ ├── 03_interpretability.ipynb
│ └── 04_final_evaluation.ipynb
│
├── outputs/
│ ├── plots/ # Gráficos generados (métricas, ROC, SHAP)
│ ├── metrics/ # Métricas exportadas si corresponde
│ └── predictions/ # Predicciones exportadas si se guardan
│
├── src/ # Módulos reutilizables de Python
│ └── preprocessing.py
│
├── README.md # Este archivo
└── requirements.txt # Requerimientos del entorno
```
---

## 🚀 Flujo de Trabajo

### 1. Preprocesamiento (`01_preprocessing.ipynb`)

- Carga y limpieza de datos
- Conversión de columnas categóricas y numéricas
- División en train / validation / test
- Guardado del preprocesador como `preprocessor.joblib`

### 2. Modelado (`02_modeling.ipynb`)

- Entrenamiento de 3 modelos:
  - **Regresión Logística**
  - **Random Forest**
  - **XGBoost**
- Evaluación con F1, ROC AUC, precisión y recall
- Visualizaciones de resultados y matriz de confusión
- Modelos guardados en `models/`

### 3. Interpretabilidad (`03_interpretability.ipynb`)

Se utilizó SHAP (SHapley Additive exPlanations) para interpretar el modelo XGBoost:

- SHAP summary plot: permite entender cuáles son las variables que más influyen en  el modelo globalmente.
-SHAP waterfall plot: se utilizó para explicar la predicción de un cliente específico, visualizando cómo cada variable afectó la decisión del modelo.

### 4. Evaluación Final (`04_final_evaluation.ipynb`)
- Carga de modelos entrenados
- Comparación de métricas
- Visualización final de resultados (barras, ROC, confusión)

---

## 📊 Visualizaciones Principales

- Comparación de métricas (F1, AUC, etc.)
- Curvas ROC por modelo
- Matrices de confusión
- SHAP summary plot (importancia global)
- SHAP waterfall plot (explicación individual)

---

## 🛠️ Instalación

1. Clonar el repositorio:

```bash
    git clone https://github.com/tu_usuario/churn-prediction.git
    cd churn-prediction 
```
2. Crear entorno virtual e instalar dependencias:

```bash
    python -m venv venv
    source venv/bin/activate  # en Windows: venv\Scripts\activate
    pip install -r requirements.txt
```
3. Ejecutar los notebooks desde Jupyter o VS Code.

---

## 🧠 Tecnologías y Librerías

- Python 3.12

- Pandas, NumPy, Scikit-learn

- XGBoost, SHAP

- Matplotlib, Seaborn

---

## 👩‍💻 Autora

- Romina Castillo – Data Scientist en formación

