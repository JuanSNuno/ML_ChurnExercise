# Proyecto de Machine Learning: Guía Completa desde la Recepción de Datos

## 1. Introducción al Proyecto y Definición del Problema de Negocio

### Objetivo de esta sección
Presentar el proyecto y vincular el problema de negocio con un problema de Machine Learning. Es crucial entender el problema antes de proponer una solución.

### 1.1 Definición del Problema Empresarial

En este notebook, trabajaremos con un ejemplo práctico: **Predicción de abandono de clientes (Churn)** en una empresa de telecomunicaciones. 

El problema de negocio es el siguiente:
- La empresa está perdiendo aproximadamente 26% de sus clientes anualmente
- Adquirir un nuevo cliente cuesta 5x más que retener uno existente
- Necesitamos identificar clientes en riesgo de abandono para tomar acciones preventivas

```python
# Importación de librerías necesarias para todo el proyecto
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('ggplot')
%matplotlib inline
```

### 1.2 Mapeo a un Problema de Machine Learning

Este problema de negocio se traduce en:
- **Tipo de problema ML**: Clasificación binaria supervisada
- **Variable objetivo**: Churn (Sí/No)
- **Features**: Características del cliente, uso del servicio, información de facturación

```python
# Definición del problema en términos de ML
problema_ml = {
    'tipo': 'Clasificación Binaria',
    'variable_objetivo': 'Churn',
    'clases': ['No abandona (0)', 'Abandona (1)'],
    'enfoque': 'Aprendizaje Supervisado',
    'metricas_clave': ['Precision', 'Recall', 'F1-Score', 'AUC-ROC']
}

print("=== DEFINICIÓN DEL PROBLEMA DE ML ===")
for key, value in problema_ml.items():
    print(f"{key}: {value}")
```

### 1.3 Criterios de Éxito

```python
# Definición de criterios de éxito
criterios_exito = {
    'ml_metrics': {
        'precision_minima': 0.85,
        'recall_minimo': 0.80,
        'f1_score_minimo': 0.82
    },
    'business_metrics': {
        'reduccion_churn_esperada': '15%',
        'roi_esperado': '3:1',
        'tiempo_implementacion': '3 meses'
    }
}

print("=== CRITERIOS DE ÉXITO ===")
print("\nMétricas de Machine Learning:")
for metric, value in criterios_exito['ml_metrics'].items():
    print(f"  - {metric}: {value}")
print("\nMétricas de Negocio:")
for metric, value in criterios_exito['business_metrics'].items():
    print(f"  - {metric}: {value}")
```

### 1.4 Consideraciones Iniciales

```python
# Evaluación de viabilidad del proyecto ML
consideraciones = {
    'es_problema_ml': {
        'complejidad_logica': 'Alta - múltiples factores interrelacionados',
        'volumen_datos': 'Suficiente - 10,000+ registros históricos',
        'variable_objetivo_clara': 'Sí - cliente abandona o no',
        'patron_identificable': 'Probable - comportamiento antes del abandono'
    },
    'cuando_usar_ml': {
        'check_1': '✓ Lógica compleja difícil de codificar',
        'check_2': '✓ Gran volumen de datos disponibles',
        'check_3': '✓ Variable objetivo bien definida',
        'check_4': '✓ Patrones históricos disponibles'
    },
    'riesgos': {
        'calidad_datos': 'Verificar completitud y consistencia',
        'sesgo_historico': 'Revisar representatividad de muestras',
        'cambios_mercado': 'Modelo puede degradarse con el tiempo'
    }
}

print("=== ANÁLISIS DE VIABILIDAD ===")
print("\n¿Es realmente un problema de ML?")
for aspecto, valor in consideraciones['es_problema_ml'].items():
    print(f"  {aspecto}: {valor}")
```

## 2. Adquisición y Análisis Exploratorio de Datos (EDA)

### Objetivo de esta sección
Entender la naturaleza de los datos recibidos, identificar patrones, anomalías y preparar el terreno para el preprocesamiento.

### 2.1 Recopilación de Datos

```python
# Simulación de carga de datos (en un caso real, cargarías desde tu fuente)
# Para este ejemplo, crearemos un dataset sintético representativo

np.random.seed(42)
n_samples = 5000

# Generación de datos sintéticos de clientes de telecomunicaciones
data = {
    'CustomerID': range(1, n_samples + 1),
    'Tenure': np.random.randint(0, 72, n_samples),  # Meses como cliente
    'MonthlyCharges': np.random.uniform(20, 120, n_samples),
    'TotalCharges': np.random.uniform(100, 8000, n_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.25, 0.25]),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet'], n_samples),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet'], n_samples),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
    'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    'Partner': np.random.choice(['Yes', 'No'], n_samples),
    'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
}

# Variable objetivo con correlación realista
churn_probability = []
for i in range(n_samples):
    prob = 0.15  # Probabilidad base
    if data['Contract'][i] == 'Month-to-month':
        prob += 0.3
    if data['Tenure'][i] < 12:
        prob += 0.2
    if data['MonthlyCharges'][i] > 80:
        prob += 0.1
    if data['TechSupport'][i] == 'No':
        prob += 0.1
    churn_probability.append(min(prob, 0.9))

data['Churn'] = np.random.binomial(1, churn_probability)

# Crear DataFrame
df = pd.DataFrame(data)

print("=== INFORMACIÓN DEL DATASET ===")
print(f"Dimensiones del dataset: {df.shape}")
print(f"Número de clientes: {df.shape[0]}")
print(f"Número de características: {df.shape[1] - 1}")  # -1 por la variable objetivo
print(f"\nPrimeras 5 filas del dataset:")
df.head()
```

### 2.2 Tipos de Datos e Información Básica

```python
# Análisis de tipos de datos
print("=== TIPOS DE DATOS ===")
print(df.dtypes)

# Información general del dataset
print("\n=== INFORMACIÓN GENERAL DEL DATASET ===")
df.info()

# Identificación de características numéricas y categóricas
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nCaracterísticas numéricas ({len(numerical_features)}): {numerical_features}")
print(f"Características categóricas ({len(categorical_features)}): {categorical_features}")
```

### 2.3 Estadística Descriptiva

```python
# Estadísticas descriptivas para variables numéricas
print("=== ESTADÍSTICAS DESCRIPTIVAS - VARIABLES NUMÉRICAS ===")
df[numerical_features].describe()
```

```python
# Análisis de la variable objetivo
print("=== DISTRIBUCIÓN DE LA VARIABLE OBJETIVO (CHURN) ===")
churn_dist = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100

print("Conteo absoluto:")
print(churn_dist)
print("\nPorcentaje:")
print(churn_pct)

# Visualización
plt.figure(figsize=(8, 6))
df['Churn'].value_counts().plot(kind='bar')
plt.title('Distribución de Churn')
plt.xlabel('Churn (0 = No, 1 = Sí)')
plt.ylabel('Cantidad de clientes')
plt.xticks(rotation=0)
plt.show()

# Verificar si hay desbalance de clases
if churn_pct.min() < 20:
    print("\n⚠️ ADVERTENCIA: Dataset desbalanceado detectado. Considerar técnicas de balanceo.")
```

### 2.4 Exploración Visual de Datos

```python
# Matriz de correlación para variables numéricas
plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlación - Variables Numéricas')
plt.tight_layout()
plt.show()
```

```python
# Análisis de distribuciones por variable objetivo
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Tenure vs Churn
axes[0, 0].hist([df[df['Churn']==0]['Tenure'], df[df['Churn']==1]['Tenure']], 
                label=['No Churn', 'Churn'], bins=20, alpha=0.7)
axes[0, 0].set_xlabel('Tenure (meses)')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].set_title('Distribución de Tenure por Churn')
axes[0, 0].legend()

# MonthlyCharges vs Churn
axes[0, 1].hist([df[df['Churn']==0]['MonthlyCharges'], df[df['Churn']==1]['MonthlyCharges']], 
                label=['No Churn', 'Churn'], bins=20, alpha=0.7)
axes[0, 1].set_xlabel('Cargos Mensuales')
axes[0, 1].set_ylabel('Frecuencia')
axes[0, 1].set_title('Distribución de Cargos Mensuales por Churn')
axes[0, 1].legend()

# Contract type vs Churn
contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
contract_churn.plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Tasa de Churn por Tipo de Contrato')
axes[1, 0].set_ylabel('Porcentaje (%)')
axes[1, 0].set_xlabel('Tipo de Contrato')
axes[1, 0].legend(['No Churn', 'Churn'])

# PaymentMethod vs Churn
payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index') * 100
payment_churn.plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Tasa de Churn por Método de Pago')
axes[1, 1].set_ylabel('Porcentaje (%)')
axes[1, 1].set_xlabel('Método de Pago')
axes[1, 1].legend(['No Churn', 'Churn'])

plt.tight_layout()
plt.show()
```

### 2.5 Detección de Problemas en los Datos

```python
# Verificación de valores faltantes
print("=== ANÁLISIS DE VALORES FALTANTES ===")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Columna': missing_values.index,
    'Valores_Faltantes': missing_values.values,
    'Porcentaje': missing_percentage.values
})

print(missing_df[missing_df['Valores_Faltantes'] > 0])

if missing_values.sum() == 0:
    print("✅ No se encontraron valores faltantes en el dataset")

# Verificación de duplicados
print("\n=== ANÁLISIS DE DUPLICADOS ===")
duplicates = df.duplicated().sum()
print(f"Número de filas duplicadas: {duplicates}")

if duplicates > 0:
    print(f"Porcentaje de duplicados: {(duplicates/len(df))*100:.2f}%")
else:
    print("✅ No se encontraron registros duplicados")
```

```python
# Detección de outliers usando el método IQR
print("=== DETECCIÓN DE OUTLIERS ===")

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Análisis de outliers para variables numéricas clave
for col in ['Tenure', 'MonthlyCharges', 'TotalCharges']:
    outliers, lower, upper = detect_outliers_iqr(df, col)
    print(f"\n{col}:")
    print(f"  - Límite inferior: {lower:.2f}")
    print(f"  - Límite superior: {upper:.2f}")
    print(f"  - Número de outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

# Visualización de outliers
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, col in enumerate(['Tenure', 'MonthlyCharges', 'TotalCharges']):
    df.boxplot(column=col, ax=axes[idx])
    axes[idx].set_title(f'Boxplot de {col}')
plt.tight_layout()
plt.show()
```

## 3. Ingeniería de Características (Feature Engineering) y Preprocesamiento

### Objetivo de esta sección
Transformar los datos crudos en un formato que sea más adecuado para el modelado de Machine Learning, mejorando el rendimiento y la robustez del modelo.

### 3.1 Limpieza de Datos

```python
# Crear una copia del dataset para preprocesamiento
df_preprocessed = df.copy()

# Manejo de valores faltantes (si existieran)
# En este caso no hay, pero aquí está el código para manejarlos

# Para variables numéricas: imputación con mediana
numerical_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges']
for col in numerical_cols:
    if df_preprocessed[col].isnull().sum() > 0:
        median_value = df_preprocessed[col].median()
        df_preprocessed[col].fillna(median_value, inplace=True)
        print(f"Imputados {col} valores faltantes con mediana: {median_value}")

# Para variables categóricas: imputación con moda
categorical_cols = [col for col in categorical_features if col != 'CustomerID']
for col in categorical_cols:
    if df_preprocessed[col].isnull().sum() > 0:
        mode_value = df_preprocessed[col].mode()[0]
        df_preprocessed[col].fillna(mode_value, inplace=True)
        print(f"Imputados {col} valores faltantes con moda: {mode_value}")

print("✅ Limpieza de datos completada")
```

### 3.2 Creación de Características Derivadas

```python
# Feature Engineering: Crear nuevas características basadas en conocimiento del dominio

# 1. Ratio de cargos totales sobre tenure (gasto promedio mensual real)
df_preprocessed['AvgChargesPerMonth'] = np.where(
    df_preprocessed['Tenure'] > 0,
    df_preprocessed['TotalCharges'] / df_preprocessed['Tenure'],
    df_preprocessed['MonthlyCharges']
)

# 2. Categorización de tenure
df_preprocessed['TenureCategory'] = pd.cut(
    df_preprocessed['Tenure'],
    bins=[-1, 12, 24, 48, 72],
    labels=['Nuevo', 'Regular', 'Establecido', 'Leal']
)

# 3. Indicador de servicio premium
df_preprocessed['PremiumServices'] = (
    (df_preprocessed['OnlineSecurity'] == 'Yes').astype(int) +
    (df_preprocessed['TechSupport'] == 'Yes').astype(int)
)

# 4. Indicador de cliente de alto valor
high_value_threshold = df_preprocessed['MonthlyCharges'].quantile(0.75)
df_preprocessed['HighValueCustomer'] = (
    df_preprocessed['MonthlyCharges'] > high_value_threshold
).astype(int)

# 5. Indicador de compromiso (contrato largo + sin factura en papel)
df_preprocessed['EngagementScore'] = 0
df_preprocessed.loc[df_preprocessed['Contract'] == 'Two year', 'EngagementScore'] += 2
df_preprocessed.loc[df_preprocessed['Contract'] == 'One year', 'EngagementScore'] += 1
df_preprocessed.loc[df_preprocessed['PaperlessBilling'] == 'Yes', 'EngagementScore'] += 1

print("=== NUEVAS CARACTERÍSTICAS CREADAS ===")
new_features = ['AvgChargesPerMonth', 'TenureCategory', 'PremiumServices', 
                'HighValueCustomer', 'EngagementScore']
print(f"Características nuevas: {new_features}")
print(f"\nTotal de características ahora: {df_preprocessed.shape[1]}")

# Mostrar estadísticas de las nuevas características
df_preprocessed[new_features].describe()
```

### 3.3 Codificación de Variables Categóricas

```python
# Preparar datos para modelado
# Separar CustomerID ya que no es una característica predictiva
customer_ids = df_preprocessed['CustomerID']
df_model = df_preprocessed.drop('CustomerID', axis=1)

# Separar variable objetivo
X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

# Identificar columnas categóricas para codificar
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
print(f"Columnas categóricas a codificar: {categorical_columns}")

# One-Hot Encoding para variables categóricas
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

print(f"\nDimensiones después de One-Hot Encoding:")
print(f"Antes: {X.shape}")
print(f"Después: {X_encoded.shape}")

# Mostrar algunas de las nuevas columnas creadas
print("\nEjemplo de nuevas columnas creadas:")
new_columns = [col for col in X_encoded.columns if col not in X.columns]
print(new_columns[:10])  # Mostrar primeras 10
```

### 3.4 Normalización de Datos

```python
# Identificar columnas numéricas para normalizar
numerical_cols_to_scale = ['Tenure', 'MonthlyCharges', 'TotalCharges', 
                           'AvgChargesPerMonth', 'EngagementScore']

# Crear una copia para preservar los datos originales
X_scaled = X_encoded.copy()

# Aplicar StandardScaler a las columnas numéricas
scaler = StandardScaler()
X_scaled[numerical_cols_to_scale] = scaler.fit_transform(X_scaled[numerical_cols_to_scale])

print("=== NORMALIZACIÓN COMPLETADA ===")
print("\nEstadísticas después de la normalización:")
print(X_scaled[numerical_cols_to_scale].describe())

# Visualizar el efecto de la normalización
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Antes de normalizar
X_encoded['MonthlyCharges'].hist(bins=30, ax=axes[0], alpha=0.7)
axes[0].set_title('Distribución de MonthlyCharges - Original')
axes[0].set_xlabel('Valor')

# Después de normalizar
X_scaled['MonthlyCharges'].hist(bins=30, ax=axes[1], alpha=0.7)
axes[1].set_title('Distribución de MonthlyCharges - Normalizado')
axes[1].set_xlabel('Valor normalizado')

plt.tight_layout()
plt.show()
```

### 3.5 Manejo de Desbalance de Clases

```python
# Análisis del desbalance
from sklearn.utils import class_weight

print("=== ANÁLISIS DE BALANCE DE CLASES ===")
class_counts = y.value_counts()
class_ratio = class_counts[0] / class_counts[1]

print(f"Distribución de clases:")
print(class_counts)
print(f"\nRatio de clases (No Churn : Churn): {class_ratio:.2f}:1")

# Calcular pesos de clases para usar en el modelo
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y),
    y=y
)
class_weight_dict = dict(enumerate(class_weights))

print(f"\nPesos de clase calculados: {class_weight_dict}")
print("\nEstos pesos se usarán durante el entrenamiento para compensar el desbalance")
```

## 4. Construcción y Evaluación del Modelo

### Objetivo de esta sección
Seleccionar, entrenar y evaluar el modelo de Machine Learning que mejor se adapte al problema y a los datos preparados.

### 4.1 División de Datos

```python
# División estratificada para mantener la proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Crear conjunto de validación del conjunto de entrenamiento
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print("=== DIVISIÓN DE DATOS ===")
print(f"Conjunto completo: {X_scaled.shape[0]} muestras")
print(f"Entrenamiento: {X_train_final.shape[0]} muestras ({X_train_final.shape[0]/X_scaled.shape[0]*100:.1f}%)")
print(f"Validación: {X_val.shape[0]} muestras ({X_val.shape[0]/X_scaled.shape[0]*100:.1f}%)")
print(f"Prueba: {X_test.shape[0]} muestras ({X_test.shape[0]/X_scaled.shape[0]*100:.1f}%)")

# Verificar que la estratificación funcionó
print("\nDistribución de clases en cada conjunto:")
print(f"Train: {y_train_final.value_counts(normalize=True).round(3).to_dict()}")
print(f"Val: {y_val.value_counts(normalize=True).round(3).to_dict()}")
print(f"Test: {y_test.value_counts(normalize=True).round(3).to_dict()}")
```

### 4.2 Entrenamiento de Múltiples Modelos

```python
# Función helper para evaluar modelos
def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    # Entrenar modelo
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Métricas
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'val_accuracy': accuracy_score(y_val, y_pred_val),
        'train_precision': precision_score(y_train, y_pred_train),
        'val_precision': precision_score(y_val, y_pred_val),
        'train_recall': recall_score(y_train, y_pred_train),
        'val_recall': recall_score(y_val, y_pred_val),
        'train_f1': f1_score(y_train, y_pred_train),
        'val_f1': f1_score(y_val, y_pred_val)
    }
    
    print(f"\n=== RESULTADOS PARA {model_name} ===")
    print(f"Accuracy - Train: {metrics['train_accuracy']:.4f}, Val: {metrics['val_accuracy']:.4f}")
    print(f"Precision - Train: {metrics['train_precision']:.4f}, Val: {metrics['val_precision']:.4f}")
    print(f"Recall - Train: {metrics['train_recall']:.4f}, Val: {metrics['val_recall']:.4f}")
    print(f"F1-Score - Train: {metrics['train_f1']:.4f}, Val: {metrics['val_f1']:.4f}")
    
    return model, metrics

# Entrenar diferentes modelos
models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', 
        random_state=42,
        max_iter=1000
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',
        class_weight='balanced',
        random_state=42,
        probability=True
    )
}

trained_models = {}
model_metrics = {}

for name, model in models.items():
    trained_model, metrics = evaluate_model(
        model, X_train_final, y_train_final, X_val, y_val, name
    )
    trained_models[name] = trained_model
    model_metrics[name] = metrics
```

### 4.3 Optimización de Hiperparámetros

```python
# Optimización del mejor modelo (Random Forest en este caso)
print("=== OPTIMIZACIÓN DE HIPERPARÁMETROS - RANDOM FOREST ===")

# Definir grid de parámetros
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Crear modelo base
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)

# Grid Search con Cross Validation
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Entrenar
print("Iniciando Grid Search...")
grid_search.fit(X_train_final, y_train_final)

# Mejores parámetros
print(f"\nMejores parámetros: {grid_search.best_params_}")
print(f"Mejor score F1 (CV): {grid_search.best_score_:.4f}")

# Evaluar el mejor modelo
best_model = grid_search.best_estimator_
y_pred_val_best = best_model.predict(X_val)

print("\nRendimiento del modelo optimizado en validación:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_val_best):.4f}")
print(f"Precision: {precision_score(y_val, y_pred_val_best):.4f}")
print(f"Recall: {recall_score(y_val, y_pred_val_best):.4f}")
print(f"F1-Score: {f1_score(y_val, y_pred_val_best):.4f}")
```

### 4.4 Evaluación Final en Conjunto de Prueba

```python
# Evaluación final con el mejor modelo
print("=== EVALUACIÓN FINAL EN CONJUNTO DE PRUEBA ===")

# Predicciones en test
y_pred_test = best_model.predict(X_test)
y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]

# Métricas finales
final_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_test),
    'precision': precision_score(y_test, y_pred_test),
    'recall': recall_score(y_test, y_pred_test),
    'f1_score': f1_score(y_test, y_pred_test)
}

print("Métricas en conjunto de prueba:")
for metric, value in final_metrics.items():
    print(f"{metric}: {value:.4f}")

# Verificar si cumplimos los criterios de éxito
print("\n=== VERIFICACIÓN DE CRITERIOS DE ÉXITO ===")
for metric, threshold in criterios_exito['ml_metrics'].items():
    metric_name = metric.replace('_minima', '').replace('_minimo', '')
    actual_value = final_metrics.get(metric_name, 0)
    status = "✅" if actual_value >= threshold else "❌"
    print(f"{metric}: {actual_value:.4f} (objetivo: {threshold}) {status}")

# Matriz de confusión
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Conjunto de Prueba')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Reporte de clasificación detallado
print("\n=== REPORTE DE CLASIFICACIÓN DETALLADO ===")
print(classification_report(y_test, y_pred_test, 
                          target_names=['No Churn', 'Churn']))
```

### 4.5 Análisis de Importancia de Características

```python
# Importancia de características del modelo Random Forest
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

# Top 20 características más importantes
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(20)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Importancia')
plt.title('Top 20 Características más Importantes')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("=== TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES ===")
print(feature_importance.head(10))
```

### 4.6 Curva ROC y AUC

```python
from sklearn.metrics import roc_curve, auc

# Calcular curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test)
roc_auc = auc(fpr, tpr)

# Visualizar curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

print(f"AUC-ROC Score: {roc_auc:.4f}")
```

### 4.7 Análisis de Errores y Diagnóstico del Modelo

```python
# Análisis de errores - identificar patrones en las predicciones incorrectas
X_test_with_predictions = X_test.copy()
X_test_with_predictions['y_true'] = y_test.values
X_test_with_predictions['y_pred'] = y_pred_test
X_test_with_predictions['correct'] = (y_test.values == y_pred_test)

# Falsos positivos y falsos negativos
false_positives = X_test_with_predictions[(X_test_with_predictions['y_true'] == 0) & 
                                         (X_test_with_predictions['y_pred'] == 1)]
false_negatives = X_test_with_predictions[(X_test_with_predictions['y_true'] == 1) & 
                                         (X_test_with_predictions['y_pred'] == 0)]

print("=== ANÁLISIS DE ERRORES ===")
print(f"Falsos Positivos: {len(false_positives)} ({len(false_positives)/len(X_test)*100:.2f}%)")
print(f"Falsos Negativos: {len(false_negatives)} ({len(false_negatives)/len(X_test)*100:.2f}%)")

# Analizar características de los errores
print("\nCaracterísticas promedio - Falsos Positivos vs Predicciones Correctas:")
numerical_features_in_test = ['Tenure', 'MonthlyCharges', 'TotalCharges']
for feature in numerical_features_in_test:
    if feature in X_test.columns:
        fp_mean = false_positives[feature].mean()
        correct_mean = X_test_with_predictions[X_test_with_predictions['correct']][feature].mean()
        print(f"{feature}: FP={fp_mean:.2f}, Correcto={correct_mean:.2f}")
```

## 5. Obtención de Insights y Orientación al Negocio (Post-Modelado)

### Objetivo de esta sección
Traducir los resultados del modelo en valor de negocio tangible y planificar su implementación y monitoreo continuo.

### 5.1 Interpretación de Resultados para el Negocio

```python
# Simulación de análisis de impacto de negocio
print("=== IMPACTO DE NEGOCIO ESTIMADO ===")

# Parámetros de negocio
avg_customer_lifetime_value = 3500  # USD
cost_retention_campaign = 50  # USD por cliente
churn_rate_without_intervention = 0.26  # 26%

# Cálculos de impacto
total_customers = 10000  # Ejemplo de base de clientes
predicted_churners = int(total_customers * final_metrics['recall'] * churn_rate_without_intervention)
retention_rate_with_intervention = 0.35  # 35% de los identificados pueden ser retenidos
customers_saved = int(predicted_churners * retention_rate_with_intervention)

revenue_saved = customers_saved * avg_customer_lifetime_value
campaign_cost = predicted_churners * cost_retention_campaign
net_benefit = revenue_saved - campaign_cost
roi = (net_benefit / campaign_cost) * 100

print(f"\nPara una base de {total_customers:,} clientes:")
print(f"- Clientes en riesgo identificados correctamente: {predicted_churners:,}")
print(f"- Clientes potencialmente salvados: {customers_saved:,}")
print(f"- Ingresos salvados: ${revenue_saved:,}")
print(f"- Costo de campañas de retención: ${campaign_cost:,}")
print(f"- Beneficio neto: ${net_benefit:,}")
print(f"- ROI: {roi:.1f}%")
```

### 5.2 Recomendaciones Accionables

```python
# Generar recomendaciones basadas en el análisis
print("=== RECOMENDACIONES ACCIONABLES ===")

# Basadas en la importancia de características
top_3_features = feature_importance.head(3)['feature'].tolist()

recommendations = {
    'Inmediatas': [
        'Implementar campañas de retención focalizadas en clientes con contratos mes a mes',
        'Ofrecer incentivos para migrar a contratos anuales o bianuales',
        'Mejorar la experiencia de soporte técnico para reducir frustración'
    ],
    'Mediano_plazo': [
        'Desarrollar programa de lealtad basado en tenure',
        'Revisar estructura de precios para clientes de alto valor',
        'Implementar sistema de alertas tempranas basado en el modelo'
    ],
    'Largo_plazo': [
        'Integrar el modelo con CRM para automatización',
        'Desarrollar modelos complementarios de upselling/cross-selling',
        'Establecer proceso de reentrenamiento trimestral'
    ]
}

for plazo, acciones in recommendations.items():
    print(f"\n{plazo.replace('_', ' ').title()}:")
    for i, accion in enumerate(acciones, 1):
        print(f"  {i}. {accion}")
```

### 5.3 Plan de Implementación y Monitoreo

```python
# Plan de implementación
implementation_plan = {
    'Fase 1: Piloto (Mes 1-2)': {
        'Acciones': [
            'Seleccionar grupo de control y experimental',
            'Implementar scoring diario de clientes',
            'Diseñar campañas de retención específicas'
        ],
        'KPIs': ['Tasa de retención', 'ROI de campaña', 'Satisfacción del cliente']
    },
    'Fase 2: Despliegue (Mes 3-4)': {
        'Acciones': [
            'Escalar a toda la base de clientes',
            'Integrar con sistemas de marketing automation',
            'Capacitar al equipo de customer success'
        ],
        'KPIs': ['Cobertura del modelo', 'Tiempo de respuesta', 'Adopción del equipo']
    },
    'Fase 3: Optimización (Mes 5+)': {
        'Acciones': [
            'Reentrenar modelo con nuevos datos',
            'Ajustar umbrales de decisión',
            'Expandir a otros productos/servicios'
        ],
        'KPIs': ['Drift del modelo', 'Mejora en métricas', 'Expansión del alcance']
    }
}

print("=== PLAN DE IMPLEMENTACIÓN ===")
for fase, detalles in implementation_plan.items():
    print(f"\n{fase}")
    print("  Acciones:")
    for accion in detalles['Acciones']:
        print(f"    • {accion}")
    print("  KPIs a monitorear:")
    for kpi in detalles['KPIs']:
        print(f"    • {kpi}")
```

### 5.4 Monitoreo y Alertas

```python
# Definición de sistema de monitoreo
monitoring_metrics = {
    'Métricas de Modelo': {
        'precision': {'umbral': 0.85, 'frecuencia': 'diaria'},
        'recall': {'umbral': 0.80, 'frecuencia': 'diaria'},
        'f1_score': {'umbral': 0.82, 'frecuencia': 'diaria'},
        'distribucion_predicciones': {'umbral': '±10% del baseline', 'frecuencia': 'semanal'}
    },
    'Métricas de Negocio': {
        'tasa_retencion': {'umbral': '15% mejora', 'frecuencia': 'semanal'},
        'costo_por_retencion': {'umbral': '$50', 'frecuencia': 'mensual'},
        'satisfaccion_cliente': {'umbral': '4.0/5.0', 'frecuencia': 'mensual'}
    },
    'Métricas de Datos': {
        'completitud_datos': {'umbral': '95%', 'frecuencia': 'diaria'},
        'drift_features': {'umbral': 'KS < 0.1', 'frecuencia': 'semanal'},
        'volumen_predicciones': {'umbral': '±20% esperado', 'frecuencia': 'diaria'}
    }
}

print("=== SISTEMA DE MONITOREO Y ALERTAS ===")
for categoria, metricas in monitoring_metrics.items():
    print(f"\n{categoria}:")
    for metrica, config in metricas.items():
        print(f"  • {metrica}: umbral={config['umbral']}, monitoreo {config['frecuencia']}")

# Simulación de dashboard de monitoreo
print("\n=== EJEMPLO DE DASHBOARD DE MONITOREO (Semana Actual) ===")
dashboard_data = {
    'Precisión del Modelo': {'actual': 0.87, 'objetivo': 0.85, 'status': '✅'},
    'Recall del Modelo': {'actual': 0.79, 'objetivo': 0.80, 'status': '⚠️'},
    'Clientes en Riesgo Identificados': {'actual': 312, 'objetivo': 'N/A', 'status': '📊'},
    'Campañas Ejecutadas': {'actual': 287, 'objetivo': 312, 'status': '🔄'},
    'Tasa de Retención': {'actual': '32%', 'objetivo': '30%', 'status': '✅'},
    'ROI de Campaña': {'actual': '285%', 'objetivo': '300%', 'status': '⚠️'}
}

for metrica, valores in dashboard_data.items():
    print(f"{valores['status']} {metrica}: {valores['actual']} (objetivo: {valores['objetivo']})")
```

### 5.5 Consideraciones Éticas y de Privacidad

```python
# Framework de consideraciones éticas
ethical_framework = {
    'Transparencia': {
        'Principio': 'Los clientes deben poder entender por qué fueron clasificados como en riesgo',
        'Implementación': [
            'Proporcionar explicaciones claras en las comunicaciones',
            'Permitir opt-out de campañas automatizadas',
            'Documentar la lógica del modelo para auditorías'
        ]
    },
    'Equidad': {
        'Principio': 'El modelo no debe discriminar por características protegidas',
        'Implementación': [
            'Análisis regular de sesgo por género, edad, ubicación',
            'Ajustes para garantizar equidad en las predicciones',
            'Revisión humana de casos límite'
        ]
    },
    'Privacidad': {
        'Principio': 'Proteger la información personal de los clientes',
        'Implementación': [
            'Cumplimiento con GDPR/LGPD',
            'Anonimización de datos en reportes',
            'Acceso restringido a predicciones individuales'
        ]
    },
    'Beneficencia': {
        'Principio': 'Las acciones deben beneficiar tanto al cliente como a la empresa',
        'Implementación': [
            'Ofertas personalizadas que agreguen valor real',
            'No penalizar a clientes identificados como en riesgo',
            'Mejorar servicios basándose en insights del modelo'
        ]
    }
}

print("=== CONSIDERACIONES ÉTICAS Y DE PRIVACIDAD ===")
for aspecto, detalles in ethical_framework.items():
    print(f"\n{aspecto}:")
    print(f"  Principio: {detalles['Principio']}")
    print("  Medidas de implementación:")
    for medida in detalles['Implementación']:
        print(f"    • {medida}")
```

## 6. Conclusión y Próximos Pasos (Implementación y MLOps)

### Objetivo de esta sección
Resumir el proceso y destacar la importancia de la implementación continua y las prácticas de MLOps para el éxito a largo plazo.

### 6.1 Resumen del Proyecto

```python
print("=== RESUMEN EJECUTIVO DEL PROYECTO ===")

project_summary = {
    'Problema': 'Alta tasa de abandono de clientes (26%) en telecomunicaciones',
    'Solución': 'Modelo predictivo de ML para identificar clientes en riesgo',
    'Resultado': f"F1-Score de {final_metrics['f1_score']:.3f} con ROI estimado de {roi:.0f}%",
    'Impacto': f"Potencial para salvar {customers_saved} clientes anuales",
    'Inversión': 'Implementación en 3-4 meses con equipo dedicado',
    'Riesgos': 'Degradación del modelo, resistencia al cambio, calidad de datos'
}

for key, value in project_summary.items():
    print(f"{key}: {value}")
```

### 6.2 Arquitectura MLOps Propuesta

```python
# Componentes de la arquitectura MLOps
mlops_architecture = {
    'Data Pipeline': {
        'Componentes': ['Data Lake', 'ETL automatizado', 'Validación de calidad'],
        'Herramientas': ['Apache Airflow', 'dbt', 'Great Expectations'],
        'Frecuencia': 'Diaria'
    },
    'Model Training': {
        'Componentes': ['Experimentación', 'Entrenamiento automatizado', 'Registro de modelos'],
        'Herramientas': ['MLflow', 'Kubeflow', 'DVC'],
        'Frecuencia': 'Mensual/Trimestral'
    },
    'Model Serving': {
        'Componentes': ['API REST', 'Batch scoring', 'Edge deployment'],
        'Herramientas': ['FastAPI', 'Kubernetes', 'Seldon'],
        'Frecuencia': 'Tiempo real / Batch diario'
    },
    'Monitoring': {
        'Componentes': ['Métricas de rendimiento', 'Data drift', 'Alertas'],
        'Herramientas': ['Prometheus', 'Grafana', 'Evidently AI'],
        'Frecuencia': 'Continua'
    }
}

print("=== ARQUITECTURA MLOPS PROPUESTA ===")
for componente, detalles in mlops_architecture.items():
    print(f"\n{componente}:")
    print(f"  Componentes: {', '.join(detalles['Componentes'])}")
    print(f"  Herramientas sugeridas: {', '.join(detalles['Herramientas'])}")
    print(f"  Frecuencia: {detalles['Frecuencia']}")
```

### 6.3 Pipeline CI/CD/CT

```python
# Definición del pipeline CI/CD/CT
pipeline_stages = {
    'Continuous Integration (CI)': [
        'Validación de código Python (linting, type checking)',
        'Pruebas unitarias de funciones de preprocesamiento',
        'Validación de esquemas de datos',
        'Construcción de imágenes Docker'
    ],
    'Continuous Deployment (CD)': [
        'Despliegue en ambiente de staging',
        'Pruebas de integración con sistemas existentes',
        'Validación de performance (latencia, throughput)',
        'Despliegue gradual en producción (canary/blue-green)'
    ],
    'Continuous Training (CT)': [
        'Monitoreo de drift en datos y modelo',
        'Trigger automático de reentrenamiento',
        'Validación de nuevo modelo vs baseline',
        'Actualización automática si mejora métricas'
    ]
}

print("=== PIPELINE CI/CD/CT ===")
for stage, steps in pipeline_stages.items():
    print(f"\n{stage}:")
    for step in steps:
        print(f"  ✓ {step}")
```

### 6.4 Próximos Pasos Inmediatos

```python
# Plan de acción para las próximas semanas
action_plan = {
    'Semana 1-2': {
        'Objetivo': 'Preparación de infraestructura',
        'Tareas': [
            'Configurar ambiente de desarrollo MLOps',
            'Establecer pipelines de datos',
            'Definir APIs de scoring',
            'Crear dashboards de monitoreo'
        ],
        'Entregables': ['Ambiente configurado', 'Pipeline básico funcionando']
    },
    'Semana 3-4': {
        'Objetivo': 'Integración con sistemas',
        'Tareas': [
            'Conectar con CRM existente',
            'Implementar lógica de negocio',
            'Configurar alertas y notificaciones',
            'Capacitar al equipo de customer success'
        ],
        'Entregables': ['Integración completa', 'Equipo capacitado']
    },
    'Semana 5-6': {
        'Objetivo': 'Piloto controlado',
        'Tareas': [
            'Seleccionar grupo de prueba (5% clientes)',
            'Ejecutar campañas de retención',
            'Monitorear métricas clave',
            'Recopilar feedback'
        ],
        'Entregables': ['Resultados del piloto', 'Plan de mejoras']
    },
    'Semana 7-8': {
        'Objetivo': 'Escalamiento',
        'Tareas': [
            'Ajustar basándose en resultados del piloto',
            'Escalar gradualmente a toda la base',
            'Optimizar performance',
            'Documentar procesos'
        ],
        'Entregables': ['Sistema en producción', 'Documentación completa']
    }
}

print("=== PLAN DE ACCIÓN - PRÓXIMAS 8 SEMANAS ===")
for periodo, detalles in action_plan.items():
    print(f"\n{periodo}: {detalles['Objetivo']}")
    print("  Tareas principales:")
    for tarea in detalles['Tareas']:
        print(f"    • {tarea}")
    print(f"  Entregables: {', '.join(detalles['Entregables'])}")
```

### 6.5 Lecciones Clave y Mejores Prácticas

```python
# Resumen de mejores prácticas aprendidas
best_practices = {
    'Datos': [
        'La calidad de datos es más importante que algoritmos sofisticados',
        'Invertir tiempo en feature engineering da grandes retornos',
        'Mantener pipeline de datos robusto y monitoreado'
    ],
    'Modelado': [
        'Comenzar simple y agregar complejidad gradualmente',
        'Validar con métricas de negocio, no solo técnicas',
        'Considerar interpretabilidad vs performance'
    ],
    'Implementación': [
        'MLOps no es opcional para proyectos en producción',
        'Monitoreo continuo es crítico para mantener performance',
        'Involucrar stakeholders desde el inicio'
    ],
    'Organización': [
        'Fomentar colaboración entre data scientists y engineers',
        'Documentar decisiones y supuestos',
        'Establecer procesos de gobierno de modelos'
    ]
}

print("=== LECCIONES CLAVE Y MEJORES PRÁCTICAS ===")
for categoria, practicas in best_practices.items():
    print(f"\n{categoria}:")
    for practica in practicas:
        print(f"  ★ {practica}")

# Mensaje final
print("\n" + "="*60)
print("🎯 CONCLUSIÓN FINAL")
print("="*60)
print("""
Este proyecto demuestra el ciclo completo de un proyecto de Machine Learning,
desde la comprensión del problema de negocio hasta la implementación en producción.

El éxito no termina con un modelo preciso - requiere:
- Integración continua con sistemas empresariales
- Monitoreo y mantenimiento constantes
- Evolución basada en feedback y cambios del negocio
- Compromiso organizacional con la cultura data-driven

El verdadero valor del ML se materializa cuando los modelos se convierten en
sistemas productivos que mejoran continuamente las decisiones empresariales.
""")

print("\n¡Éxito en tu proyecto de Machine Learning! 🚀")
```