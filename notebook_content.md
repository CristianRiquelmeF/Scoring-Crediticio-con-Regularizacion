**"Modelo Predictivo de Riesgo Crediticio con Técnicas Avanzadas de Machine Learning"**


> **Objetivo**
Utilizando un dataset de historial crediticio, este proyecto busca predecir el riesgo crediticio de clientes (clasificación binaria: alto riesgo vs. bajo riesgo).

**Metodología**

**1.Modelos de Ensamble:**
  
  - Implementación de técnicas que combinan múltiples modelos débiles (principalmente árboles de decisión) para mejorar la **precisión** y **robustez**.

  - Algoritmos incluidos:

    - Random Forest (bagging) con optimización Grid, Search y Bayesiana.

    - XGBoost y AdaBoost (boosting).

**2.Regularización:**

   - Aplicación de métodos como Lasso (L1) y Ridge (L2) para evaluar su impacto en la generalización del modelo.

**3.Evaluación y Comparación:**

   - Análisis de métricas clave: *Accuracy, Precision, Recall, F1-Score* y AUC-ROC.

   - Interpretación de la importancia de características para identificar predictores críticos.

   - Selección del mejor modelo basado en rendimiento y capacidad explicativa.

**Resultados Esperados**

   - Identificación del algoritmo óptimo para predecir riesgo crediticio.

   - Explicación clara de los factores que influyen en la clasificación.

   - Documentación técnica que justifique la elección del modelo final.


```python
import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, roc_curve, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
```


```python
pip install scikit-optimize
```


```python
from skopt import BayesSearchCV
```

## **Carga dataset y revisión preliminar**


```python
# Cargar el dataset "credit"
df = fetch_openml("credit", version=1)
X = df.data
y = df.target
```


```python
# Convertir a DataFrame para exploración
df = pd.DataFrame(X, columns=df.feature_names)
df['riesgo'] = y

# Ver las primeras filas
df.head()
```





  <div id="df-933e4221-222f-4d8b-a85e-6eca1f544223" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RevolvingUtilizationOfUnsecuredLines</th>
      <th>age</th>
      <th>NumberOfTime30-59DaysPastDueNotWorse</th>
      <th>DebtRatio</th>
      <th>MonthlyIncome</th>
      <th>NumberOfOpenCreditLinesAndLoans</th>
      <th>NumberOfTimes90DaysLate</th>
      <th>NumberRealEstateLoansOrLines</th>
      <th>NumberOfTime60-89DaysPastDueNotWorse</th>
      <th>NumberOfDependents</th>
      <th>riesgo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.006999</td>
      <td>38.0</td>
      <td>0.0</td>
      <td>0.302150</td>
      <td>5440.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.704592</td>
      <td>63.0</td>
      <td>0.0</td>
      <td>0.471441</td>
      <td>8000.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.063113</td>
      <td>57.0</td>
      <td>0.0</td>
      <td>0.068586</td>
      <td>5000.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.368397</td>
      <td>68.0</td>
      <td>0.0</td>
      <td>0.296273</td>
      <td>6250.0</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.000000</td>
      <td>34.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>3500.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-933e4221-222f-4d8b-a85e-6eca1f544223')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-933e4221-222f-4d8b-a85e-6eca1f544223 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-933e4221-222f-4d8b-a85e-6eca1f544223');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-6d34e2bf-d648-46f1-af5f-84243102e2ce">
      <button class="colab-df-quickchart" onclick="quickchart('df-6d34e2bf-d648-46f1-af5f-84243102e2ce')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-6d34e2bf-d648-46f1-af5f-84243102e2ce button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
print("Cantidad de filas:")
display(df.shape[0])
print("")
print("Número de columnas:")
display(df.shape[1])
```

    Cantidad de filas:



    16714


    
    Número de columnas:



    11



```python
# Revisión del contenido
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 16714 entries, 0 to 16713
    Data columns (total 11 columns):
     #   Column                                Non-Null Count  Dtype  
    ---  ------                                --------------  -----  
     0   RevolvingUtilizationOfUnsecuredLines  16714 non-null  float64
     1   age                                   16714 non-null  float64
     2   NumberOfTime30-59DaysPastDueNotWorse  16714 non-null  float64
     3   DebtRatio                             16714 non-null  float64
     4   MonthlyIncome                         16714 non-null  float64
     5   NumberOfOpenCreditLinesAndLoans       16714 non-null  float64
     6   NumberOfTimes90DaysLate               16714 non-null  float64
     7   NumberRealEstateLoansOrLines          16714 non-null  float64
     8   NumberOfTime60-89DaysPastDueNotWorse  16714 non-null  float64
     9   NumberOfDependents                    16714 non-null  float64
     10  riesgo                                16714 non-null  float64
    dtypes: float64(11)
    memory usage: 1.4 MB


| Nº | Columna                                | Tipo      | Descripción (estimada por nombre)                     |
| -- | -------------------------------------- | --------- | ----------------------------------------------------- |
| 0  | `RevolvingUtilizationOfUnsecuredLines` | `float64` | Proporción de crédito usado vs. disponible (0–1)      |
| 1  | `age`                                  | `float64` | Edad del solicitante                                  |
| 2  | `NumberOfTime30-59DaysPastDueNotWorse` | `float64` | Número de veces con mora de 30-59 días                |
| 3  | `DebtRatio`                            | `float64` | Proporción deuda / ingresos                           |
| 4  | `MonthlyIncome`                        | `float64` | Ingreso mensual (puede tener outliers)                |
| 5  | `NumberOfOpenCreditLinesAndLoans`      | `float64` | Cantidad de créditos/líneas activas                   |
| 6  | `NumberOfTimes90DaysLate`              | `float64` | Número de veces con mora de 90+ días                  |
| 7  | `NumberRealEstateLoansOrLines`         | `float64` | Cantidad de hipotecas u otros préstamos inmobiliarios |
| 8  | `NumberOfTime60-89DaysPastDueNotWorse` | `float64` | Número de veces con mora de 60-89 días                |
| 9  | `NumberOfDependents`                   | `float64` | Número de dependientes económicos                     |
| 10 | `riesgo`                             | `float64` | **Variable objetivo**: 1 si , 0  no     |


- En este dataset la variable objetivo 'riesgo', representa riesgo crediticio del cliente: 1 = Sí, 0 = No


```python
# Conteo de clases en variable objetivo
df['riesgo'].value_counts()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>riesgo</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>8357</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>8357</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>



**Variables clave:**

- Comportamiento de pagos (NumberOfTime30-59DaysPastDueNotWorse, NumberOfTimes90DaysLate)

- Situación financiera (DebtRatio, MonthlyIncome)

- Historial crediticio (NumberOfOpenCreditLinesAndLoans, NumberRealEstateLoansOrLines)


```python
df.describe()
```





  <div id="df-faeef5f2-1662-4c4d-b45b-63800d462641" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RevolvingUtilizationOfUnsecuredLines</th>
      <th>age</th>
      <th>NumberOfTime30-59DaysPastDueNotWorse</th>
      <th>DebtRatio</th>
      <th>MonthlyIncome</th>
      <th>NumberOfOpenCreditLinesAndLoans</th>
      <th>NumberOfTimes90DaysLate</th>
      <th>NumberRealEstateLoansOrLines</th>
      <th>NumberOfTime60-89DaysPastDueNotWorse</th>
      <th>NumberOfDependents</th>
      <th>riesgo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>16714.000000</td>
      <td>16714.000000</td>
      <td>16714.000000</td>
      <td>16714.000000</td>
      <td>16714.000000</td>
      <td>16714.000000</td>
      <td>16714.000000</td>
      <td>16714.000000</td>
      <td>16714.000000</td>
      <td>16714.000000</td>
      <td>16714.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.799862</td>
      <td>48.798672</td>
      <td>1.110267</td>
      <td>30.980298</td>
      <td>6118.120258</td>
      <td>8.503709</td>
      <td>0.863827</td>
      <td>1.047445</td>
      <td>0.734354</td>
      <td>0.944358</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>204.062345</td>
      <td>13.906078</td>
      <td>7.172890</td>
      <td>719.694859</td>
      <td>5931.841779</td>
      <td>5.370965</td>
      <td>7.167576</td>
      <td>1.272565</td>
      <td>7.138737</td>
      <td>1.198791</td>
      <td>0.500015</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.082397</td>
      <td>38.000000</td>
      <td>0.000000</td>
      <td>0.155971</td>
      <td>3128.500000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.443080</td>
      <td>48.000000</td>
      <td>0.000000</td>
      <td>0.322299</td>
      <td>5000.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.926637</td>
      <td>58.000000</td>
      <td>1.000000</td>
      <td>0.533426</td>
      <td>7573.000000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>22000.000000</td>
      <td>101.000000</td>
      <td>98.000000</td>
      <td>61106.500000</td>
      <td>250000.000000</td>
      <td>57.000000</td>
      <td>98.000000</td>
      <td>29.000000</td>
      <td>98.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-faeef5f2-1662-4c4d-b45b-63800d462641')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-faeef5f2-1662-4c4d-b45b-63800d462641 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-faeef5f2-1662-4c4d-b45b-63800d462641');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-c30d83a5-02b5-4d6a-9076-6efa016c7765">
      <button class="colab-df-quickchart" onclick="quickchart('df-c30d83a5-02b5-4d6a-9076-6efa016c7765')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-c30d83a5-02b5-4d6a-9076-6efa016c7765 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
# Observación de valores nulos
df.isnull().sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>RevolvingUtilizationOfUnsecuredLines</th>
      <td>0</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0</td>
    </tr>
    <tr>
      <th>NumberOfTime30-59DaysPastDueNotWorse</th>
      <td>0</td>
    </tr>
    <tr>
      <th>DebtRatio</th>
      <td>0</td>
    </tr>
    <tr>
      <th>MonthlyIncome</th>
      <td>0</td>
    </tr>
    <tr>
      <th>NumberOfOpenCreditLinesAndLoans</th>
      <td>0</td>
    </tr>
    <tr>
      <th>NumberOfTimes90DaysLate</th>
      <td>0</td>
    </tr>
    <tr>
      <th>NumberRealEstateLoansOrLines</th>
      <td>0</td>
    </tr>
    <tr>
      <th>NumberOfTime60-89DaysPastDueNotWorse</th>
      <td>0</td>
    </tr>
    <tr>
      <th>NumberOfDependents</th>
      <td>0</td>
    </tr>
    <tr>
      <th>riesgo</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
# Revisión valores duplicados
df.duplicated().sum()
```




    np.int64(2)




```python
# Revisión de filas duplicadas
duplicated_rows = df[df.duplicated(keep=False)]  # `keep=False` marca TODOS los duplicados (incluyendo el primero)
duplicated_rows.sort_values(by=list(df.columns))  # Ordena para verlos agrupados
```





  <div id="df-9d54e417-7eec-4eaf-954f-021c7f2e9a31" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RevolvingUtilizationOfUnsecuredLines</th>
      <th>age</th>
      <th>NumberOfTime30-59DaysPastDueNotWorse</th>
      <th>DebtRatio</th>
      <th>MonthlyIncome</th>
      <th>NumberOfOpenCreditLinesAndLoans</th>
      <th>NumberOfTimes90DaysLate</th>
      <th>NumberRealEstateLoansOrLines</th>
      <th>NumberOfTime60-89DaysPastDueNotWorse</th>
      <th>NumberOfDependents</th>
      <th>riesgo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10133</th>
      <td>1.0</td>
      <td>22.0</td>
      <td>98.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>98.0</td>
      <td>0.0</td>
      <td>98.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11054</th>
      <td>1.0</td>
      <td>22.0</td>
      <td>98.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>98.0</td>
      <td>0.0</td>
      <td>98.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11673</th>
      <td>1.0</td>
      <td>55.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12257</th>
      <td>1.0</td>
      <td>55.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9d54e417-7eec-4eaf-954f-021c7f2e9a31')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-9d54e417-7eec-4eaf-954f-021c7f2e9a31 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9d54e417-7eec-4eaf-954f-021c7f2e9a31');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-f2f537a2-77bd-4e92-8518-fdbbe562b562">
      <button class="colab-df-quickchart" onclick="quickchart('df-f2f537a2-77bd-4e92-8518-fdbbe562b562')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-f2f537a2-77bd-4e92-8518-fdbbe562b562 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
# Gráfica de la distribución edades
plt.figure(figsize=(8,4))
x = df['age']
plt.title('Edad de clientes', fontsize=15)
ax = sns.distplot(x, color='teal', rug=True)
plt.xlabel('Edad')
```

    /tmp/ipython-input-1291960022.py:5: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      ax = sns.distplot(x, color='teal', rug=True)





    Text(0.5, 0, 'Edad')




    
![png](output_18_2.png)
    



```python
# Correlación con el target
corr_with_target = df.corr()['riesgo'].sort_values(ascending=False)
plt.figure(figsize=(8, 4))
sns.barplot(x=corr_with_target.index, y=corr_with_target.values)
plt.xticks(rotation=90)
plt.title("Correlación con 'riesgo'")
plt.show()
```


    
![png](output_19_0.png)
    



```python
# Outliers en variables clave
outlier_vars = [
    'RevolvingUtilizationOfUnsecuredLines',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfTimes90DaysLate'
]

plt.figure(figsize=(10, 6))
for i, col in enumerate(outlier_vars, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df, y=col)
    plt.title(f"Boxplot de {col}")
plt.tight_layout()
plt.show()
```


    
![png](output_20_0.png)
    


## **Preprocesamiento**


```python
# Convertir de float a entero para evitar problemas en modelos de clasificación en ML.
y = y.astype(int)
```


```python
# Copia de X para aplicar tratamiento de outliers
X_pre = X.copy()
```


```python
# Tratamiento de outliers antes de escalar, afectan desempeño en regresión logística.
cols_to_clip = ['DebtRatio', 'RevolvingUtilizationOfUnsecuredLines', 'MonthlyIncome', 'NumberOfTimes90DaysLate']
X_pre[cols_to_clip] = X_pre[cols_to_clip].apply(lambda x: x.clip(upper=x.quantile(0.99)))
```


```python
# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X_pre, y, test_size=0.2, random_state=42, stratify=y)
```


```python
# Escalar datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## **Entrenamiento de modelos**


```python
# Función de evaluación
def evaluar_modelo(nombre, modelo):
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    print(f"\n--- {nombre} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-Score:", f1_score(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_proba))
    print("\n", classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_estimator(modelo, X_test, y_test, cmap='Blues')
    plt.title(f"Matriz de Confusión - {nombre}")
    plt.show()
```

**Random Forest**


```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
evaluar_modelo("Random Forest", rf)
```

    
    --- Random Forest ---
    Accuracy: 0.7759497457373616
    Precision: 0.7744047619047619
    Recall: 0.7785757031717534
    F1-Score: 0.7764846314532975
    AUC-ROC: 0.8521247269062161
    
                   precision    recall  f1-score   support
    
               0       0.78      0.77      0.78      1672
               1       0.77      0.78      0.78      1671
    
        accuracy                           0.78      3343
       macro avg       0.78      0.78      0.78      3343
    weighted avg       0.78      0.78      0.78      3343
    



    
![png](output_30_1.png)
    


**Random Forest con Grid Search**


```python
# Definir los parámetros para Grid Search
param_grid = {
    'n_estimators': [80, 100, 120],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Configurar Grid Search
grid_search = GridSearchCV(estimator=rf,
                          param_grid=param_grid,
                          cv=3,
                          n_jobs=-1,
                          verbose=2,
                          scoring='accuracy')
# Ejecutar Grid Search
grid_search.fit(X_train, y_train)
print("Mejores hiperparámetros (Grid Search):", grid_search.best_params_)
best_grid = grid_search.best_estimator_

evaluar_modelo("Random Forest con Grid Search", best_grid)
```

    Fitting 3 folds for each of 36 candidates, totalling 108 fits
    Mejores hiperparámetros (Grid Search): {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 120}
    
    --- Random Forest con Grid Search ---
    Accuracy: 0.7813341310200419
    Precision: 0.7841596130592503
    Recall: 0.7761819269898265
    F1-Score: 0.7801503759398496
    AUC-ROC: 0.8618778973711413
    
                   precision    recall  f1-score   support
    
               0       0.78      0.79      0.78      1672
               1       0.78      0.78      0.78      1671
    
        accuracy                           0.78      3343
       macro avg       0.78      0.78      0.78      3343
    weighted avg       0.78      0.78      0.78      3343
    



    
![png](output_32_1.png)
    


**Random Forest con Random Search**


```python
# Definir la distribución de parámetros para Random Search
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
}

# Configurar Random Search
random_search = RandomizedSearchCV(estimator=rf,
                                 param_distributions=param_dist,
                                 n_iter=30,  # Número de combinaciones a probar
                                 cv=3,
                                 n_jobs=-1,
                                 verbose=2,
                                 random_state=42,
                                 scoring='accuracy')

# Ejecutar Random Search
random_search.fit(X_train, y_train)
print("Mejores parámetros (Random Search):", random_search.best_params_)
best_random = random_search.best_estimator_

evaluar_modelo("Random Forest con Random Search", best_random)
```

    Fitting 3 folds for each of 30 candidates, totalling 90 fits
    Mejores parámetros (Random Search): {'max_depth': 10, 'min_samples_leaf': 9, 'min_samples_split': 9, 'n_estimators': 61}
    
    --- Random Forest con Random Search ---
    Accuracy: 0.7816332635357464
    Precision: 0.783946891973446
    Recall: 0.77737881508079
    F1-Score: 0.7806490384615384
    AUC-ROC: 0.8615296401604633
    
                   precision    recall  f1-score   support
    
               0       0.78      0.79      0.78      1672
               1       0.78      0.78      0.78      1671
    
        accuracy                           0.78      3343
       macro avg       0.78      0.78      0.78      3343
    weighted avg       0.78      0.78      0.78      3343
    



    
![png](output_34_1.png)
    


**Random Forest con Optimización Bayesiana - Scikit Optimize**


```python
# Definición de espacio de búsqueda
searh_space = {'n_estimators': (50, 200),
               'max_depth': (3, 20),
               'min_samples_split': (2, 10)}

# Configurar el modelo
opt= BayesSearchCV(estimator= rf,
                   search_spaces=searh_space,
                   n_iter=30,
                   cv=3,
                   scoring='f1',
                   n_jobs=-1,
                   random_state=42)

opt.fit(X_train, y_train)
evaluar_modelo("Random Forest con Optimización Bayesiana", opt)
```

    
    --- Random Forest con Optimización Bayesiana ---
    Accuracy: 0.7792402034101107
    Precision: 0.7811934900542495
    Recall: 0.7755834829443446
    F1-Score: 0.7783783783783784
    AUC-ROC: 0.8610378565967718
    
                   precision    recall  f1-score   support
    
               0       0.78      0.78      0.78      1672
               1       0.78      0.78      0.78      1671
    
        accuracy                           0.78      3343
       macro avg       0.78      0.78      0.78      3343
    weighted avg       0.78      0.78      0.78      3343
    



    
![png](output_36_1.png)
    


**XGBoost**


```python
#Crear y entrenar modelo XGBoost
xg = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss')

xg.fit(X_train, y_train)
evaluar_modelo("XGBoost", xg)
```

    
    --- XGBoost ---
    Accuracy: 0.780735865988633
    Precision: 0.7923940149625935
    Recall: 0.760622381807301
    F1-Score: 0.7761832061068702
    AUC-ROC: 0.8600324562835194
    
                   precision    recall  f1-score   support
    
               0       0.77      0.80      0.79      1672
               1       0.79      0.76      0.78      1671
    
        accuracy                           0.78      3343
       macro avg       0.78      0.78      0.78      3343
    weighted avg       0.78      0.78      0.78      3343
    



    
![png](output_38_1.png)
    


**AdaBoost**


```python
# Crear y entrenar modelo AdaBoost
ada = AdaBoostClassifier(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42)

ada.fit(X_train, y_train)
evaluar_modelo("AdaBoost", ada)
```

    
    --- AdaBoost ---
    Accuracy: 0.7690696978761591
    Precision: 0.7843137254901961
    Recall: 0.7420706163973668
    F1-Score: 0.7626076260762608
    AUC-ROC: 0.8496452286256689
    
                   precision    recall  f1-score   support
    
               0       0.76      0.80      0.78      1672
               1       0.78      0.74      0.76      1671
    
        accuracy                           0.77      3343
       macro avg       0.77      0.77      0.77      3343
    weighted avg       0.77      0.77      0.77      3343
    



    
![png](output_40_1.png)
    


**Regresión logística sin regularización**


```python
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(X_train, y_train)
evaluar_modelo("Logistic Regression (sin regularización)", logreg)
```

    
    --- Logistic Regression (sin regularización) ---
    Accuracy: 0.759796589889321
    Precision: 0.7699004975124378
    Recall: 0.7408737283064033
    F1-Score: 0.755108264714852
    AUC-ROC: 0.8423407752284252
    
                   precision    recall  f1-score   support
    
               0       0.75      0.78      0.76      1672
               1       0.77      0.74      0.76      1671
    
        accuracy                           0.76      3343
       macro avg       0.76      0.76      0.76      3343
    weighted avg       0.76      0.76      0.76      3343
    



    
![png](output_42_1.png)
    


**Regresión Lasso (L1)**


```python
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
lasso.fit(X_train, y_train)
evaluar_modelo("Logistic Regression (Lasso - L1)", lasso)
```

    
    --- Logistic Regression (Lasso - L1) ---
    Accuracy: 0.7606939874364343
    Precision: 0.7713395638629283
    Recall: 0.7408737283064033
    F1-Score: 0.7557997557997558
    AUC-ROC: 0.8425741397724766
    
                   precision    recall  f1-score   support
    
               0       0.75      0.78      0.77      1672
               1       0.77      0.74      0.76      1671
    
        accuracy                           0.76      3343
       macro avg       0.76      0.76      0.76      3343
    weighted avg       0.76      0.76      0.76      3343
    



    
![png](output_44_1.png)
    


**Regresión Ridge (L2)**


```python
ridge = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000)
ridge.fit(X_train, y_train)
evaluar_modelo("Logistic Regression (Ridge - L2)", ridge)
```

    
    --- Logistic Regression (Ridge - L2) ---
    Accuracy: 0.759796589889321
    Precision: 0.7699004975124378
    Recall: 0.7408737283064033
    F1-Score: 0.755108264714852
    AUC-ROC: 0.8423407752284252
    
                   precision    recall  f1-score   support
    
               0       0.75      0.78      0.76      1672
               1       0.77      0.74      0.76      1671
    
        accuracy                           0.76      3343
       macro avg       0.76      0.76      0.76      3343
    weighted avg       0.76      0.76      0.76      3343
    



    
![png](output_46_1.png)
    


**Regresión Elastic Net**


```python
elastic = LogisticRegression(penalty='elasticnet', solver='saga',
                                   l1_ratio=0.5, max_iter=1000, random_state=42)
elastic.fit(X_train, y_train)
evaluar_modelo("Logistic Regression (Elastic Net)", elastic)
```

    
    --- Logistic Regression (Elastic Net) ---
    Accuracy: 0.759796589889321
    Precision: 0.7699004975124378
    Recall: 0.7408737283064033
    F1-Score: 0.755108264714852
    AUC-ROC: 0.8425000501089512
    
                   precision    recall  f1-score   support
    
               0       0.75      0.78      0.76      1672
               1       0.77      0.74      0.76      1671
    
        accuracy                           0.76      3343
       macro avg       0.76      0.76      0.76      3343
    weighted avg       0.76      0.76      0.76      3343
    



    
![png](output_48_1.png)
    


## **Gráficos que muestran la importancia de las características y cómo impactan en las predicciones**


```python
# Función para graficar la importancia de las caracteríscas por modelo
def ft_importance(model, X, top_n=None, palette='viridis', title=None):
    # Verificar que el modelo tenga el atributo
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("El modelo no tiene el atributo 'feature_importances_'.")
    # Obtener importancias
    importancias = model.feature_importances_
    features = X.columns
    # Crear DataFrame ordenado
    df_importancias = pd.DataFrame({
        'Feature': features,
        'Importancia': importancias
    }).sort_values(by='Importancia', ascending=False)

    # Seleccionar top_n si corresponde
    if top_n is not None:
        df_importancias = df_importancias.head(top_n)

    plt.figure(figsize=(10, max(4, len(df_importancias) * 0.4)))
    sns.barplot(data=df_importancias, x='Importancia', y='Feature', palette=palette)
    plt.title(title if title else 'Importancia de Características')
    plt.xlabel('Importancia')
    plt.ylabel('Características')
    plt.tight_layout()
    plt.show()

    return df_importancias
```


```python
ft_importance(rf, X, top_n=10, title="Importancia de Características - Random Forest")
```

    /tmp/ipython-input-2216649977.py:19: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(data=df_importancias, x='Importancia', y='Feature', palette=palette)



    
![png](output_51_1.png)
    






  <div id="df-5bd3ee5d-1652-449b-ab12-7d4cb1fca1d6" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Importancia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RevolvingUtilizationOfUnsecuredLines</td>
      <td>0.253268</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DebtRatio</td>
      <td>0.137151</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MonthlyIncome</td>
      <td>0.128407</td>
    </tr>
    <tr>
      <th>1</th>
      <td>age</td>
      <td>0.112569</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NumberOfTime30-59DaysPastDueNotWorse</td>
      <td>0.085389</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NumberOfOpenCreditLinesAndLoans</td>
      <td>0.083725</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NumberOfTimes90DaysLate</td>
      <td>0.081617</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NumberOfTime60-89DaysPastDueNotWorse</td>
      <td>0.045401</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NumberOfDependents</td>
      <td>0.036976</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NumberRealEstateLoansOrLines</td>
      <td>0.035496</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5bd3ee5d-1652-449b-ab12-7d4cb1fca1d6')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5bd3ee5d-1652-449b-ab12-7d4cb1fca1d6 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5bd3ee5d-1652-449b-ab12-7d4cb1fca1d6');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-d2685778-b388-47d0-a8cd-57f3aa32282c">
      <button class="colab-df-quickchart" onclick="quickchart('df-d2685778-b388-47d0-a8cd-57f3aa32282c')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-d2685778-b388-47d0-a8cd-57f3aa32282c button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
ft_importance(xg, X, top_n=10, title="Importancia de Características - XGBoost")
```

    /tmp/ipython-input-2216649977.py:19: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(data=df_importancias, x='Importancia', y='Feature', palette=palette)



    
![png](output_52_1.png)
    






  <div id="df-7e8e2749-5c50-4d87-97ab-38e0f8e6f6a8" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Importancia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>NumberOfTimes90DaysLate</td>
      <td>0.284518</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NumberOfTime30-59DaysPastDueNotWorse</td>
      <td>0.219938</td>
    </tr>
    <tr>
      <th>0</th>
      <td>RevolvingUtilizationOfUnsecuredLines</td>
      <td>0.152697</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NumberOfTime60-89DaysPastDueNotWorse</td>
      <td>0.145735</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NumberRealEstateLoansOrLines</td>
      <td>0.042343</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DebtRatio</td>
      <td>0.034412</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NumberOfOpenCreditLinesAndLoans</td>
      <td>0.033508</td>
    </tr>
    <tr>
      <th>1</th>
      <td>age</td>
      <td>0.032597</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MonthlyIncome</td>
      <td>0.030661</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NumberOfDependents</td>
      <td>0.023591</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7e8e2749-5c50-4d87-97ab-38e0f8e6f6a8')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7e8e2749-5c50-4d87-97ab-38e0f8e6f6a8 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7e8e2749-5c50-4d87-97ab-38e0f8e6f6a8');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-f82f3595-2c78-4c6a-9e18-ab303b30df5b">
      <button class="colab-df-quickchart" onclick="quickchart('df-f82f3595-2c78-4c6a-9e18-ab303b30df5b')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-f82f3595-2c78-4c6a-9e18-ab303b30df5b button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
ft_importance(ada, X, top_n=10, title="Importancia de Características - AdaBoost")
```

    /tmp/ipython-input-2216649977.py:19: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(data=df_importancias, x='Importancia', y='Feature', palette=palette)



    
![png](output_53_1.png)
    






  <div id="df-48f0f7d6-8e75-4154-a7a0-2f7348f7dd75" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Importancia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>NumberOfTime60-89DaysPastDueNotWorse</td>
      <td>0.338955</td>
    </tr>
    <tr>
      <th>0</th>
      <td>RevolvingUtilizationOfUnsecuredLines</td>
      <td>0.213334</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NumberOfTimes90DaysLate</td>
      <td>0.195856</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NumberOfTime30-59DaysPastDueNotWorse</td>
      <td>0.190149</td>
    </tr>
    <tr>
      <th>1</th>
      <td>age</td>
      <td>0.028528</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DebtRatio</td>
      <td>0.024862</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NumberRealEstateLoansOrLines</td>
      <td>0.008318</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MonthlyIncome</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NumberOfOpenCreditLinesAndLoans</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NumberOfDependents</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-48f0f7d6-8e75-4154-a7a0-2f7348f7dd75')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-48f0f7d6-8e75-4154-a7a0-2f7348f7dd75 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-48f0f7d6-8e75-4154-a7a0-2f7348f7dd75');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-275b5552-44d7-470e-aac4-79eb690b62bf">
      <button class="colab-df-quickchart" onclick="quickchart('df-275b5552-44d7-470e-aac4-79eb690b62bf')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-275b5552-44d7-470e-aac4-79eb690b62bf button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
# Función de visualización para regresiones
def plot_coef(modelo, feature_names, nombre_modelo="Regresión"):
    """
    Parámetros:
    - modelo: modelo entrenado (con atributo .coef_)
    - feature_names: lista con nombres de las características
    - nombre_modelo: título para el gráfico
    """
    # Extraer coeficientes
    coefs = modelo.coef_[0] if len(modelo.coef_.shape) > 1 else modelo.coef_
    # Crear DataFrame ordenado
    df_coef = pd.DataFrame({
        'Feature': feature_names,
        'Coeficiente': coefs
    }).sort_values(by='Coeficiente', key=abs, ascending=False)
    # Graficar
    plt.figure(figsize=(10, 4))
    sns.barplot(data=df_coef, x='Coeficiente', y='Feature', palette='coolwarm')
    plt.title(f'Impacto de Características - {nombre_modelo}')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.show()

```

- Los coeficientes del modelo indican la influencia de cada variable. Por ejemplo, en este caso, un coeficiente positivo implica que un aumento en esa variable aumenta la probabilidad de desaprobación, que el cliente posee riesgo crediticio.
- "Número de veces que el cliente tuvo pagos con 30-59 días de atraso" (sin llegar a mora grave).	Valores altos sugieren riesgo crediticio.
- "Número de veces que el cliente tuvo pagos con +90 días de atraso".	Indicador fuerte de morosidad grave.


```python
plot_coef(logreg, X.columns, "Sin regularización")
```

    /tmp/ipython-input-54-939371788.py:18: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(data=df_coef, x='Coeficiente', y='Feature', palette='coolwarm')



    
![png](output_56_1.png)
    



```python
plot_coef(lasso, X.columns, "Lasso")
```

    /tmp/ipython-input-54-939371788.py:18: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(data=df_coef, x='Coeficiente', y='Feature', palette='coolwarm')



    
![png](output_57_1.png)
    



```python
plot_coef(ridge, X.columns, "Ridge")
```

    /tmp/ipython-input-54-939371788.py:18: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(data=df_coef, x='Coeficiente', y='Feature', palette='coolwarm')



    
![png](output_58_1.png)
    



```python
plot_coef(elastic, X.columns, "ElasticNet")
```

    /tmp/ipython-input-54-939371788.py:18: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(data=df_coef, x='Coeficiente', y='Feature', palette='coolwarm')



    
![png](output_59_1.png)
    


## **Análisis Comparativo de Modelos de Regresión Logística**


```python
# Función para recolectar métricas de los modelos
def get_metrics(model_name, model):
    y_pred = model.predict(X_test)
    return {
        'Modelo': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }
```


```python
#Obtener métricas para cada modelo
metrics_reglog = get_metrics("Regresión Logística", logreg)
metrics_lasso = get_metrics("Regresión Logística Lasso ", lasso)
metrics_ridge = get_metrics("Regresión Logística Ridge", ridge)
metrics_elastic = get_metrics("Regresión Logística Elastic Net", elastic)
#Crear DataFrame con los resultados
result_reg = pd.DataFrame([metrics_reglog, metrics_lasso,metrics_ridge,metrics_elastic])
```


```python
result_reg
```





  <div id="df-3a4cdd7b-0065-41d0-9574-cca66b194031" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Modelo</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-Score</th>
      <th>AUC-ROC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Regresión Logística</td>
      <td>0.759797</td>
      <td>0.76990</td>
      <td>0.740874</td>
      <td>0.755108</td>
      <td>0.842341</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Regresión Logística Lasso</td>
      <td>0.760694</td>
      <td>0.77134</td>
      <td>0.740874</td>
      <td>0.755800</td>
      <td>0.842574</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Regresión Logística Ridge</td>
      <td>0.759797</td>
      <td>0.76990</td>
      <td>0.740874</td>
      <td>0.755108</td>
      <td>0.842341</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Regresión Logística Elastic Net</td>
      <td>0.759797</td>
      <td>0.76990</td>
      <td>0.740874</td>
      <td>0.755108</td>
      <td>0.842500</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-3a4cdd7b-0065-41d0-9574-cca66b194031')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-3a4cdd7b-0065-41d0-9574-cca66b194031 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3a4cdd7b-0065-41d0-9574-cca66b194031');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-3d54458c-24b4-400b-aa3b-5080416fd82f">
      <button class="colab-df-quickchart" onclick="quickchart('df-3d54458c-24b4-400b-aa3b-5080416fd82f')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-3d54458c-24b4-400b-aa3b-5080416fd82f button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_0801e8e4-eeba-46f5-a7a1-6e1ae85664eb">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('result_reg')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_0801e8e4-eeba-46f5-a7a1-6e1ae85664eb button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('result_reg');
      }
      })();
    </script>
  </div>

    </div>
  </div>




**Comparación entre modelos**

Todos los modelos tienen métricas muy cercanas, lo que sugiere que la regularización no está teniendo un impacto significativo en este dataset. Esto puede deberse a bajo overfitting, es decir, los datos ya son linealmente separables o no tienen ruido excesivo. En este dataset la variable objetivo está dividida de manera balanceada 50/50 lo que puede ser el motivo del desempeño.

- El mejor modelo dentro de este grupo es la **Regresión Logística con Lasso (L1)**, ya que logra:

   - Mayor Accuracy (0.760694 vs. 0.759797 en los demás).

   - Mayor Precision (0.77134).

   - Mejor F1-Score (0.755800).

  - Mayor AUC-ROC (0.842574).

Aunque la diferencia es pequeña (probablemente no significativa estadísticamente), Lasso tiene la ventaja adicional de hacer selección de variables al poner a cero coeficientes irrelevantes, lo que mejora la interpretabilidad y puede beneficiar el modelo final.

## **Análisis comparativo entre algoritmos de bagging y boosting**


```python
#Obtener métricas para cada modelo
metrics_rf = get_metrics("Random Forest", rf)
metrics_rf_grid = get_metrics("Random Forest Grid Search", grid_search)
metrics_rf_random = get_metrics("Random Forest Random Search", best_random)
metrics_rf_opt = get_metrics("Random Forest Opt Bayesiana", opt)
metrics_xg = get_metrics("XGBoost", xg)
metrics_ada = get_metrics("AdaBoost", ada)

#Crear DataFrame con los resultados
result_ensamb = pd.DataFrame([metrics_rf, metrics_rf_grid, metrics_rf_random, metrics_rf_opt, metrics_xg, metrics_ada])
```


```python
result_ensamb
```





  <div id="df-88ff8cb3-9025-4df4-b940-d801c3876aa8" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Modelo</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-Score</th>
      <th>AUC-ROC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Random Forest</td>
      <td>0.775950</td>
      <td>0.774405</td>
      <td>0.778576</td>
      <td>0.776485</td>
      <td>0.852125</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Random Forest Grid Search</td>
      <td>0.781334</td>
      <td>0.784160</td>
      <td>0.776182</td>
      <td>0.780150</td>
      <td>0.861878</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Random Forest Random Search</td>
      <td>0.781633</td>
      <td>0.783947</td>
      <td>0.777379</td>
      <td>0.780649</td>
      <td>0.861530</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Random Forest Opt Bayesiana</td>
      <td>0.779240</td>
      <td>0.781193</td>
      <td>0.775583</td>
      <td>0.778378</td>
      <td>0.861038</td>
    </tr>
    <tr>
      <th>4</th>
      <td>XGBoost</td>
      <td>0.780736</td>
      <td>0.792394</td>
      <td>0.760622</td>
      <td>0.776183</td>
      <td>0.860032</td>
    </tr>
    <tr>
      <th>5</th>
      <td>AdaBoost</td>
      <td>0.769070</td>
      <td>0.784314</td>
      <td>0.742071</td>
      <td>0.762608</td>
      <td>0.849645</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-88ff8cb3-9025-4df4-b940-d801c3876aa8')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-88ff8cb3-9025-4df4-b940-d801c3876aa8 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-88ff8cb3-9025-4df4-b940-d801c3876aa8');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-71132b58-b6e2-4da7-a9c2-df42e0c8947c">
      <button class="colab-df-quickchart" onclick="quickchart('df-71132b58-b6e2-4da7-a9c2-df42e0c8947c')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-71132b58-b6e2-4da7-a9c2-df42e0c8947c button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_165f65e6-4a3e-4885-bbf2-08a9cbe6ba74">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('result_ensamb')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_165f65e6-4a3e-4885-bbf2-08a9cbe6ba74 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('result_ensamb');
      }
      })();
    </script>
  </div>

    </div>
  </div>




**Comparación entre modelos:**


- En Accuracy, el mejor es **Random Forest Random Search** (0.781633), apenas por encima de Grid Search (0.781334).

- En F1-Score, también gana Random Forest Random Search (0.780649).

- En AUC-ROC, gana Random Forest Grid Search (0.861878), seguido muy de cerca por Random Search (0.861530).

- En Precision, XGBoost tiene el valor más alto (0.792394), pero sacrifica Recall y F1-Score.

- Mejor balance general: Random Forest Random Search

    - Mantiene el mayor Accuracy y F1-Score.

    - Tiene un AUC-ROC casi idéntico al mejor (diferencia de 0.000348).

    - Rendimiento muy parejo en todas las métricas, sin sacrificar Recall.

## **Resumen de Comparación de Modelos de Clasificación**

**Análisis comparativo de Random Forest optimizado con Random Search y Regresión Logística Lasso (mejor entre las regresiones)**


```python
#Crear DataFrame con los resultados
resultF = pd.DataFrame([metrics_rf_random, metrics_lasso])
resultF
```





  <div id="df-60ee411a-a216-44ec-9acf-d6d3f1743ae7" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Modelo</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-Score</th>
      <th>AUC-ROC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Random Forest Random Search</td>
      <td>0.781633</td>
      <td>0.783947</td>
      <td>0.777379</td>
      <td>0.780649</td>
      <td>0.861530</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Regresión Logística Lasso</td>
      <td>0.760694</td>
      <td>0.771340</td>
      <td>0.740874</td>
      <td>0.755800</td>
      <td>0.842574</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-60ee411a-a216-44ec-9acf-d6d3f1743ae7')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-60ee411a-a216-44ec-9acf-d6d3f1743ae7 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-60ee411a-a216-44ec-9acf-d6d3f1743ae7');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-2d040170-4037-4c11-88eb-781a5a0d82c2">
      <button class="colab-df-quickchart" onclick="quickchart('df-2d040170-4037-4c11-88eb-781a5a0d82c2')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-2d040170-4037-4c11-88eb-781a5a0d82c2 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_1edb24ad-7f94-4d6f-a6a5-8331b91fcb32">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('resultF')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_1edb24ad-7f94-4d6f-a6a5-8331b91fcb32 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('resultF');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
# Gráfica de la comparación de modelos
metrics = resultF.columns[1:]
x = np.arange(len(metrics))  # posiciones en el eje X
width = 0.35  # ancho de cada barra

# Crear gráfico
fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, resultF.loc[0, metrics], width,
               label=resultF.loc[0, 'Modelo'], color='#1f77b4')
bars2 = ax.bar(x + width/2, resultF.loc[1, metrics], width,
               label=resultF.loc[1, 'Modelo'], color='#ff7f0e')

# Etiquetas y título
ax.set_ylabel('Valor')
ax.set_title('Comparación de métricas entre modelos')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0.7, 0.9)  # ajusta rango visual
ax.legend()

# Añadir valores sobre las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()
```


    
![png](output_72_0.png)
    


**Conclusión**

En la fase de evaluación de modelos, se analizaron dos enfoques principales para la predicción de riesgo crediticio: modelos de regresión logística (con regularización Lasso, Ridge y Elastic Net) y modelos de ensamble (Random Forest, XGBoost y AdaBoost), aplicando distintas estrategias de optimización de hiperparámetros.

Dentro del grupo de modelos de regresión, el mejor desempeño lo obtuvo la Regresión Logística con regularización Lasso (L1), alcanzando un Accuracy de 0.7607, un F1-Score de 0.7558 y un AUC-ROC de 0.8426. Este modelo presentó un buen equilibrio entre precisión y exhaustividad, además de la ventaja adicional de realizar selección de variables, mejorando la interpretabilidad.

En el grupo de modelos de ensamble, el rendimiento más alto se logró con Random Forest optimizado mediante búsqueda aleatoria de hiperparámetros (Random Search). Este modelo obtuvo un Accuracy de 0.7816, un F1-Score de 0.7806 y un AUC-ROC de 0.8615, superando a los demás en casi todas las métricas clave. Su desempeño muestra una mejora significativa respecto a la regresión logística, particularmente en la capacidad de discriminar entre clientes de alto y bajo riesgo.

El Random Forest con búsqueda aleatoria de hiperparámetros es claramente superior:

- Accuracy +2.1 puntos respecto a Lasso.

- F1-Score notablemente mayor (+0.025).

- AUC-ROC más alto (+0.019), lo que significa mejor discriminación entre clases.

- Buen equilibrio entre Precision y Recall, lo que es clave en un problema de riesgo crediticio.

En un escenario real, esto significa que Random Forest Random Search no solo clasifica más casos correctamente, sino que también identifica mejor a los clientes de alto riesgo sin perder demasiados casos positivos.

La regresión logística (con o sin regularización) permite ver coeficientes positivos o negativos y cómo cada variable afecta la probabilidad de aprobación, pero quedan descartadas por su menor rendimiento en este caso.
