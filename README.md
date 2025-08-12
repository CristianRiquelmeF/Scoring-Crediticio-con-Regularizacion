# 📊 Scoring Crediticio con Regularización

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/usuario/nombre-repo/blob/main/M5Final_CristianRiquelme.ipynb)

## 📌 Descripción
Este proyecto desarrolla un **modelo predictivo de scoring crediticio** utilizando un dataset de historial crediticio.  
El objetivo principal es predecir si un cliente tiene **alto riesgo crediticio** (variable objetivo binaria) y evaluar el rendimiento del modelo junto con la capacidad de interpretar sus decisiones.

Se aplican técnicas de **regularización** como *Lasso* y *Ridge*, así como modelos basados en árboles para mejorar la precisión y reducir el sobreajuste.

---

## 📂 Contenido del proyecto
- **Exploración de datos:** análisis estadístico y visualización de las variables.
- **Preprocesamiento:** estandarización de variables numéricas y tratamiento de valores atípicos.
- **Modelado:**
  - Regresión logística con regularización L1 y L2.
  - Random Forest para comparación.
- **Evaluación de modelos:** métricas como *accuracy*, *precision*, *recall*, *F1-score* y *AUC-ROC*.
- **Interpretación:** análisis de importancia de variables para explicar las predicciones.

---

## 📊 Dataset
El dataset contiene información de clientes, incluyendo:
- **RevolvingUtilizationOfUnsecuredLines:** Proporción de crédito usado vs. disponible.
- **Age:** Edad del solicitante.
- **NumberOfTimes90DaysLate:** Número de atrasos de 90 días.
- **DebtRatio:** Relación entre deuda e ingresos.
- **MonthlyIncome:** Ingreso mensual estimado.
- **NumberOfOpenCreditLinesAndLoans:** Número de líneas de crédito y préstamos abiertos.
- Entre otras variables relevantes para el riesgo crediticio.

---

## ⚙️ Tecnologías utilizadas
- **Python 3**
- **Pandas**, **NumPy** → Manipulación y análisis de datos
- **Matplotlib**, **Seaborn** → Visualización
- **Scikit-learn** → Modelado y evaluación
- **GridSearchCV** → Optimización de hiperparámetros

---

## 🚀 Ejecución del proyecto
1. Clonar este repositorio:
   ```bash
   git clone https://github.com/usuario/nombre-repo.git
   cd nombre-repo
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Abrir el notebook en **Google Colab** haciendo clic en el badge al inicio del README.

---

## 📈 Resultados principales
- El modelo de **Random Forest** obtuvo un mejor equilibrio entre *precision* y *recall* en comparación con la regresión logística.
- La curva **ROC** mostró un área bajo la curva (AUC) superior al 0.80 para el mejor modelo.
- Variables más influyentes: `NumberOfTimes90DaysLate`, `RevolvingUtilizationOfUnsecuredLines` y `DebtRatio`.

---

## 📌 Conclusiones
- La regularización ayuda a manejar el sobreajuste y mejora la interpretabilidad.
- Los modelos basados en árboles proporcionan mejor rendimiento, pero la regresión logística ofrece mayor transparencia.
- Es posible integrar este modelo en un sistema de aprobación crediticia para reducir riesgos.

---

## ✒️ Autor
**Cristian Riquelme**  
[GitHub](https://github.com/CristianRiquelmeF)

---

### 📦 requirements.txt
Este archivo incluye las librerías necesarias:
```
matplotlib
pandas
scikit-learn
seaborn
```

---

## 🛠 Cómo crear un nuevo repositorio desde cero con este proyecto (Google Colab)
Si tu archivo principal está en Google Colab y quieres subirlo a GitHub:

1. **Crear el repositorio en GitHub**
   - Ve a [https://github.com/new](https://github.com/new)
   - Asigna un nombre al repositorio, por ejemplo: `scoring-crediticio`
   - Marca la opción **"Add a README file"** (si quieres que se inicialice con README)
   - Crea el repositorio.

2. **Descargar el notebook desde Colab**
   - En Colab, ve a `Archivo > Descargar > Descargar .ipynb`.
   - Guarda el archivo en tu computadora.

3. **Clonar el repositorio en tu PC**
   ```bash
   git clone https://github.com/usuario/scoring-crediticio.git
   cd scoring-crediticio
   ```

4. **Agregar los archivos del proyecto**
   - Copia en la carpeta del repositorio:
     - `M5Final_CristianRiquelme.ipynb`
     - `README.md` (este que te di)
     - `requirements.txt`

5. **Subirlos a GitHub**
   ```bash
   git add .
   git commit -m "Proyecto scoring crediticio inicial"
   git push origin main
   ```

6. **Abrir el notebook en Colab desde GitHub**
   - En Colab, ve a `Archivo > Abrir cuaderno > GitHub`.
   - Pega la URL de tu repositorio y selecciona el notebook.

---

📌 **Importante:** Antes de subir a GitHub, cambia en el enlace del badge:
```
https://github.com/usuario/nombre-repo/blob/main/M5Final_CristianRiquelme.ipynb
```
por tu usuario y nombre real del repositorio para que el botón funcione.
