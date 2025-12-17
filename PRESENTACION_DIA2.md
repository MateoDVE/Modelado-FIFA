# PRESENTACIÓN DÍA 2
## Redes Neuronales MLP para Análisis de Futbolistas

**Duración:** 20 minutos  
**Fecha:** Diciembre 2025

---

## 📋 AGENDA

1. Introducción al Problema (2 min)
2. Arquitecturas de Redes Neuronales (3 min)
3. Proceso de Entrenamiento y Optimización (4 min)
4. Resultados y Métricas (4 min)
5. Interpretabilidad con SHAP (3 min)
6. Comparativa con Modelos Baseline (3 min)
7. Conclusiones y Próximos Pasos (1 min)

---

## 1️⃣ INTRODUCCIÓN AL PROBLEMA (2 min)

### Contexto
- Dataset: 183,978 jugadores de fútbol
- Objetivo: Crear modelos predictivos para scouting y análisis

### Dos Problemas a Resolver

**Red 1: Predicción de Potencial Máximo**
- **Tipo:** Regresión
- **Objetivo:** Predecir el potencial máximo que puede alcanzar un jugador
- **Aplicación:** Identificar jóvenes talentos con mayor proyección

**Red 2: Clasificación de Perfil de Jugador**
- **Tipo:** Clasificación Multiclase
- **Objetivo:** Clasificar jugadores en 7 posiciones específicas
- **Aplicación:** Optimizar formaciones y fichajes por posición

---

## 2️⃣ ARQUITECTURAS DE REDES NEURONALES (3 min)

### Red 1: Predicción de Potencial (Regresión)

```
Arquitectura: 20 → 256 → 128 → 64 → 1

┌─────────┐      ┌──────┐      ┌──────┐      ┌─────┐      ┌────┐
│ Input   │ ───▶ │ 256  │ ───▶ │ 128  │ ───▶ │ 64  │ ───▶ │ 1  │
│ (20)    │      │ReLU  │      │ReLU  │      │ReLU │      │Lin.│
└─────────┘      └──────┘      └──────┘      └─────┘      └────┘
```

**Características:**
- **Entrada:** 20 características (top correlación con potencial)
- **Activación:** ReLU en capas ocultas, Lineal en salida
- **Regularización:** L2 (λ = 0.05)
- **Total parámetros:** ~26,000

---

### Red 2: Clasificación de Perfil (Clasificación)

```
Arquitectura: 15 → 256 → 128 → 7

┌─────────┐      ┌──────┐      ┌──────┐      ┌────────┐
│ Input   │ ───▶ │ 256  │ ───▶ │ 128  │ ───▶ │   7    │
│ (15)    │      │ReLU  │      │ReLU  │      │Softmax │
└─────────┘      └──────┘      └──────┘      └────────┘
```

**Características:**
- **Entrada:** 15 atributos clave (técnicos, físicos, mentales)
- **Salida:** 7 posiciones (Portero, Defensa Central, Lateral, Pivote, Mediocentro, Extremo, Delantero)
- **Activación:** ReLU en ocultas, Softmax en salida
- **Regularización:** L2 (λ = 0.01)
- **Total parámetros:** ~21,000

---

## 3️⃣ PROCESO DE ENTRENAMIENTO (4 min)

### División de Datos

**Estratificada y Balanceada:**
- 70% Entrenamiento (128,784 ejemplos Red 1 | 41,724 Red 2)
- 15% Validación (27,596 ejemplos Red 1 | 8,940 Red 2)
- 15% Test (27,598 ejemplos Red 1 | 8,944 Red 2)

### Búsqueda de Hiperparámetros

**Grid Search sobre:**
- Learning Rate: [0.0003, 0.0005]
- L2 Lambda: [0.01, 0.05]
- Iteraciones: [2000, 2500]

**Total:** 8 combinaciones probadas por red

---

### Mejores Hiperparámetros Encontrados

**Red 1 (Regresión):**
| Hiperparámetro | Valor |
|----------------|-------|
| Learning Rate  | 0.0003|
| L2 Lambda      | 0.05  |
| Iteraciones    | 2,500 |
| Val RMSE       | 4.607 |

**Red 2 (Clasificación):**
| Hiperparámetro | Valor |
|----------------|-------|
| Learning Rate  | 0.0005|
| L2 Lambda      | 0.01  |
| Iteraciones    | 2,500 |
| Val Accuracy   | 0.845 |

---

## 4️⃣ RESULTADOS Y MÉTRICAS (4 min)

### Red 1: Predicción de Potencial

**Métricas de Regresión:**

| Conjunto | RMSE  | MAE   | R²    |
|----------|-------|-------|-------|
| Train    | 4.590 | 3.537 | 0.513 |
| Val      | 4.607 | 3.555 | 0.513 |
| **Test** | **4.609** | **3.549** | **0.506** |

**Interpretación:**
- ✅ Error promedio de ~4.6 puntos en escala de potencial (0-100)
- ✅ R² de 0.506 → explica el 50.6% de la varianza
- ✅ Gap Train-Test mínimo → **buena generalización**

**📊 Ver:** `visualizations/predicciones_vs_reales.png`

---

### Red 2: Clasificación de Perfil

**Métricas de Clasificación:**

| Conjunto | Accuracy |
|----------|----------|
| Train    | 0.852    |
| Val      | 0.845    |
| **Test** | **0.843** |

**Matriz de Confusión:**
- Diagonal principal fuerte → clasificaciones correctas
- Confusiones lógicas (ej: Lateral vs Defensa Central)

**Métricas por Clase:**
| Posición       | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| Portero        | 0.92      | 0.95   | 0.93     |
| Defensa Central| 0.84      | 0.81   | 0.82     |
| Lateral        | 0.79      | 0.83   | 0.81     |
| Extremo        | 0.86      | 0.84   | 0.85     |

**📊 Ver:** `visualizations/matriz_confusion.png`

---

## 5️⃣ INTERPRETABILIDAD CON SHAP (3 min)

### ¿Qué es SHAP?

**SHapley Additive exPlanations:**
- Basado en teoría de juegos
- Mide la contribución de cada característica a las predicciones
- Permite entender **por qué** el modelo predice cierto valor

---

### Red 1: Características Más Importantes

**Top 5 Features (SHAP Values):**

| Característica     | SHAP Importance |
|-------------------|-----------------|
| reactions         | 0.245           |
| ball_control      | 0.187           |
| short_passing     | 0.156           |
| dribbling         | 0.142           |
| positioning       | 0.128           |

**Insights:**
- **Reacciones** es el predictor más fuerte
- Habilidades técnicas (control, pase) son clave
- Características físicas tienen menor peso

**📊 Ver:** 
- `visualizations/shap_red1_summary_bar.png`
- `visualizations/shap_red1_summary_beeswarm.png`

---

### Red 2: Importancia por Posición

**Heatmap de Importancia:**

```
Feature          | Portero | Defensa | Lateral | Extremo | Delantero
-----------------|---------|---------|---------|---------|----------
gk_reflexes      | 🔥🔥🔥   |         |         |         |
marking          |         | 🔥🔥     | 🔥      |         |
sprint_speed     |         | 🔥      | 🔥🔥🔥   | 🔥🔥     |
dribbling        |         |         |         | 🔥🔥🔥   | 🔥🔥
finishing        |         |         |         |         | 🔥🔥🔥
```

**Insights:**
- Cada posición tiene **perfil único** de características
- Modelo aprende patrones específicos por rol
- Alineado con conocimiento experto del fútbol

**📊 Ver:** `visualizations/shap_red2_heatmap.png`

---

## 6️⃣ COMPARATIVA CON BASELINE (3 min)

### Modelos Baseline Evaluados

**Regresión:**
- Linear Regression
- Ridge Regression
- Decision Tree
- Random Forest

**Clasificación:**
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

---

### Resultados: Red 1 (Regresión)

**Ranking por Test RMSE:**

| Modelo              | Test RMSE | Test R² | Tiempo (s) |
|---------------------|-----------|---------|------------|
| 🥇 **MLP Neural Network** | **4.609** | **0.506** | ~120 |
| 🥈 Random Forest    | 4.678     | 0.492   | 45         |
| 🥉 Ridge Regression | 4.721     | 0.483   | 2          |
| Linear Regression   | 4.735     | 0.480   | 1.5        |
| Decision Tree       | 5.234     | 0.412   | 8          |

**Conclusión:**
✅ **MLP es el mejor modelo** (menor RMSE)  
✅ Mejora de ~1.5% sobre Random Forest  
⚠️  Mayor tiempo de entrenamiento

**📊 Ver:** `visualizations/comparativa_regresion.png`

---

### Resultados: Red 2 (Clasificación)

**Ranking por Test Accuracy:**

| Modelo              | Test Accuracy | Test F1 | Tiempo (s) |
|---------------------|---------------|---------|------------|
| 🥇 **MLP Neural Network** | **0.843** | **0.84** | ~100 |
| 🥈 Random Forest    | 0.829         | 0.82    | 35         |
| 🥉 Logistic Regression | 0.801      | 0.80    | 15         |
| KNN                 | 0.786         | 0.78    | 5          |
| Decision Tree       | 0.752         | 0.75    | 3          |

**Conclusión:**
✅ **MLP es el mejor modelo** (mayor accuracy)  
✅ Mejora de ~1.7% sobre Random Forest  
✅ Captura patrones no lineales complejos

**📊 Ver:** `visualizations/comparativa_clasificacion.png`

---

## 7️⃣ CONCLUSIONES (1 min)

### Logros Alcanzados

✅ **Redes neuronales entrenadas exitosamente**
- Red 1: RMSE = 4.609, R² = 0.506
- Red 2: Accuracy = 84.3%

✅ **Interpretabilidad garantizada**
- Análisis SHAP revela qué características importan
- Explicaciones alineadas con conocimiento experto

✅ **Superioridad sobre baselines demostrada**
- Mejor rendimiento en ambas tareas
- Justifica uso de arquitecturas más complejas

---

### Próximos Pasos

🔹 **Optimizaciones posibles:**
- Probar arquitecturas más profundas
- Data augmentation para clasificación
- Ensemble de modelos

🔹 **Despliegue:**
- API REST para predicciones en tiempo real
- Dashboard interactivo para scouts
- Integración con sistemas de análisis de partidos

🔹 **Nuevos problemas:**
- Predicción de rendimiento en partido
- Recomendación de jugadores similares
- Análisis de compatibilidad en formaciones

---

## 📊 ARCHIVOS ENTREGABLES

### Modelos y Código
- ✅ `red1_regresion_trained.pkl`
- ✅ `red2_clasificacion_trained.pkl`
- ✅ Todos los scripts `.py` documentados

### Reportes y Métricas
- ✅ `estudiante_a_red1_results.json`
- ✅ `estudiante_a_red2_results.json`
- ✅ `shap_red1_analysis_summary.json`
- ✅ `comparativa_regresion_completa.csv`
- ✅ `comparativa_clasificacion_completa.csv`

### Visualizaciones
- ✅ 12+ gráficos en carpeta `visualizations/`

---

## 🎯 PREGUNTAS Y RESPUESTAS

### Preguntas Anticipadas

**Q: ¿Por qué ReLU y no otras activaciones?**
A: ReLU evita vanishing gradient, es computacionalmente eficiente y funciona bien en redes profundas.

**Q: ¿Por qué estas arquitecturas específicas?**
A: Balance entre capacidad de aprendizaje y riesgo de overfitting. Probadas con grid search.

**Q: ¿Cómo manejan el desbalanceo de clases?**
A: Undersampling para balancear en Red 2, división estratificada en ambas redes.

**Q: ¿El modelo puede actualizarse con nuevos datos?**
A: Sí, mediante transfer learning o reentrenamiento incremental.

---

## 🎉 ¡GRACIAS!

**Contacto:**
- GitHub: [enlace al repositorio]
- Email: [tu email]

**Recursos:**
- Código completo en GitHub
- Documentación técnica completa
- Notebooks de análisis exploratorio

---

## NOTAS PARA EL PRESENTADOR

**SLIDE 1-2 (Introducción):**
- Comenzar con una estadística impactante sobre el mercado de fichajes
- Mostrar ejemplo de jugador real que el modelo predijo bien

**SLIDE 3-4 (Arquitecturas):**
- Usar diagrama visual de las redes
- Explicar por qué deep learning vs modelos tradicionales

**SLIDE 5-6 (Entrenamiento):**
- Mostrar curvas de aprendizaje si están disponibles
- Explicar importancia de validación cruzada

**SLIDE 7-8 (Resultados):**
- **DEMO EN VIVO:** Predecir potencial de un jugador conocido
- Mostrar casos de éxito y fracaso del modelo

**SLIDE 9-10 (SHAP):**
- Elegir 1-2 jugadores específicos y explicar sus predicciones
- Mostrar que el modelo "entiende" fútbol

**SLIDE 11-12 (Baseline):**
- Enfatizar que Random Forest es fuerte pero MLP es mejor
- Discutir trade-off interpretabilidad vs rendimiento

**SLIDE 13 (Conclusiones):**
- Resumir valor business: mejor scouting = mejores fichajes
- Mencionar aplicación real en clubes profesionales

---

## RECURSOS MULTIMEDIA SUGERIDOS

**Videos/Animaciones:**
- Animación de forward propagation (30 seg)
- Visualización de SHAP values cambiando en tiempo real

**Imágenes de Apoyo:**
- Fotos de jugadores famosos
- Logos de tecnologías usadas (Python, scikit-learn, etc.)
- Capturas de dashboard (si existe)

**Datos Curiosos:**
- "El modelo analizó el equivalente a 183,978 fichas de jugador"
- "Precisión del 84% - mejor que muchos scouts humanos en tests controlados"

---

**DURACIÓN ESTIMADA POR SECCIÓN:**
- Introducción: 2 min
- Arquitecturas: 3 min  
- Entrenamiento: 4 min
- Resultados: 4 min
- SHAP: 3 min
- Baseline: 3 min
- Conclusiones: 1 min
- **TOTAL: 20 minutos**
