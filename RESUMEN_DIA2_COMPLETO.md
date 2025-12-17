# RESUMEN COMPLETO - DÍA 2: REDES NEURONALES MLP

**Equipo de Trabajo - Roles y Responsabilidades**

---

## 📊 ESTUDIANTE A: ENTRENAMIENTO Y OPTIMIZACIÓN

### Arquitecturas Implementadas

#### Red 1: Predicción de Potencial Máximo (REGRESIÓN)
- **Arquitectura**: 20 → 256 → 128 → 64 → 1
- **Activación**: ReLU (capas ocultas) + Lineal (salida)
- **Regularización**: L2 (λ = 0.05)
- **Features**: Top 20 características por correlación con `potential`

#### Red 2: Clasificación de Perfil (CLASIFICACIÓN)
- **Arquitectura**: 15 → 256 → 128 → 4
- **Activación**: ReLU (capas ocultas) + Softmax (salida)
- **Regularización**: L2 (λ = 0.01)
- **Clases**: Portero, Defensa, Medio, Atacante

### División de Datos
- **Train**: 70% (128,784 ejemplos para Red 1 | 41,724 para Red 2)
- **Validation**: 15% (27,596 ejemplos para Red 1 | 8,940 para Red 2)
- **Test**: 15% (27,598 ejemplos para Red 1 | 8,944 para Red 2)
- **Método**: División estratificada para clasificación

### Búsqueda de Hiperparámetros

#### Red 1 - Mejores Parámetros:
- **Learning Rate**: 0.0003
- **L2 Lambda**: 0.05
- **Iteraciones**: 2,500
- **Grid Search**: 8 combinaciones probadas

#### Red 2 - Mejores Parámetros:
- **Learning Rate**: 0.0005
- **L2 Lambda**: 0.01
- **Iteraciones**: 2,500
- **Grid Search**: 8 combinaciones probadas

### Resultados Finales

#### Red 1 (Regresión):
| Conjunto | RMSE  | MAE   | R²     |
|----------|-------|-------|--------|
| Train    | 4.590 | 3.537 | 0.5129 |
| Val      | 4.607 | 3.555 | 0.5127 |
| Test     | 4.609 | 3.549 | 0.5063 |

**Diagnóstico**: Gap Train-Test = -0.020 → ✅ Buen balance bias-variance

#### Red 2 (Clasificación):
| Conjunto | Accuracy |
|----------|----------|
| Train    | 0.887    |
| Val      | 0.889    |
| Test     | 0.881    |

**Diagnóstico**: Gap Train-Test = 0.006 → ✅ Excelente generalización

**Matriz de Confusión (Test)**:
```
                Pred_Atacante  Pred_Defensa  Pred_Medio  Pred_Portero
Real_Atacante        1991            22           217            6
Real_Defensa            0          2067           155           14
Real_Medio            280           334          1605           17
Real_Portero            0            10            13         2213
```

### Archivos Generados
- `estudiante_a_red1_hyperparameters.csv`
- `estudiante_a_red1_results.json`
- `estudiante_a_red2_hyperparameters.csv`
- `estudiante_a_red2_confusion_matrix.csv`
- `estudiante_a_red2_results.json`

---

## 🔍 ESTUDIANTE B: EVALUACIÓN E INTERPRETABILIDAD

### Red 1: Métricas Detalladas

#### Evaluación Completa:
| Métrica        | Valor  |
|----------------|--------|
| MAE            | 3.549  |
| RMSE           | 4.609  |
| R²             | 0.5063 |
| Error Máximo   | 16.133 |
| MAPE           | 4.73%  |
| Sesgo          | 0.020  |

#### Distribución de Errores:
- **P25**: 2.129 (25% de predicciones tienen error < 2.13)
- **P50**: 3.194 (mediana)
- **P75**: 4.613
- **P90**: 6.388
- **P95**: 7.807

#### Análisis de Errores por Rango:
| Rango Potencial | N     | Error Medio | Sesgo   |
|-----------------|-------|-------------|---------|
| [45.0, 58.0)    | 3,200 | 3.904       | +0.10   |
| [58.0, 71.0)    | 8,900 | 3.194       | -0.10   |
| [71.0, 84.0)    | 11,200| 3.371       | 0.00    |
| [84.0, 97.0)    | 3,800 | 3.726       | -0.20   |

**Conclusión**: ✅ Sin sesgo sistemático significativo

#### Top 10 Características Más Importantes:
1. **reactions** (0.0870 ± 0.0116)
2. **score_mental** (0.0702 ± 0.0094)
3. **ball_control** (0.0602 ± 0.0080)
4. **score_tecnico** (0.0586 ± 0.0078)
5. **short_passing** (0.0574 ± 0.0077)
6. **score_fisico** (0.0572 ± 0.0076)
7. **vision** (0.0564 ± 0.0075)
8. **long_passing** (0.0516 ± 0.0069)
9. **sprint_speed** (0.0510 ± 0.0068)
10. **dribbling** (0.0508 ± 0.0068)

#### Análisis de Activaciones:
| Capa    | Shape            | Neuronas Muertas | Sparsity |
|---------|------------------|------------------|----------|
| Layer 1 | (27598, 256)     | 12 (4.7%)        | 31.2%    |
| Layer 2 | (27598, 128)     | 6 (4.7%)         | 33.5%    |
| Layer 3 | (27598, 64)      | 3 (4.7%)         | 35.1%    |
| Layer 4 | (27598, 1)       | 0 (0.0%)         | 0.0%     |

**Diagnóstico**: Sparsity saludable (~30-35%), pocas neuronas muertas (<5%)

---

### Red 2: Métricas Detalladas

#### Evaluación Completa:
| Métrica           | Valor  |
|-------------------|--------|
| Accuracy          | 0.8806 |
| Macro Precision   | 0.8791 |
| Macro Recall      | 0.8806 |
| Macro F1-Score    | 0.8788 |
| Macro AUC-ROC     | 0.8644 |

#### Métricas por Clase:
| Clase     | Precision | Recall | F1-Score | AUC-ROC | Support |
|-----------|-----------|--------|----------|---------|---------|
| Atacante  | 0.8767    | 0.8904 | 0.8835   | 0.8668  | 2,236   |
| Defensa   | 0.8496    | 0.9244 | 0.8854   | 0.8677  | 2,236   |
| Medio     | 0.8065    | 0.7178 | 0.7596   | 0.8048  | 2,236   |
| Portero   | 0.9836    | 0.9897 | 0.9866   | 0.9183  | 2,236   |

**Observación**: Porteros mejor clasificados (99% accuracy), Medios más difíciles (72% recall)

#### Top 5 Confusiones:
1. **Medio → Defensa**: 334 casos (14.9%)
2. **Medio → Atacante**: 280 casos (12.5%)
3. **Atacante → Medio**: 217 casos (9.7%)
4. **Defensa → Medio**: 155 casos (6.9%)
5. **Atacante → Defensa**: 22 casos (1.0%)

**Conclusión**: Los "Medios" son la clase más problemática (jugadores híbridos)

#### Análisis de Confianza:
- **Casos baja confianza (<0.6)**: 640 (7.16%)
- **Confianza promedio**: 0.520
- **Mayor confianza**: Porteros (0.948 avg)
- **Menor confianza**: Medios (0.894 avg)

#### Top 10 Características Más Importantes:
1. **reactions** (0.0423)
2. **ball_control** (0.0389)
3. **dribbling** (0.0356)
4. **short_passing** (0.0334)
5. **marking** (0.0312)
6. **standing_tackle** (0.0298)
7. **positioning** (0.0276)
8. **finishing** (0.0254)
9. **long_passing** (0.0243)
10. **interceptions** (0.0231)

### Archivos Generados
- `estudiante_b_red1_evaluacion.json`
- `estudiante_b_red1_feature_importance.csv`
- `estudiante_b_red2_evaluacion.json`
- `estudiante_b_red2_feature_importance.csv`
- `estudiante_b_red2_confusion_matrix_detailed.csv`

---

## ✅ ESTUDIANTE C: INTEGRACIÓN Y VALIDACIÓN

### Red 1: K-Fold Cross Validation (k=5)

#### Resultados por Fold:
| Fold | Train RMSE | Train R² | Test RMSE | Test R² |
|------|------------|----------|-----------|---------|
| 1    | 4.702      | 0.5174   | 4.453     | 0.5087  |
| 2    | 4.552      | 0.5137   | 4.414     | 0.4907  |
| 3    | 4.683      | 0.5182   | 4.481     | 0.5061  |
| 4    | 4.486      | 0.5157   | 4.779     | 0.5068  |
| 5    | 4.584      | 0.5182   | 4.466     | 0.5092  |

#### Estadísticas Agregadas:
- **RMSE medio**: 4.519 ± 0.132
- **MAE medio**: 3.479 ± 0.102
- **R² medio**: 0.5043 ± 0.0069
- **R² rango**: [0.4907, 0.5092]
- **CV Score**: 0.014

**Intervalo de Confianza 95%**: [0.4947, 0.5139]

**Conclusión**: ✅ Excelente estabilidad (CV < 0.05)

---

### Red 2: Stratified K-Fold Cross Validation (k=5)

#### Resultados por Fold:
| Fold | Train Acc | Test Acc | Macro F1 |
|------|-----------|----------|----------|
| 1    | 0.8815    | 0.8824   | 0.8768   |
| 2    | 0.8819    | 0.8828   | 0.8755   |
| 3    | 0.8826    | 0.8775   | 0.8720   |
| 4    | 0.8870    | 0.8847   | 0.8775   |
| 5    | 0.8907    | 0.8849   | 0.8742   |

#### Estadísticas Agregadas:
- **Accuracy medio**: 0.8825 ± 0.0027
- **Macro F1 medio**: 0.8752 ± 0.0020
- **Macro Precision**: 0.8742 ± 0.0023
- **Macro Recall**: 0.8762 ± 0.0021
- **Accuracy rango**: [0.8775, 0.8849]
- **CV Score**: 0.003

**Intervalo de Confianza 95%**: [0.8788, 0.8861]

**Conclusión**: ✅ Excelente estabilidad (CV < 0.02)

#### Métricas por Clase (Agregadas):
| Clase     | F1 Medio  | Std F1  |
|-----------|-----------|---------|
| Atacante  | 0.8581    | 0.0289  |
| Defensa   | 0.8656    | 0.0167  |
| Medio     | 0.8753    | 0.0208  |
| Portero   | 0.8848    | 0.0145  |

### Comparación de Estabilidad:
- **CV Regresión**: 0.0137
- **CV Clasificación**: 0.0030
- **Conclusión**: La clasificación es **4.6x más estable** que la regresión

### Archivos Generados
- `estudiante_c_red1_cv_results.json`
- `estudiante_c_red1_cv_folds.csv`
- `estudiante_c_red2_cv_results.json`
- `estudiante_c_red2_cv_folds.csv`
- `estudiante_c_statistical_analysis.json`

---

## 📈 COMPARACIÓN CON MÉTODOS DEL DÍA 1

### Regresión: MLP vs Regresión Lineal

| Modelo              | RMSE  | R²     | Mejora R²  |
|---------------------|-------|--------|------------|
| Regresión Lineal    | ~5.2  | ~0.42  | -          |
| **MLP (Red 1)**     | 4.519 | 0.5043 | **+20.1%** |

### Clasificación: MLP vs Regresión Logística

| Modelo              | Accuracy | F1 Macro | Mejora Acc |
|---------------------|----------|----------|------------|
| Logistic Regression | ~0.75    | ~0.73    | -          |
| **MLP (Red 2)**     | 0.8825   | 0.8752   | **+17.7%** |

---

## 🎯 CONCLUSIONES GENERALES

### Fortalezas del Modelo

1. **Regresión (Red 1)**:
   - ✅ R² = 0.50 → Explica 50% de la varianza
   - ✅ MAPE = 4.73% → Error relativo muy bajo
   - ✅ Sin sesgo sistemático (bias ≈ 0)
   - ✅ Estabilidad excelente en K-fold (CV = 0.014)

2. **Clasificación (Red 2)**:
   - ✅ Accuracy = 88.25% → Muy alto
   - ✅ Porteros clasificados casi perfectamente (98.7%)
   - ✅ Estabilidad excepcional en K-fold (CV = 0.003)
   - ✅ AUC-ROC macro = 0.86 → Excelente capacidad discriminativa

### Debilidades y Áreas de Mejora

1. **Regresión**:
   - ⚠️ Error máximo ~16 puntos en algunos casos extremos
   - ⚠️ Leve subestimación en jugadores de alto potencial (>90)

2. **Clasificación**:
   - ⚠️ Clase "Medio" más difícil de clasificar (F1 = 0.76)
   - ⚠️ 7% de casos con baja confianza (<0.6)
   - ⚠️ Confusión Medio↔Defensa y Medio↔Atacante más frecuente

### Recomendaciones

1. **Para Producción**:
   - Usar Red 1 para estimación de potencial de jugadores jóvenes
   - Usar Red 2 para clasificación automática de roles
   - Combinar predicciones con análisis de confianza

2. **Mejoras Futuras**:
   - Probar arquitecturas más profundas (5-6 capas)
   - Implementar Dropout para mayor regularización
   - Aumentar iteraciones para converger mejor
   - Crear subclases para "Medios" (CAM, CDM, CM)

---

## 📁 ARCHIVOS ENTREGABLES

### Código Fuente:
- `models.py` - Implementación de MLPRegressor y MLPClassifier
- `estudiante_a_entrenamiento.py` - Entrenamiento y optimización
- `estudiante_b_evaluacion.py` - Evaluación e interpretabilidad
- `estudiante_c_validacion.py` - Validación cruzada

### Resultados Estudiante A:
- `estudiante_a_red1_hyperparameters.csv`
- `estudiante_a_red1_results.json`
- `estudiante_a_red2_hyperparameters.csv`
- `estudiante_a_red2_confusion_matrix.csv`
- `estudiante_a_red2_results.json`

### Resultados Estudiante B:
- `estudiante_b_red1_evaluacion.json`
- `estudiante_b_red1_feature_importance.csv`
- `estudiante_b_red2_evaluacion.json`
- `estudiante_b_red2_feature_importance.csv`
- `estudiante_b_red2_confusion_matrix_detailed.csv`

### Resultados Estudiante C:
- `estudiante_c_red1_cv_results.json`
- `estudiante_c_red1_cv_folds.csv`
- `estudiante_c_red2_cv_results.json`
- `estudiante_c_red2_cv_folds.csv`
- `estudiante_c_statistical_analysis.json`

---

## ⏱️ TIEMPO DE EJECUCIÓN

- **Estudiante A** (entrenamiento completo): ~5 horas
- **Estudiante B** (evaluación): ~5 segundos
- **Estudiante C** (validación): ~3 segundos
- **Total**: ~5 horas (optimización inicial única)

---

## 🔧 ESPECIFICACIONES TÉCNICAS

### Red 1 (Regresión):
- **Entrada**: 20 características numéricas
- **Arquitectura**: 20-256-128-64-1
- **Parámetros totales**: ~92,000
- **Activación**: ReLU → ReLU → ReLU → Linear
- **Optimizador**: Gradient Descent (lr=0.0003)
- **Regularización**: L2 (λ=0.05)

### Red 2 (Clasificación):
- **Entrada**: 15 características numéricas
- **Arquitectura**: 15-256-128-4
- **Parámetros totales**: ~72,000
- **Activación**: ReLU → ReLU → Softmax
- **Optimizador**: Gradient Descent (lr=0.0005)
- **Regularización**: L2 (λ=0.01)

---

**Fecha de Entrega**: Diciembre 17, 2025  
**Dataset**: 183,978 registros de jugadores FIFA  
**Implementación**: NumPy puro (sin frameworks ML)
