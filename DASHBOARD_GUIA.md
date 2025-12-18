# 🎯 Guía del Dashboard Web Integrado

## Sistema Completo: ML (Día 2) + EDO (Día 3)

---

## 🚀 Iniciar el Dashboard

### 1. Asegúrate de tener los modelos entrenados

Los archivos deben existir:
- `red1_regresion_trained.pkl`
- `red2_clasificacion_trained.pkl`

### 2. Inicia el servidor Flask

```bash
python app.py
```

### 3. Abre tu navegador

Navega a: **http://localhost:5000**

---

## 📊 Componentes del Dashboard

### Panel Izquierdo: Configuración

**Parámetros del Jugador:**

1. **Físico (0-100):** Velocidad, resistencia, fuerza
   - <50: Bajo nivel físico
   - 50-70: Nivel medio
   - 70-85: Buen nivel
   - >85: Elite

2. **Talento/Técnica (0-100):** Habilidad con el balón, pases, tiros
   - Influye directamente en la posición predicha
   - Alto valor → mejor envejecimiento

3. **Mentalidad (0-100):** Visión, posicionamiento, tácticas
   - Importante para GK y DEF
   - Mejora con experiencia en la simulación

4. **Rating Actual (0-100):** Nivel actual del jugador
   - Punto de partida para calcular crecimiento
   - Comparado con potential predicho

5. **Edad (15-45):** Edad actual del jugador
   - <22: Joven promesa (alto potencial de crecimiento)
   - 22-28: Prime time (pico de rendimiento)
   - 28-32: Madurez (cerca del pico)
   - >32: Veterano (mantenimiento/declive)

6. **Riesgo Lesiones (0-100):** Probabilidad/severidad de lesiones
   - 0: Sin lesiones
   - 1-30: Riesgo bajo
   - 31-60: Riesgo moderado
   - >60: Alto riesgo (impacto significativo)

7. **Régimen de Entrenamiento:**
   - ⚖️ **Balanceado (0.7, 0.7, 0.7):** Desarrollo equilibrado
   - 🔥 **Intensivo (0.9, 0.9, 0.8):** Máximo crecimiento rápido
   - ⚽ **Técnico (0.5, 0.9, 0.7):** Enfoque en habilidades
   - 💪 **Físico (0.9, 0.6, 0.6):** Potencia atlética
   - 🛡️ **Conservador (0.5, 0.5, 0.5):** Prevención, carrera larga

---

### Panel Derecho: Resultados

#### Sección 1: Predicciones ML

**Potencial Estimado (Red 1):**
- Predicción del rating máximo alcanzable
- Crecimiento esperado desde nivel actual
- Basado en atributos físicos, técnicos y mentales

**Posición Ideal (Red 2):**
- Posición recomendada: GK, DEF, MID, FWD
- Clasificación con red neuronal
- Probabilidades por cada posición

#### Sección 2: Métricas Clave

**⭐ Rating Pico:**
- Máximo rating alcanzado en la simulación
- Edad en la que ocurre
- Indicador de mejor momento de carrera

**📈 Desarrollo:**
- Diferencia entre rating inicial y pico
- Positivo: crecimiento
- Negativo: declive

**🏆 ¿Alcanza Potencial?:**
- ✓ Sí: Si alcanza ≥95% del potential predicho
- ✗ No: Si se queda por debajo
- Muestra % del potencial logrado

**⏱️ Carrera Útil:**
- Años hasta que rating < 70
- Indicador de longevidad
- Útil para planificación

#### Sección 3: Gráficas

**Probabilidades por Posición:**
- Gráfico de barras con % para cada posición
- Ayuda a visualizar versatilidad
- Verde = alta probabilidad

**Evolución del Rendimiento (10+ años):**
- **Línea Roja (R):** Overall Rating
  - Principal métrica de rendimiento
  - Muestra pico y declive
- **Línea Verde (F):** Físico
  - Sube hasta ~27 años (pico físico)
  - Decae después de 30
  - Afectado por lesiones
- **Línea Azul (T):** Técnica
  - Desarrollo más constante
  - Menor declive con la edad
- **Línea Amarilla (M):** Mentalidad
  - Mejora gradualmente (experiencia)
  - Casi no decae

#### Sección 4: Recomendaciones

El sistema genera automáticamente recomendaciones basadas en:

**Categorías:**

1. **🌱 Edad y Fase de Carrera:**
   - Joven promesa (<22): Maximizar entrenamiento
   - Prime (22-28): Ventana crítica
   - Madurez (28-32): Ajustar intensidad
   - Veterano (>32): Mantenimiento

2. **⚽ Desarrollo Técnico:**
   - Si técnica <70: Recomienda incrementar ET
   - Si técnica >85: Destaca ventaja de envejecimiento

3. **💪 Condición Física:**
   - Alto físico + edad >26: Alerta de desgaste
   - Recomienda reducir intensidad

4. **🏥 Gestión de Lesiones:**
   - Riesgo alto: Régimen conservador
   - Riesgo moderado: Balance

5. **📊 Potencial:**
   - No alcanzado: Aumentar intensidad
   - Alcanzado: Confirmación positiva

6. **📅 Planificación:**
   - Ventana óptima de transferencia
   - Momento ideal para contratos

7. **🚀 Crecimiento:**
   - Alto desarrollo: Inversión recomendada
   - Declive: Ajustar expectativas

---

## 🎮 Ejemplos de Uso

### Caso 1: Joven Promesa

**Inputs:**
- Edad: 20
- Físico: 78
- Técnica: 72
- Mentalidad: 65
- Rating: 72
- Lesiones: 10
- Régimen: Intensivo

**Resultados Esperados:**
- Potential: ~88-92
- Posición: FWD o MID
- Pico: ~90 @ 25 años
- Desarrollo: +18 puntos
- Recomendación: Maximizar entrenamiento hasta 24

### Caso 2: Portero Veterano

**Inputs:**
- Edad: 30
- Físico: 65
- Técnica: 82
- Mentalidad: 88
- Rating: 84
- Lesiones: 5
- Régimen: Conservador

**Resultados Esperados:**
- Potential: ~87
- Posición: GK
- Pico: ~87 @ 32 años
- Desarrollo: +3 puntos
- Recomendación: Mantenimiento, carrera larga hasta ~38

### Caso 3: Jugador con Lesiones

**Inputs:**
- Edad: 26
- Físico: 82
- Técnica: 78
- Mentalidad: 72
- Rating: 80
- Lesiones: 70 (alto riesgo)
- Régimen: Balanceado

**Resultados Esperados:**
- Potential: ~84
- Posición: DEF
- Pico: ~82 @ 28 años (reducido por lesiones)
- Desarrollo: +2 puntos (limitado)
- Recomendación: Programa preventivo urgente

---

## 🔧 Detalles Técnicos

### Backend (app.py)

**Flujo de Procesamiento:**

1. **Recepción de datos:** Flask recibe JSON con parámetros
2. **Predicción Red 1:** Regresión → Potential
3. **Predicción Red 2:** Clasificación → Posición + probabilidades
4. **Calibración EDO:** Mapeo potential → α, β, γ
5. **Simulación RK4:** Integración numérica de EDOs
6. **Postprocesamiento:** Downsampling para gráficas
7. **Respuesta JSON:** Todos los resultados al frontend

**Ajustes Dinámicos:**

```python
# Físico alto → mayor desgaste
if fisico > 80:
    params.slopeF = 0.15  # Mayor declive físico

# Técnica alta → menor declive
if tecnica > 80:
    params.slopeT = 0.04  # Envejece mejor

# Lesiones → aumenta decaimiento
risk_factor = lesiones / 100.0
params.slopeF += 0.20 * risk_factor
params.betaF0 += 0.05 * risk_factor
```

### Frontend (dashboard.html)

**Tecnologías:**
- Bootstrap 5: Diseño responsive
- Chart.js: Gráficos interactivos
- Font Awesome: Iconos
- JavaScript vanilla: Lógica

**Características:**
- Actualización en tiempo real (sliders)
- Gráficos interactivos (hover)
- Cálculo de métricas en cliente
- Generación dinámica de recomendaciones

---

## 📈 Interpretación de Resultados

### Rating Pico

| Valor | Interpretación |
|-------|----------------|
| <70 | Nivel bajo/amateur |
| 70-75 | Profesional promedio |
| 76-82 | Buen jugador |
| 83-88 | Muy bueno/internacional |
| 89-94 | Elite/estrella |
| >95 | Leyenda |

### Desarrollo

| Valor | Significado |
|-------|-------------|
| +15+ | Enorme potencial |
| +10 a +15 | Alto desarrollo |
| +5 a +10 | Buen crecimiento |
| 0 a +5 | Crecimiento limitado |
| <0 | En declive |

### Carrera Útil

| Años | Interpretación |
|------|----------------|
| <5 | Carrera corta |
| 5-10 | Normal |
| 10-15 | Larga |
| >15 | Muy larga (típico GK) |

---

## 🐛 Solución de Problemas

### Error: "No se pudieron cargar los modelos"

**Solución:**
1. Verifica que existan:
   - `red1_regresion_trained.pkl`
   - `red2_clasificacion_trained.pkl`
2. Re-entrena las redes con los scripts del Día 2

### Gráficas no aparecen

**Solución:**
- Verifica la consola del navegador (F12)
- Asegúrate de que `result.simulation` existe en la respuesta

### Predicciones extrañas

**Posibles causas:**
- Modelos entrenados con datos diferentes
- Normalización inconsistente
- Parámetros extremos (ej: edad = 45, físico = 100)

### Simulación muy lenta

**Solución:**
- El downsampling ya reduce puntos (step=4)
- Si persiste, aumentar `step` en app.py

---

## 💡 Mejores Prácticas

### Para Análisis Realista

1. **Coherencia de Inputs:**
   - Joven (18-22) → Físico alto, técnica en desarrollo
   - Veterano (30+) → Técnica alta, físico moderado

2. **Régimen apropiado:**
   - Jóvenes: Intensivo
   - Prime: Balanceado
   - Veteranos: Conservador

3. **Lesiones realistas:**
   - Delanteros rápidos: Riesgo moderado (30-40)
   - Defensas físicos: Riesgo alto (50-60)
   - Porteros: Riesgo bajo (10-20)

### Para Experimentación

- **Variar un parámetro a la vez** para ver impacto
- **Comparar regímenes** con mismo jugador
- **Analizar extremos** (edad 18 vs 35)

---

## 📚 Referencias

- **Día 2:** Redes neuronales en `Dia2/`
- **Día 3:** EDOs y simulación en `Dia3_EDO___sIMULACION/`
- **Reporte Técnico:** `Dia3_EDO___sIMULACION/REPORTE_TECNICO.md`

---

**¡Dashboard completo y listo para uso!** 🚀⚽

Para más información, consulta `REPORTE_TECNICO.md` o `README.md`.
