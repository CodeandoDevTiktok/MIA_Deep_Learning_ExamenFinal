# Diapositivas: Predicción Espacio-Temporal de Deforestación en la Amazonía Peruana

## Guía para crear las diapositivas (15-20 slides, 25 minutos)

---

## Slide 1: Portada (30 seg)

**Título:** Predicción Espacio-Temporal de la Deforestación en la Amazonía Peruana mediante Redes Neuronales Convolucionales y Mecanismos de Atención

**Autores:**
- Integrante 1
- Integrante 2
- Integrante 3

**Curso:** MIA-203 Redes Neuronales y Aprendizaje Profundo
**Universidad Nacional de Ingeniería**
**Fecha:** Noviembre 2024

---

## Slide 2: Motivación (2 min)

**La Amazonía Peruana en Crisis**

- 60% del territorio nacional
- 2.7 millones de hectáreas perdidas (2001-2023)
- Tasa promedio: 120,000 ha/año
- Impactos:
  - Biodiversidad
  - Ciclo del carbono
  - Comunidades indígenas

**Imagen:** Mapa de Perú con regiones de deforestación resaltadas

---

## Slide 3: Problema y Objetivo (1.5 min)

**Problema:**
- Predicción precisa de deforestación es esencial para conservación
- Desafíos:
  - Variabilidad espacial
  - Dinámica temporal compleja
  - Grandes volúmenes de datos satelitales

**Objetivo:**
Desarrollar un modelo de aprendizaje profundo para predecir la deforestación futura usando series históricas de imágenes satelitales

---

## Slide 4: Contribuciones (1 min)

1. **ST-AttentionNet**: Arquitectura que integra CNN + ConvLSTM + Atención espacio-temporal

2. **Evaluación rigurosa**: Comparación con líneas base + estudios de ablación

3. **Interpretabilidad**: Grad-CAM + mapas de atención temporal

4. **Código abierto**: Pipeline reproducible para contexto peruano

---

## Slide 5: Trabajos Relacionados (2 min)

| Enfoque | Referencia | Limitación |
|---------|------------|------------|
| Índices espectrales | Hansen et al. (2013) | No captura dependencias temporales |
| U-Net | Ronneberger (2015) | Solo espacial |
| ConvLSTM | Shi et al. (2015) | Atención implícita |
| Attention | Vaswani et al. (2017) | No especializado para imágenes |

**Nuestro aporte:** Integración unificada de atención espacial, temporal y modelado secuencial

---

## Slide 6: Datos (2 min)

**Fuentes:**
- GeoBosques (MINAM): Mapas de pérdida de bosque 2001-2023
- MapBiomas Perú: Cobertura y uso del suelo 1985-2023

**Preprocesamiento:**
- 6 bandas espectrales (Blue, Green, Red, NIR, SWIR1, SWIR2)
- Parches 256×256, stride 128
- Secuencias de 5 años consecutivos

**Partición geográfica:**
- Train: Ucayali, Loreto (70%)
- Val: San Martín (15%)
- Test: Madre de Dios (15%)

---

## Slide 7: Arquitectura ST-AttentionNet - Visión General (2 min)

**Diagrama de arquitectura mostrando:**

```
Input: [B, T, C, H, W]
        ↓
┌─────────────────────────────┐
│   Encoder Espacial (U-Net)  │ → Features multi-escala + CBAM
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│        ConvLSTM             │ → Modelado temporal
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│   Atención Temporal         │ → Ponderación de timesteps
│   (Multi-cabeza)            │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│        Decoder              │ → Reconstrucción + Skip connections
└─────────────────────────────┘
        ↓
Output: [B, 2, H, W]
```

---

## Slide 8: Componente CBAM (1.5 min)

**Convolutional Block Attention Module**

**Atención por Canal:**
$$F' = \sigma(MLP(AvgPool(F)) + MLP(MaxPool(F))) \odot F$$

**Atención Espacial:**
$$F'' = \sigma(Conv([AvgPool(F'); MaxPool(F')])) \odot F'$$

**Diagrama:** Flujo de CBAM con visualización de mapas de atención

---

## Slide 9: ConvLSTM y Atención Temporal (1.5 min)

**ConvLSTM:** Captura dependencias temporales manteniendo estructura espacial

**Atención Multi-Cabeza:**
$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

- 8 cabezas de atención
- Permite conexiones directas entre timesteps distantes
- Genera pesos interpretativos

---

## Slide 10: Entrenamiento (1 min)

| Parámetro | Valor |
|-----------|-------|
| Optimizador | AdamW |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Épocas | 100 (early stopping) |
| Loss | CrossEntropy + Dice |
| Hardware | NVIDIA T4 (Colab) |
| Tiempo | ~4 horas |

---

## Slide 11: Resultados Principales (2 min)

**Tabla de resultados:**

| Modelo | IoU | F1 | Precisión | Recall |
|--------|-----|----|----|--------|
| Random Forest | 0.612 | 0.759 | 0.782 | 0.738 |
| U-Net | 0.701 | 0.824 | 0.841 | 0.808 |
| ConvLSTM | 0.723 | 0.839 | 0.852 | 0.827 |
| **ST-AttentionNet** | **0.781** | **0.877** | **0.889** | **0.866** |

**Mejoras:**
- +17% IoU vs Random Forest
- +8% IoU vs U-Net
- +6% IoU vs ConvLSTM

---

## Slide 12: Estudio de Ablación (1.5 min)

**Gráfico de barras mostrando impacto de cada componente:**

| Configuración | IoU | ΔIoU |
|--------------|-----|------|
| Completo | 0.781 | -- |
| Sin CBAM | 0.754 | -0.027 |
| Sin Atención Temporal | 0.741 | -0.040 |
| Sin ConvLSTM | 0.712 | -0.069 |

**Conclusión:** Todos los componentes contribuyen significativamente

---

## Slide 13: Predicciones Visuales (1.5 min)

**Grid de ejemplos:**

| Imagen | Ground Truth | Predicción | Diferencia |
|--------|--------------|------------|------------|

- Mostrar 3-4 ejemplos de predicciones
- Destacar casos de éxito
- Mostrar casos difíciles

---

## Slide 14: Interpretabilidad - Grad-CAM (1.5 min)

**Visualización de Grad-CAM:**

- El modelo se enfoca en:
  - Bordes de áreas deforestadas
  - Zonas de transición bosque/agricultura
  - Áreas de expansión

**Imagen:** Comparación imagen original vs superposición Grad-CAM

---

## Slide 15: Interpretabilidad - Atención Temporal (1.5 min)

**Gráfico de pesos de atención:**

| Timestep | Peso |
|----------|------|
| t-4 | 0.12 |
| t-3 | 0.15 |
| t-2 | 0.21 |
| t-1 | 0.25 |
| t | 0.27 |

**Insight:** Mayor peso a años recientes, pero considera tendencia histórica

---

## Slide 16: Análisis de Errores (1 min)

**Falsos Positivos:**
- Vegetación secundaria
- Bordes de ríos
- Sombras de nubes

**Falsos Negativos:**
- Deforestación < 1 ha
- Eventos muy recientes

---

## Slide 17: Limitaciones y Uso Responsable (1 min)

**Limitaciones:**
- Resolución temporal anual
- Generalización geográfica limitada
- Dependencia de datos sin nubes

**Sesgos potenciales:**
- Cobertura de nubes diferencial
- Errores en etiquetas

**Uso responsable:**
- No usar como única fuente para enforcement
- Requiere verificación de campo

---

## Slide 18: Costo Computacional (30 seg)

| Recurso | Valor |
|---------|-------|
| GPU | NVIDIA T4 |
| Memoria | 12 GB |
| Entrenamiento | 4 horas |
| Inferencia | 50 ms/imagen |
| CO2eq | ~0.5 kg |

---

## Slide 19: Conclusiones (1 min)

1. ST-AttentionNet logra **IoU 0.781**, superando líneas base

2. La combinación de **CNN + ConvLSTM + Atención** es efectiva

3. El modelo aprende representaciones **interpretables**

4. Código y datos disponibles para **reproducibilidad**

---

## Slide 20: Trabajo Futuro (30 seg)

1. Incorporar datos **SAR** para robustez ante nubes
2. **Predicción multi-paso** (varios años)
3. Integrar variables **socioeconómicas**
4. **Despliegue** como servicio de monitoreo

---

## Slide 21: Agradecimientos y Referencias (30 seg)

**Agradecimientos:**
- MINAM/GeoBosques por los datos
- MapBiomas Perú
- Google Colab por recursos computacionales

**Código disponible:**
github.com/[usuario]/deforestation-prediction-peru

---

## Notas para la presentación:

1. **Tiempo total:** 25 minutos de exposición + 5 minutos de preguntas
2. **Distribución sugerida:** Cada integrante presenta ~8 minutos
3. **Demo opcional:** Preparar notebook de inferencia como backup
4. **Video backup:** Grabar demo por si falla la conexión

### Distribución por integrante:

**Integrante 1 (Slides 1-7):** ~8 min
- Motivación, problema, contribuciones
- Trabajos relacionados, datos

**Integrante 2 (Slides 8-13):** ~8 min
- Arquitectura, entrenamiento
- Resultados principales, ablación

**Integrante 3 (Slides 14-21):** ~9 min
- Interpretabilidad, análisis de errores
- Limitaciones, conclusiones, trabajo futuro
