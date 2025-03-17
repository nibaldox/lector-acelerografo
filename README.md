# Visor de Acelerógrafos

Aplicación web para visualizar y analizar datos de acelerógrafos, desarrollada con Streamlit y Python.

## Características

- **Visualización de Datos**
  - Vista individual de registros
  - Comparación múltiple de registros
  - Control de zoom y navegación temporal
  - Escalas y unidades calibradas
  - Gráficos con fondos transparentes y diseño consistente
  - Anotaciones de valores máximos
  - Visualización del vector suma

- **Análisis Espectral**
  - Transformada de Fourier (FFT)
  - Espectrogramas
  - Configuración de ventanas y parámetros
  - Visualización en escala logarítmica
  - Análisis de respuesta combinada (SRSS y Porcentual)

- **Filtros Interactivos**
  - Filtros pasa bajos/altos
  - Filtros pasa banda
  - Visualización de respuesta en frecuencia
  - Ajuste de parámetros en tiempo real

- **Detección de Eventos**
  - Algoritmo STA/LTA
  - Detección de picos
  - Análisis de características de eventos
  - Visualización y marcadores

- **Exportación de Datos**
  - Datos crudos (CSV, Excel, JSON)
  - Resultados de análisis
  - Gráficos (HTML, PNG, JSON)

- **Soporte para Múltiples Formatos**
  - MS/SS (formato propietario de acelerógrafos)
  - SAC (Seismic Analysis Code)
  - miniSEED
  - SEG-Y
  - ASCII (txt, csv, dat, asc)

## Requisitos

```bash
streamlit
numpy
pandas
plotly
scipy
obspy (opcional, para formatos miniSEED y SEG-Y)
```

## Instalación

1. Clonar el repositorio:

   ```bash
   git clone <url-repositorio>
   cd visor-acelerografos
   ```

2. Crear y activar entorno virtual (opcional):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. Ejecutar la aplicación:

   ```bash
   streamlit run app.py
   ```

2. Acceder a través del navegador:

   ```bash
   http://localhost:8501
   ```

3. Cargar datos sísmicos:

   - **Opción 1: Archivos Individuales**
     - Seleccione los archivos en cualquiera de los formatos soportados
     - Para archivos MS/SS, asegúrese de subir ambos archivos para cada registro

   - **Opción 2: Carpeta Completa (ZIP)**
     - Comprima la carpeta que contiene los archivos en un archivo ZIP
     - Suba el archivo ZIP mediante el selector correspondiente
     - La aplicación procesará automáticamente toda la estructura de directorios
     - Se mantendrá la jerarquía original de archivos y subcarpetas

4. Seleccione los registros a visualizar y analizar

## Estructura de Archivos y Módulos

### Módulos Principales

#### `app.py` - Aplicación Principal

- Interfaz gráfica desarrollada con Streamlit
- Visualización interactiva de señales
- Integración de todos los módulos de análisis
- Navegación mediante pestañas para diferentes funcionalidades
- Personalización de parámetros en tiempo real

#### `format_readers.py` - Lectura de Múltiples Formatos

- Arquitectura modular con clase base `BaseReader`
- Lectores específicos para cada formato:
  - `MSReader`: Formato propietario MS/SS
  - `SACReader`: Formato SAC (Seismic Analysis Code)
  - `MiniSEEDReader`: Formato miniSEED
  - `SEGYReader`: Formato SEG-Y
  - `ASCIIReader`: Formatos de texto (CSV, TXT, DAT, ASC)
- Extracción de metadata y parámetros del sensor
- Manejo de múltiples canales (E, N, Z)
- Detección automática del formato basada en extensión

#### `signal_processor.py` - Procesamiento de Señales

- Cálculo de velocidad y desplazamiento a partir de aceleración
- Filtrado de señales
- Cálculo del vector suma
- Cálculo de espectros de respuesta
- Análisis estadístico de señales

#### `report_generator.py` - Generación de Informes

- Creación de informes en múltiples formatos (PDF, HTML, DOCX)
- Inclusión de gráficos y resultados de análisis
- Personalización de plantillas
- Exportación de datos y metadatos

#### `fft_processor.py` - Análisis Espectral

- Transformada rápida de Fourier (FFT)
- Cálculo de espectrogramas
- Ventanas configurables (Hann, Hamming, Blackman)
- Visualización en escala logarítmica
- Promediado de segmentos para reducción de ruido

#### `filters.py` - Procesamiento de Señales

- Filtros digitales Butterworth
- Configuración de filtros paso bajo/alto/banda
- Visualización de respuesta en frecuencia
- Orden del filtro ajustable
- Frecuencias de corte configurables

#### `event_detector.py` - Detección de Eventos

- Algoritmo STA/LTA para detección automática
- Detección basada en umbral y picos
- Cálculo de características de eventos
- Ventanas de análisis configurables
- Métricas de eventos (amplitud, RMS, energía)

#### `data_exporter.py` - Exportación de Datos

- Exportación en múltiples formatos (CSV, Excel, JSON)
- Guardado de gráficos (HTML, PNG)
- Exportación de resultados de análisis
- Métricas y características de eventos
- Formato de datos configurable

## Herramientas y Funcionalidades

### 1. Visualización de Datos

- **Vista Individual**
  - Visualización de componentes E, N, Z
  - Escala automática y manual
  - Control de zoom y navegación temporal
  - Información de metadata y calibración
  - Rangeslider para navegación rápida
  - Anotaciones de valores máximos
  - Gráficos con fondos transparentes
  - Visualización del vector suma

- **Vista Comparativa**
  - Comparación de múltiples registros
  - Alineación temporal de señales
  - Normalización opcional de amplitudes
  - Leyendas y etiquetas personalizables
  - Selección múltiple de componentes

### 2. Análisis Espectral

- **Transformada de Fourier**
  - Selección de ventana de análisis
  - Configuración de segmentos y superposición
  - Visualización de magnitud y fase
  - Detección de frecuencias dominantes
  - Escala logarítmica configurable

- **Espectrogramas**
  - Resolución tiempo-frecuencia ajustable
  - Paleta de colores personalizable
  - Control de superposición de ventanas
  - Visualización de energía espectral
  - Zoom y navegación interactiva

### 3. Filtros Digitales

- **Tipos de Filtros**
  - Paso bajo: Elimina altas frecuencias
  - Paso alto: Elimina bajas frecuencias
  - Paso banda: Selecciona rango de frecuencias
  
- **Configuración**
  - Frecuencias de corte ajustables
  - Orden del filtro variable
  - Visualización de respuesta en frecuencia
  - Preview en tiempo real
  - Comparación antes/después

### 4. Detección de Eventos

- **Método STA/LTA**
  - Ventanas corta/larga configurables
  - Ratio de disparo ajustable
  - Visualización del ratio STA/LTA
  - Marcadores de eventos detectados
  - Estadísticas de detección

- **Detección de Picos**
  - Umbral basado en desviación estándar
  - Distancia mínima entre eventos
  - Características de picos detectados
  - Anotaciones automáticas
  - Filtrado de falsos positivos

### 5. Exportación de Datos

- **Formatos Soportados**
  - CSV: Datos crudos y procesados
  - Excel: Hojas múltiples con metadata
  - JSON: Estructura jerárquica completa
  - PNG/HTML: Gráficos interactivos
  - PDF: Reportes de análisis

- **Contenido Exportable**
  - Series temporales
  - Espectros de frecuencia
  - Resultados de filtrado
  - Eventos detectados
  - Características y métricas

## Contribuir

1. Fork del repositorio
2. Crear rama de características (`git checkout -b feature/nueva-caracteristica`)
3. Commit cambios (`git commit -am 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crear Pull Request

## Licencia

MIT License - ver archivo [LICENSE](LICENSE) para más detalles.
