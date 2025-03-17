import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import os
import tempfile
import shutil
import zipfile
import io
from scipy.fft import fft, fftfreq
import base64
from signal_processor import SignalProcessor
from report_generator import ReportGenerator
from format_readers import get_reader_for_file

# Función para obtener unidades según el tipo de dato
def get_units_for_data_type(data_type):
    """
    Devuelve las unidades correspondientes al tipo de dato
    
    Args:
        data_type (str): Tipo de dato (aceleracion, velocidad, desplazamiento)
        
    Returns:
        str: Unidades correspondientes
    """
    if data_type == "aceleracion":
        return "Aceleración (m/s²)"
    elif data_type == "velocidad":
        return "Velocidad (m/s)"
    elif data_type == "desplazamiento":
        return "Desplazamiento (m)"
    else:
        return "Valor"

# Configuración inicial de la página
st.set_page_config(
    page_title="Visor de Acelerógrafos",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'Visor de Acelerógrafos - Desarrollado para análisis sísmico'
    }
)

# Estilos personalizados
st.markdown("""
    <script>
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        localStorage.setItem('theme', prefersDark ? 'dark' : 'light');
    </script>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    :root {
        --background-color: white;
        --text-color: #262730;
        --secondary-text-color: #4b5563;
        --container-bg: #f0f2f6;
        --metric-bg: #f8f9fa;
        --grid-color: rgba(128, 128, 128, 0.2);
        --zero-line-color: rgba(128, 128, 128, 0.4);
        --plot-bg: rgba(255, 255, 255, 0);
        --legend-bg: rgba(255, 255, 255, 0.8);
        --hover-bg: white;
        --modebar-bg: rgba(255, 255, 255, 0.9);
        --modebar-color: rgba(0, 0, 0, 0.6);
        --modebar-active: rgba(0, 0, 0, 0.9);
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #0e1117;
            --text-color: #e0e0e0;
            --secondary-text-color: #b0b0b0;
            --container-bg: #1e1e1e;
            --metric-bg: #1e1e1e;
            --grid-color: rgba(128, 128, 128, 0.2);
            --zero-line-color: rgba(128, 128, 128, 0.4);
            --plot-bg: rgba(0, 0, 0, 0);
            --legend-bg: rgba(30, 30, 30, 0.5);
            --hover-bg: #1e1e1e;
            --modebar-bg: rgba(30, 30, 30, 0.5);
            --modebar-color: rgba(200, 200, 200, 0.6);
            --modebar-active: rgba(255, 255, 255, 0.9);
        }
    }
    
    .main {
        padding: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #0078D4;
        color: white;
    }
    .stMetric {
        background-color: var(--metric-bg);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        color: var(--text-color);
    }
    .stMetric label {
        color: var(--text-color) !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: var(--text-color) !important;
        font-weight: bold;
    }
    .info-container, div.stMarkdown p, div.stMarkdown div {
        background-color: var(--container-bg) !important;
        color: var(--text-color) !important;
    }
    /* Forzar el color de fondo en todos los elementos */
    .stMarkdown * {
        background-color: var(--container-bg) !important;
    }
    /* Asegurar que todos los contenedores tengan el mismo estilo */
    div.element-container, div.block-container {
        color: var(--text-color);
    }
    /* Estilo para todos los contenedores de Streamlit */
    div.stMarkdown, div.stText, div.stDataFrame, div.stTable, div.stPlotly {
        background-color: var(--container-bg);
        border-radius: 8px;
        padding: 0.5rem;
        margin-bottom: 1rem;
        color: var(--text-color);
    }
    /* Asegurar que los gráficos tengan fondo transparente */
    .js-plotly-plot .plotly .main-svg {
        background-color: rgba(0, 0, 0, 0) !important;
    }
    /* Estilo para los selectores y controles */
    .stSelectbox, .stSlider, .stRadio, .stCheckbox {
        background-color: var(--container-bg);
        border-radius: 8px;
        padding: 0.5rem;
        margin-bottom: 1rem;
        color: var(--text-color);
    }
    </style>
""", unsafe_allow_html=True)

def get_ss_file(ms_file_path):
    """Obtiene el archivo .ss correspondiente al archivo .ms"""
    return str(ms_file_path).replace('.ms', '.ss')

def load_metadata(ss_file_path):
    """Lee los metadatos del archivo .ss"""
    metadata = {}
    try:
        with open(ss_file_path, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    metadata[key.strip('"')] = value.strip('"')
    except Exception as e:
        st.error(f"Error al leer el archivo .ss: {str(e)}")
    return metadata

def main():
    st.title("Visor de Acelerógrafos")
    
    # Sidebar para configuración
    st.sidebar.header("Configuración")
    
    # Crear directorio temporal para archivos subidos
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Limpiar directorio temporal al inicio
    for file in upload_dir.glob("*"):
        if file.is_file():
            file.unlink()
        elif file.is_dir():
            shutil.rmtree(file)
    
    # Selector de método de carga
    st.sidebar.markdown("### Cargar Datos")
    upload_option = st.sidebar.radio("Seleccionar método de carga:", 
                                   ["Archivos Individuales", "Carpeta Completa (ZIP)"])
    
    # Procesar archivos subidos
    data_files = []
    
    if upload_option == "Carpeta Completa (ZIP)":
        zip_file = st.sidebar.file_uploader(
            "Subir carpeta comprimida (ZIP)", 
            type=["zip"],
            help="Comprima la carpeta conteniendo los archivos antes de subir"
        )
        
        if zip_file:
            # Crear directorio temporal para extracción
            extract_dir = upload_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)
            
            # Mostrar mensaje de progreso
            with st.sidebar.status("Extrayendo archivos..."):
                # Extraer contenido del ZIP
                with zipfile.ZipFile(io.BytesIO(zip_file.read())) as z:
                    z.extractall(extract_dir)
                st.sidebar.success(f"ZIP extraído correctamente: {len(z.namelist())} archivos")
            
            # Buscar archivos en la estructura extraída
            supported_extensions = [".ms", ".sac", ".mseed", ".miniseed", ".sgy", ".segy", ".txt", ".csv", ".dat", ".asc"]
            data_paths = []
            
            for ext in supported_extensions:
                data_paths.extend(list(extract_dir.rglob(f"*{ext}")))
            
            # Para archivos .ms, buscar sus correspondientes .ss
            ms_paths = list(extract_dir.rglob("*.ms"))
            for ms_path in ms_paths:
                ss_path = ms_path.with_suffix(".ss")
                if ss_path.exists():
                    data_files.append((str(ms_path), str(ss_path)))
            
            # Para otros formatos, no necesitan archivo de metadatos separado
            for path in data_paths:
                if path.suffix != ".ms" and path.suffix != ".ss":  # Evitar duplicados
                    data_files.append((str(path), None))
            
            if data_files:
                st.sidebar.success(f"Se encontraron {len(data_files)} archivos de datos sísmicos")
            else:
                st.sidebar.warning("No se encontraron archivos de datos sísmicos en el ZIP")
    else:
        # Carga individual de archivos
        supported_extensions = ["ms", "ss", "sac", "mseed", "miniseed", "sgy", "segy", "txt", "csv", "dat", "asc"]
        
        uploaded_files = st.sidebar.file_uploader(
            "Subir archivos de datos sísmicos",
            accept_multiple_files=True,
            type=supported_extensions,
            help="Formatos soportados: MS/SS, SAC, miniSEED, SEG-Y, ASCII (txt, csv, dat, asc)"
        )
        
        if uploaded_files:
            # Guardar archivos subidos en el directorio temporal
            for uploaded_file in uploaded_files:
                file_path = upload_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Buscar pares de archivos .ms y .ss
            ms_paths = list(upload_dir.glob("*.ms"))
            for ms_path in ms_paths:
                ss_path = ms_path.with_suffix(".ss")
                if ss_path.exists():
                    data_files.append((str(ms_path), str(ss_path)))
            
            # Agregar otros formatos de archivo
            for ext in [".sac", ".mseed", ".miniseed", ".sgy", ".segy", ".txt", ".csv", ".dat", ".asc"]:
                paths = list(upload_dir.glob(f"*{ext}"))
                for path in paths:
                    data_files.append((str(path), None))
    
    if not data_files:
        st.info("Por favor, sube archivos de datos sísmicos para visualizar. Se soportan los siguientes formatos: MS/SS, SAC, miniSEED, SEG-Y, ASCII (txt, csv, dat, asc).")
        return
        
    # Selector de archivos encontrados
    selected_files = st.multiselect(
        "Seleccionar registros",
        options=data_files,
        format_func=lambda x: os.path.basename(x[0]),
        default=[data_files[0]] if data_files else None
    )
    
    if not selected_files:
        st.info("Por favor, selecciona al menos un registro para visualizar")
        return

    try:
        # Procesar cada archivo seleccionado
        all_data = []
        for file_path, metadata_path in selected_files:
            try:
                # Obtener el lector adecuado para el tipo de archivo
                reader = get_reader_for_file(file_path)
                data = reader.read_data()
                
                # Asignar nombre del archivo
                data['name'] = os.path.basename(file_path)
                
                # Procesar datos para obtener velocidad y desplazamiento
                sampling_rate = float(data['metadata'].get('sampling_rate', 100))
                signal_processor = SignalProcessor(sampling_rate)
                
                # Procesar cada componente
                for component in data['components']:
                    processed_data = signal_processor.process_acceleration_data(
                        data[component], 
                        data['time']
                    )
                    # Guardar los datos originales como aceleración
                    data[f'{component}_aceleracion'] = data[component]
                    data[f'{component}_velocidad'] = processed_data['velocity']
                    data[f'{component}_desplazamiento'] = processed_data['displacement']
                
                # Calcular el vector suma (magnitud resultante) para cada tipo de dato
                if len(data['components']) > 1:  # Solo si hay múltiples componentes
                    for data_type in ['aceleracion', 'velocidad', 'desplazamiento']:
                        # Crear un array para almacenar la suma de cuadrados
                        sum_squares = np.zeros_like(data['time'])
                        
                        # Sumar los cuadrados de cada componente
                        for component in data['components']:
                            sum_squares += np.power(data[f'{component}_{data_type}'], 2)
                        
                        # Calcular la raíz cuadrada
                        data[f'vector_suma_{data_type}'] = np.sqrt(sum_squares)
                
                all_data.append(data)
                
            except Exception as e:
                st.error(f"Error al procesar el archivo {os.path.basename(file_path)}: {str(e)}")
                continue
        
        if not all_data:
            st.error("No se pudo procesar ninguno de los archivos seleccionados.")
            return
        
        # Crear pestañas para diferentes vistas
        tabs = st.tabs([
            "📊 Vista Individual",
            "📈 Análisis Espectral",
            "🔍 Filtros",
            "⚡ Detección de Eventos",
            "💾 Exportar",
            "📉 Espectro de Respuesta",
            "🔄 Análisis Multicomponente",
            "📋 Reportes Automáticos"
        ])
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = tabs
        
        with tab1:
            st.markdown("""
                <div class='info-container'>
                    <h2>Vista Individual</h2>
                    <p>Visualiza los registros de aceleración, velocidad y desplazamiento para cada componente.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Selector de registro
            selected_data_index = st.selectbox(
                "Seleccionar registro", 
                range(len(all_data)), 
                format_func=lambda i: all_data[i]['name']
            )
            data = all_data[selected_data_index]
            
            # Mostrar metadatos
            with st.expander("Metadatos", expanded=True):
                metadata_df = pd.DataFrame({
                    'Campo': list(data['metadata'].keys()),
                    'Valor': list(data['metadata'].values())
                })
                st.dataframe(metadata_df, use_container_width=True)
            
            # Mostrar información relevante con mejor diseño
            st.markdown("<h4 style='margin: 1rem 0;'>Información del Registro</h4>", unsafe_allow_html=True)
            cols = st.columns(4)
            with cols[0]:
                st.metric(
                    "Frecuencia de muestreo",
                    f"{data['metadata'].get('sampling_rate', 'N/A')} Hz",
                    help="Frecuencia de muestreo del registro"
                )
            with cols[1]:
                st.metric(
                    "Sensor",
                    data['metadata'].get('sensor_name', 'N/A'),
                    help="Nombre del sensor utilizado"
                )
            with cols[2]:
                st.metric(
                    "Unidades",
                    data['metadata'].get('unit', 'm/s/s'),
                    help="Unidades de medición"
                )
            with cols[3]:
                st.metric(
                    "Duración",
                    f"{data['time'][-1]:.2f} s",
                    help="Duración total del registro"
                )
                
            # Agregar controles de zoom sincronizados
            st.markdown("<h4 style='margin: 1rem 0;'>Controles de Visualización</h4>", unsafe_allow_html=True)
            zoom_cols = st.columns(2)
            with zoom_cols[0]:
                zoom_start = st.number_input(
                    "Tiempo inicial (s)",
                    0.0,
                    float(data['time'][-1]),
                    0.0,
                    help="Selecciona el tiempo inicial para el zoom"
                )
            with zoom_cols[1]:
                zoom_end = st.number_input(
                    "Tiempo final (s)",
                    zoom_start,
                    float(data['time'][-1]),
                    float(data['time'][-1]),
                    help="Selecciona el tiempo final para el zoom"
                )
            
            # Selector de unidades de visualización
            st.sidebar.subheader("Opciones de Visualización")
            display_unit = st.sidebar.radio(
                "Seleccionar unidad:",
                ["m/s²", "g (9.81 m/s²)"],
                key="display_unit_tab1"
            )
            
            # Selector de tipo de datos a visualizar
            data_type = st.sidebar.radio(
                "Tipo de datos:",
                ["Aceleración", "Velocidad", "Desplazamiento"],
                key="data_type_tab1"
            )
            
            # Factor de conversión según la unidad seleccionada
            conversion_factor = 1.0 if display_unit == "m/s²" else 1.0/9.81
            
            # Etiquetas de unidades según el tipo de datos
            if data_type == "Aceleración":
                unit_label = "m/s²" if display_unit == "m/s²" else "g"
                title_prefix = "Aceleración"
                data_field_suffix = "aceleracion"
            elif data_type == "Velocidad":
                unit_label = "m/s"
                title_prefix = "Velocidad"
                data_field_suffix = "velocidad"
            else:  # Desplazamiento
                unit_label = "m"
                title_prefix = "Desplazamiento"
                data_field_suffix = "desplazamiento"
            
            # Configuración común para todos los gráficos
            graph_config = {
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "modeBarButtonsToAdd": [
                    "drawopenpath",
                    "eraseshape",
                    "zoomIn2d",
                    "zoomOut2d",
                    "autoScale2d"
                ],
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "grafico",
                    "height": 800,
                    "width": 1200,
                    "scale": 2
                }
            }

            # Configuración común del layout
            layout_config = {
                "height": 350,
                "margin": dict(l=50, r=20, t=40, b=30),
                "showlegend": True,
                "legend": dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor="rgba(255, 255, 255, 0.5)",
                    bordercolor="rgba(128, 128, 128, 0.3)",
                    borderwidth=1
                ),
                "xaxis": dict(
                    range=[zoom_start, zoom_end],
                    rangeslider=dict(visible=True, thickness=0.1),
                    title="Tiempo (s)",
                    gridcolor="rgba(128, 128, 128, 0.2)",
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor="rgba(0, 0, 0, 0.3)",
                    zerolinewidth=1
                ),
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)"
            }

            # Colores personalizados para cada componente
            colors = {
                "N": "#1f77b4",    # Azul
                "E": "#2ca02c",   # Verde
                "Z": "#d62728",     # Rojo
                "vector_suma": "#9467bd"       # Morado
            }

            # Crear gráficos para cada componente con la nueva configuración
            st.markdown("""
                <div class='info-container'>
                    <h3 style='margin: 0;'>Visualización de Componentes</h3>
                    <p style='margin: 0.5rem 0 0 0;'>Gráficos detallados de cada componente del registro sísmico.</p>
                </div>
            """, unsafe_allow_html=True)

            # Crear gráficos para cada componente disponible
            for component in data['components']:
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=data['time'],
                    y=data[f'{component}_{data_field_suffix}'] * conversion_factor,
                    mode='lines',
                    name=component,
                    line=dict(
                        color=colors.get(component, "#1f77b4"),
                        width=2,
                        shape='linear'
                    ),
                    hovertemplate="<b>Tiempo:</b> %{x:.2f}s<br><b>Valor:</b> %{y:.3f} " + unit_label
                ))
                
                # Configuración específica para el componente
                max_val = abs(data[f'{component}_{data_field_suffix}']).max() * conversion_factor * 1.2
                layout_comp = layout_config.copy()
                layout_comp.update({
                    "title": dict(
                        text=f"<b>{title_prefix} - Componente {component}</b>",
                        x=0.5,
                        xanchor='center',
                        font=dict(size=16)
                    ),
                    "yaxis": dict(
                        title=dict(
                            text=f"{title_prefix} ({unit_label})",
                            standoff=10
                        ),
                        range=[-max_val, max_val],
                        gridcolor="rgba(128, 128, 128, 0.2)",
                        showgrid=True,
                        zeroline=True,
                        zerolinecolor="rgba(0, 0, 0, 0.3)",
                        zerolinewidth=1
                    )
                })
                fig_comp.update_layout(**layout_comp)
                
                # Agregar anotaciones para valores máximos y mínimos
                max_idx = np.argmax(abs(data[f'{component}_{data_field_suffix}']))
                max_time = data['time'][max_idx]
                max_value = data[f'{component}_{data_field_suffix}'][max_idx] * conversion_factor
                
                fig_comp.add_annotation(
                    x=max_time,
                    y=max_value,
                    text=f"Max: {max_value:.2f} {unit_label}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=colors.get(component, "#1f77b4"),
                    bgcolor="rgba(0, 0, 0, 0)",
                    bordercolor=colors.get(component, "#1f77b4"),
                    borderwidth=1,
                    borderpad=4,
                    font=dict(size=10, color=colors.get(component, "#1f77b4"))
                )
                
                st.plotly_chart(fig_comp, use_container_width=True, config=graph_config)
            
            # Vector Suma (si hay más de una componente)
            if len(data['components']) > 1:
                fig_suma = go.Figure()
                fig_suma.add_trace(go.Scatter(
                    x=data['time'],
                    y=data[f'vector_suma_{data_field_suffix}'] * conversion_factor,
                    mode='lines',
                    name="Vector Suma",
                    line=dict(
                        color=colors["vector_suma"],
                        width=2,
                        shape='linear'
                    ),
                    hovertemplate="<b>Tiempo:</b> %{x:.2f}s<br><b>Valor:</b> %{y:.3f} " + unit_label
                ))
                
                max_val_suma = data[f'vector_suma_{data_field_suffix}'].max() * conversion_factor * 1.2
                # Encontrar el tiempo del valor máximo para la anotación
                max_idx_suma = np.argmax(data[f'vector_suma_{data_field_suffix}'])
                max_time_suma = data['time'][max_idx_suma]
                max_value_suma = data[f'vector_suma_{data_field_suffix}'][max_idx_suma] * conversion_factor
                
                fig_suma.update_layout(
                    title=dict(
                        text=f"<b>Magnitud Resultante ({title_prefix})</b>",
                        x=0.5,
                        xanchor='center',
                        font=dict(size=16)
                    ),
                    **layout_config,
                    yaxis=dict(
                        title=dict(
                            text=f"{title_prefix} ({unit_label})",
                            standoff=10
                        ),
                        range=[0, max_val_suma],
                        gridcolor="rgba(128, 128, 128, 0.2)",
                        showgrid=True,
                        zeroline=True,
                        zerolinecolor="rgba(0, 0, 0, 0.3)",
                        zerolinewidth=1
                    )
                )
                
                # Agregar anotación para el valor máximo
                fig_suma.add_annotation(
                    x=max_time_suma,
                    y=max_value_suma,
                    text=f"Max: {max_value_suma:.2f} {unit_label}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=colors["vector_suma"],
                    bgcolor="rgba(0, 0, 0, 0)",
                    bordercolor=colors["vector_suma"],
                    borderwidth=1,
                    borderpad=4,
                    font=dict(size=10, color=colors["vector_suma"])
                )
                
                st.plotly_chart(fig_suma, use_container_width=True, config=graph_config)
            
            # Opciones adicionales para análisis
            st.subheader("Opciones adicionales para análisis")
            
            # Selector de componentes para el gráfico individual
            available_components = data['components'] + ['vector_suma'] if len(data['components']) > 1 else data['components']
            components = st.multiselect(
                "Seleccionar componentes para visualizar juntos:",
                available_components,
                default=data['components'],
                key="components_tab1"
            )
            
            # Crear gráfico individual con todos los componentes seleccionados
            if components:
                fig1 = go.Figure()
                for component in components:
                    fig1.add_trace(go.Scatter(
                        x=data['time'],
                        y=data[f'{component}_{data_field_suffix}'] * conversion_factor,
                        mode='lines',
                        name=component,
                        line=dict(color=colors.get(component, "#1f77b4"))
                    ))
        
                # Calcular el rango del eje Y basado en el máximo valor absoluto
                max_vals = []
                for component in components:
                    if component != 'vector_suma':
                        max_vals.append(abs(data[f'{component}_{data_field_suffix}']).max() * conversion_factor)
                    else:
                        max_vals.append(data[f'{component}_{data_field_suffix}'].max() * conversion_factor)
                        
                y_max = max(max_vals) * 1.2  # Ampliar el valor máximo para el rango

                # Configuración del gráfico individual
                fig1.update_layout(
                    title=dict(
                        text=f"<b>Registro de {title_prefix} - {data['name']}</b>",
                        x=0.5,
                        xanchor='center',
                        font=dict(size=16)
                    ),
                    xaxis=dict(
                        rangeslider=dict(visible=True, thickness=0.1),
                        type="linear",
                        range=[zoom_start, zoom_end],
                        title="Tiempo (s)",
                        gridcolor="rgba(128, 128, 128, 0.2)",
                        showgrid=True,
                        zeroline=True,
                        zerolinecolor="rgba(0, 0, 0, 0.3)",
                        zerolinewidth=1
                    ),
                    yaxis=dict(
                        title=dict(
                            text=f"{title_prefix} ({unit_label})",
                            standoff=10
                        ),
                        exponentformat='e',
                        showexponent='all',
                        tickformat='.2e',
                        range=[-y_max, y_max],  # Rango simétrico
                        gridcolor="rgba(128, 128, 128, 0.2)",
                        showgrid=True,
                        zeroline=True,
                        zerolinecolor="rgba(0, 0, 0, 0.3)",
                        zerolinewidth=1
                    ),
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99,
                        bgcolor="rgba(255, 255, 255, 0.5)",
                        bordercolor="rgba(128, 128, 128, 0.3)",
                        borderwidth=1
                    ),
                    height=600,
                    margin=dict(l=50, r=20, t=40, b=30),
                    plot_bgcolor="rgba(0, 0, 0, 0)",
                    paper_bgcolor="rgba(0, 0, 0, 0)"
                )
                
                st.plotly_chart(fig1, use_container_width=True, config=graph_config)
                
            # Estadísticas básicas
            with st.expander("Estadísticas", expanded=True):
                stats_data = []
                for component in data['components']:
                    y_data = data[f'{component}_{data_field_suffix}']
                    stats = {
                        "Componente": component,
                        "Valor Máximo": np.max(np.abs(y_data)),
                        "Valor Mínimo": np.min(y_data),
                        "Media": np.mean(y_data),
                        "Desviación Estándar": np.std(y_data),
                        "RMS": np.sqrt(np.mean(np.square(y_data)))
                    }
                    stats_data.append(stats)
                    
                if len(data['components']) > 1:
                    y_data = data[f'vector_suma_{data_field_suffix}']
                    stats = {
                        "Componente": "Vector Suma",
                        "Valor Máximo": np.max(y_data),
                        "Valor Mínimo": np.min(y_data),
                        "Media": np.mean(y_data),
                        "Desviación Estándar": np.std(y_data),
                        "RMS": np.sqrt(np.mean(np.square(y_data)))
                    }
                    stats_data.append(stats)
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
        
        with tab8:
            st.markdown("""
                <div class='info-container'>
                    <h2>Reportes Automáticos</h2>
                    <p>Genera reportes automáticos con los resultados del análisis.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Selector de registro
            selected_data_index = st.selectbox(
                "Seleccionar registro para el reporte", 
                range(len(all_data)), 
                format_func=lambda i: all_data[i]['name'],
                key="report_data_selector"
            )
            data = all_data[selected_data_index]
            
            # Opciones del reporte
            st.subheader("Opciones del Reporte")
            
            col1, col2 = st.columns(2)
            with col1:
                include_metadata = st.checkbox("Incluir metadatos", value=True)
                include_time_series = st.checkbox("Incluir series de tiempo", value=True)
                include_fft = st.checkbox("Incluir análisis espectral", value=True)
            
            with col2:
                include_response_spectrum = st.checkbox("Incluir espectro de respuesta", value=True)
                include_stats = st.checkbox("Incluir estadísticas", value=True)
                include_events = st.checkbox("Incluir detección de eventos", value=False)
            
            # Formato del reporte
            report_format = st.selectbox(
                "Formato del reporte",
                ["PDF", "HTML", "DOCX"],
                index=0
            )
            
            # Botón para generar reporte
            if st.button("Generar Reporte"):
                with st.spinner("Generando reporte..."):
                    try:
                        # Calcular resultados de análisis necesarios para el reporte
                        analysis_results = {}
                        
                        # Calcular FFT si se solicita
                        if include_fft:
                            analysis_results['fft'] = {}
                            for component in data['components']:
                                # Calcular FFT
                                signal = data[f'{component}_aceleracion']
                                N = len(signal)
                                T = data['time'][1] - data['time'][0]  # Intervalo de tiempo
                                
                                yf = fft(signal)
                                xf = fftfreq(N, T)[:N//2]
                                
                                # Solo usar la mitad positiva del espectro
                                analysis_results['fft'][component] = {
                                    'frequencies': xf,
                                    'amplitudes': 2.0/N * np.abs(yf[:N//2])
                                }
                        
                        # Calcular espectro de respuesta si se solicita
                        if include_response_spectrum:
                            analysis_results['response_spectrum'] = {}
                            
                            # Definir periodos para el espectro de respuesta
                            periods = np.logspace(-1, 1, 100)  # De 0.1 a 10 segundos
                            
                            # Calcular para cada componente
                            for component in data['components']:
                                signal_processor = SignalProcessor(float(data['metadata'].get('sampling_rate', 100)))
                                spectrum = signal_processor.compute_response_spectrum(
                                    data[f'{component}_aceleracion'],
                                    data['time'],
                                    periods=periods
                                )
                                analysis_results['response_spectrum'][component] = spectrum
                        
                        # Generar el reporte
                        report_generator = ReportGenerator()
                        report_file = report_generator.generate_report(
                            data, 
                            analysis_results,
                            output_format=report_format.lower(),
                            options={
                                'include_metadata': include_metadata,
                                'include_time_series': include_time_series,
                                'include_fft': include_fft,
                                'include_response_spectrum': include_response_spectrum,
                                'include_stats': include_stats,
                                'include_events': include_events
                            }
                        )
                        
                        # Preparar descarga
                        with open(report_file, "rb") as file:
                            file_bytes = file.read()
                            b64 = base64.b64encode(file_bytes).decode()
                            
                            extension = report_format.lower()
                            mime_type = {
                                'pdf': 'application/pdf',
                                'html': 'text/html',
                                'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                            }[extension]
                            
                            file_name = f"reporte_{data['name'].split('.')[0]}.{extension}"
                            
                            href = f'<a href="data:{mime_type};base64,{b64}" download="{file_name}">Descargar Reporte {report_format}</a>'
                            st.markdown(href, unsafe_allow_html=True)
                            
                            st.success(f"Reporte generado correctamente en formato {report_format}")
                    
                    except Exception as e:
                        st.error(f"Error al generar el reporte: {str(e)}")

    except Exception as e:
        st.error(f"Error al procesar los archivos: {str(e)}")

if __name__ == "__main__":
    main()
