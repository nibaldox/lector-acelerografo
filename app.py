import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import os
import numpy as np
import tempfile
import shutil
import zipfile
import io
from ms_reader import MSReader
from signal_processor import SignalProcessor
from fft_processor import FFTProcessor

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
    ms_files = []
    
    if upload_option == "Carpeta Completa (ZIP)":
        zip_file = st.sidebar.file_uploader(
            "Subir carpeta comprimida (ZIP)", 
            type=["zip"],
            help="Comprima la carpeta conteniendo los archivos .ms/.ss antes de subir"
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
            ms_paths = list(extract_dir.rglob("*.ms"))
            for ms_path in ms_paths:
                ss_path = ms_path.with_suffix(".ss")
                if ss_path.exists():
                    ms_files.append((str(ms_path), str(ss_path)))
            
            if ms_files:
                st.sidebar.success(f"Se encontraron {len(ms_files)} pares de archivos .ms/.ss")
            else:
                st.sidebar.warning("No se encontraron pares de archivos .ms/.ss en el ZIP")
    else:
        # Mantener la funcionalidad actual de subida individual
        uploaded_files = st.sidebar.file_uploader(
            "Subir archivos .ms y .ss",
            accept_multiple_files=True,
            type=["ms", "ss"]
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
                    ms_files.append((str(ms_path), str(ss_path)))
    
    if not ms_files:
        st.info("Por favor, sube pares de archivos .ms y .ss para visualizar. Asegúrate de que cada archivo .ms tenga su correspondiente archivo .ss con el mismo nombre.")
        return
        
    # Selector de archivos encontrados
    selected_files = st.multiselect(
        "Seleccionar registros",
        options=ms_files,
        format_func=lambda x: os.path.basename(x[0]),
        default=[ms_files[0]] if ms_files else None
    )
    
    if not selected_files:
        st.info("Por favor, selecciona al menos un registro para visualizar")
        return

    try:
        # Procesar cada par de archivos seleccionados
        all_data = []
        for ms_path, ss_path in selected_files:
            reader = MSReader(ms_path)
            data = reader.read_data()
            data['name'] = os.path.basename(ms_path)
            
            # Procesar datos para obtener velocidad y desplazamiento
            sampling_rate = float(data['metadata'].get('sampling_rate', 100))
            signal_processor = SignalProcessor(sampling_rate)
            
            # Procesar cada componente
            for component in ['N', 'E', 'Z']:
                processed_data = signal_processor.process_acceleration_data(
                    data[component], 
                    data['time']
                )
                # Guardar los datos originales como aceleración
                data[f'{component}_aceleracion'] = data[component]
                data[f'{component}_velocidad'] = processed_data['velocity']
                data[f'{component}_desplazamiento'] = processed_data['displacement']
            
            # Calcular el vector suma (magnitud resultante) para cada tipo de dato
            for data_type in ['aceleracion', 'velocidad', 'desplazamiento']:
                data[f'vector_suma_{data_type}'] = np.sqrt(
                    np.power(data[f'N_{data_type}'], 2) + 
                    np.power(data[f'E_{data_type}'], 2) + 
                    np.power(data[f'Z_{data_type}'], 2)
                )
            
            all_data.append(data)
        
        # Crear pestañas para diferentes vistas con íconos
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Vista Individual",
            "📈 Análisis Espectral",
            "🔍 Filtros",
            "⚡ Detección de Eventos",
            "💾 Exportar",
            "📉 Espectro de Respuesta"
        ])
        
        with tab1:
            st.markdown("""
                <div class='info-container'>
                    <h4 style='margin: 0;'>Vista Individual de Registros</h4>
                    <p style='margin: 0.5rem 0 0 0;'>Visualiza y analiza componentes individuales del registro seleccionado.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Selector de registro para vista individual
            selected_index = st.selectbox(
                "Seleccionar registro para visualizar",
                range(len(all_data)),
                format_func=lambda i: all_data[i]['name']
            )
            
            selected_data = all_data[selected_index]
            metadata = selected_data['metadata']
            
            # Mostrar información relevante con mejor diseño
            st.markdown("<h4 style='margin: 1rem 0;'>Información del Registro</h4>", unsafe_allow_html=True)
            cols = st.columns(4)
            with cols[0]:
                st.metric(
                    "Frecuencia de muestreo",
                    f"{metadata.get('sampling_rate', 'N/A')} Hz",
                    help="Frecuencia de muestreo del registro"
                )
            with cols[1]:
                st.metric(
                    "Sensor",
                    metadata.get('sensor_name', 'N/A'),
                    help="Nombre del sensor utilizado"
                )
            with cols[2]:
                st.metric(
                    "Unidades",
                    metadata.get('unit', 'm/s/s'),
                    help="Unidades de medición"
                )
            with cols[3]:
                st.metric(
                    "Duración",
                    f"{selected_data['time'][-1]:.2f} s",
                    help="Duración total del registro"
                )
                
            # Agregar controles de zoom sincronizados
            st.markdown("<h4 style='margin: 1rem 0;'>Controles de Visualización</h4>", unsafe_allow_html=True)
            zoom_cols = st.columns(2)
            with zoom_cols[0]:
                zoom_start = st.number_input(
                    "Tiempo inicial (s)",
                    0.0,
                    float(selected_data['time'][-1]),
                    0.0,
                    help="Selecciona el tiempo inicial para el zoom"
                )
            with zoom_cols[1]:
                zoom_end = st.number_input(
                    "Tiempo final (s)",
                    zoom_start,
                    float(selected_data['time'][-1]),
                    float(selected_data['time'][-1]),
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
            
            # Mapeo de nombres legibles a claves de datos
            component_map = {
                "E (Este-Oeste)": "E",
                "N (Norte-Sur)": "N",
                "Z (Vertical)": "Z"
            }
            
            # Colores para cada componente
            colors = {
                "E (Este-Oeste)": "#1f77b4",  # Azul
                "N (Norte-Sur)": "#2ca02c",   # Verde
                "Z (Vertical)": "#d62728",     # Rojo
                "Vector Suma": "#9467bd"       # Morado
            }
            
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
                    bgcolor="var(--legend-bg)",
                    bordercolor="rgba(128, 128, 128, 0.3)",
                    borderwidth=1,
                    font=dict(color="var(--text-color)")
                ),
                "xaxis": dict(
                    range=[zoom_start, zoom_end],
                    rangeslider=dict(visible=True, thickness=0.1),
                    title="Tiempo (s)",
                    gridcolor="var(--grid-color)",
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor="var(--zero-line-color)",
                    zerolinewidth=1,
                    color="var(--text-color)"
                ),
                "plot_bgcolor": "var(--plot-bg)",
                "paper_bgcolor": "var(--plot-bg)",
                "font": dict(
                    family="Arial, sans-serif",
                    size=12,
                    color="var(--text-color)"
                ),
                "hoverlabel": dict(
                    bgcolor="var(--hover-bg)",
                    font_size=12,
                    font_family="Arial, sans-serif",
                    font_color="var(--text-color)"
                ),
                "modebar": dict(
                    bgcolor="var(--modebar-bg)",
                    color="var(--modebar-color)",
                    activecolor="var(--modebar-active)"
                )
            }

            # Colores personalizados para cada componente
            colors = {
                "N (Norte-Sur)": "#1f77b4",    # Azul
                "E (Este-Oeste)": "#2ca02c",   # Verde
                "Z (Vertical)": "#d62728",     # Rojo
                "Vector Suma": "#9467bd"       # Morado
            }

            # Crear gráficos para cada componente con la nueva configuración
            st.markdown("""
                <div class='info-container' style='background-color: var(--container-bg) !important; color: var(--text-color) !important;'>
                    <h3 style='margin: 0; color: var(--text-color); background-color: var(--container-bg);'>Visualización de Componentes</h3>
                    <p style='margin: 0.5rem 0 0 0; color: var(--secondary-text-color); background-color: var(--container-bg);'>Gráficos detallados de cada componente del registro sísmico.</p>
                </div>
            """, unsafe_allow_html=True)

            # Componente Norte-Sur con mejoras visuales
            fig_ns = go.Figure()
            fig_ns.add_trace(go.Scatter(
                x=selected_data['time'],
                y=selected_data[f'N_{data_field_suffix}'] * conversion_factor,
                mode='lines',
                name="N (Norte-Sur)",
                line=dict(
                    color=colors["N (Norte-Sur)"],
                    width=2,
                    shape='linear'
                ),
                hovertemplate="<b>Tiempo:</b> %{x:.2f}s<br><b>Valor:</b> %{y:.3f} " + unit_label
            ))
            
            # Configuración específica para N-S
            max_val_ns = abs(selected_data[f'N_{data_field_suffix}']).max() * conversion_factor * 1.2
            layout_ns = layout_config.copy()
            layout_ns.update({
                "title": dict(
                    text="<b>" + title_prefix + " Norte-Sur</b>",
                    x=0.5,
                    xanchor='center',
                    font=dict(size=16)
                ),
                "yaxis": dict(
                    title=dict(
                        text=f"{title_prefix} ({unit_label})",
                        standoff=10
                    ),
                    range=[-max_val_ns, max_val_ns],
                    gridcolor="var(--grid-color)",
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor="var(--zero-line-color)",
                    zerolinewidth=1,
                    color="var(--text-color)"
                )
            })
            fig_ns.update_layout(**layout_ns)
            
            # Agregar anotaciones para valores máximos y mínimos
            max_idx = np.argmax(abs(selected_data[f'N_{data_field_suffix}']))
            max_time = selected_data['time'][max_idx]
            max_value = selected_data[f'N_{data_field_suffix}'][max_idx] * conversion_factor
            
            fig_ns.add_annotation(
                x=max_time,
                y=max_value,
                text=f"Max: {max_value:.2f} {unit_label}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=colors["N (Norte-Sur)"],
                bgcolor="var(--container-bg)",
                bordercolor=colors["N (Norte-Sur)"],
                borderwidth=1,
                borderpad=4,
                font=dict(size=10, color="var(--text-color)")
            )
            
            st.plotly_chart(fig_ns, use_container_width=True, config=graph_config)
            
            # Componente Este-Oeste (similar configuración)
            fig_eo = go.Figure()
            fig_eo.add_trace(go.Scatter(
                x=selected_data['time'],
                y=selected_data[f'E_{data_field_suffix}'] * conversion_factor,
                mode='lines',
                name="E (Este-Oeste)",
                line=dict(
                    color=colors["E (Este-Oeste)"],
                    width=2,
                    shape='linear'
                ),
                hovertemplate="<b>Tiempo:</b> %{x:.2f}s<br><b>Valor:</b> %{y:.3f} " + unit_label
            ))
            
            max_val_eo = abs(selected_data[f'E_{data_field_suffix}']).max() * conversion_factor * 1.2
            fig_eo.update_layout(
                title=dict(
                    text="<b>" + title_prefix + " Este-Oeste</b>",
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
                    range=[-max_val_eo, max_val_eo],
                    gridcolor="var(--grid-color)",
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor="var(--zero-line-color)",
                    zerolinewidth=1,
                    color="var(--text-color)"
                )
            )
            st.plotly_chart(fig_eo, use_container_width=True, config=graph_config)
            
            # Componente Vertical (similar configuración)
            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(
                x=selected_data['time'],
                y=selected_data[f'Z_{data_field_suffix}'] * conversion_factor,
                mode='lines',
                name="Z (Vertical)",
                line=dict(
                    color=colors["Z (Vertical)"],
                    width=2,
                    shape='linear'
                ),
                hovertemplate="<b>Tiempo:</b> %{x:.2f}s<br><b>Valor:</b> %{y:.3f} " + unit_label
            ))
            
            max_val_z = abs(selected_data[f'Z_{data_field_suffix}']).max() * conversion_factor * 1.2
            fig_z.update_layout(
                title=dict(
                    text="<b>" + title_prefix + " Vertical</b>",
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
                    range=[-max_val_z, max_val_z],
                    gridcolor="var(--grid-color)",
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor="var(--zero-line-color)",
                    zerolinewidth=1,
                    color="var(--text-color)"
                )
            )
            st.plotly_chart(fig_z, use_container_width=True, config=graph_config)
            
            # Vector Suma (similar configuración)
            fig_suma = go.Figure()
            fig_suma.add_trace(go.Scatter(
                x=selected_data['time'],
                y=selected_data[f'vector_suma_{data_field_suffix}'] * conversion_factor,
                mode='lines',
                name="Vector Suma",
                line=dict(
                    color=colors["Vector Suma"],
                    width=2,
                    shape='linear'
                ),
                hovertemplate="<b>Tiempo:</b> %{x:.2f}s<br><b>Valor:</b> %{y:.3f} " + unit_label
            ))
            
            max_val_suma = abs(selected_data[f'vector_suma_{data_field_suffix}']).max() * conversion_factor * 1.2
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
                    gridcolor="var(--grid-color)",
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor="var(--zero-line-color)",
                    zerolinewidth=1,
                    color="var(--text-color)"
                )
            )
            st.plotly_chart(fig_suma, use_container_width=True, config=graph_config)
            
            # Opciones adicionales para análisis
            st.subheader("Opciones adicionales para análisis")
            
            # Selector de componentes para el gráfico individual
            components = st.multiselect(
                "Seleccionar componentes para visualizar juntos:",
                ["N (Norte-Sur)", "E (Este-Oeste)", "Z (Vertical)", "Vector Suma"],
                default=["N (Norte-Sur)", "E (Este-Oeste)", "Z (Vertical)"],
                key="components_tab1"
            )
            
            # Mapeo de nombres legibles a claves de datos
            component_map = {
                "N (Norte-Sur)": "N",
                "E (Este-Oeste)": "E",
                "Z (Vertical)": "Z",
                "Vector Suma": "vector_suma"
            }
            
            # Colores para cada componente
            colors = {
                "N (Norte-Sur)": "blue",
                "E (Este-Oeste)": "red",
                "Z (Vertical)": "green",
                "Vector Suma": "purple"
            }
            
            # Crear gráfico individual con todos los componentes seleccionados
            if components:
                fig1 = go.Figure()
                for component in components:
                    key = component_map[component]
                    fig1.add_trace(go.Scatter(
                        x=selected_data['time'],
                        y=selected_data[f'{key}_{data_field_suffix}'] * conversion_factor,
                        mode='lines',
                        name=component,
                        line=dict(color=colors[component])
                    ))
        
                # Calcular el rango del eje Y basado en el máximo valor absoluto
                max_vals = []
                for component in components:
                    key = component_map[component]
                    max_vals.append(abs(selected_data[f'{key}_{data_field_suffix}']).max() * conversion_factor)
                y_max = max(max_vals) * 2  # Duplicar el valor máximo para el rango

                # Configuración del gráfico individual
                fig1.update_layout(
                    title=f"Registro de {title_prefix} - {selected_data['name']}",
                    xaxis_title="Tiempo (s)",
                    yaxis_title=f"{title_prefix} ({unit_label})",
                    showlegend=True,
                    height=600,
                    xaxis=dict(
                        rangeslider=dict(visible=True),
                        type="linear"
                    ),
                    yaxis=dict(
                        title=f"{title_prefix} ({unit_label})",
                        exponentformat='e',
                        showexponent='all',
                        tickformat='.2e',
                        range=[-y_max, y_max]  # Rango simétrico
                    )
                )
                
                st.plotly_chart(fig1, use_container_width=True, config=graph_config)
            
        with tab2:
            # Selector de registro para análisis espectral
            selected_record_spectral = st.selectbox(
                "Seleccionar registro para análisis espectral",
                [data['name'] for data in all_data],
                key="selected_record_spectral"
            )
            
            # Obtener los datos del registro seleccionado
            selected_data_spectral = next(data for data in all_data if data['name'] == selected_record_spectral)
            
            # Selector de componente para análisis espectral
            component_spectral = st.selectbox(
                "Seleccionar componente para análisis espectral",
                ["N (Norte-Sur)", "E (Este-Oeste)", "Z (Vertical)", "Vector Suma"],
                key="component_spectral"
            )
            
            # Obtener el componente seleccionado
            component_key = component_map[component_spectral]
            
            # Selector de tipo de análisis espectral
            analysis_type = st.selectbox(
                "Tipo de análisis espectral",
                ["Espectro de Fourier", "Espectro de Potencia", "Autocorrelación"],
                key="analysis_type"
            )
            
            # Obtener la frecuencia de muestreo del registro
            sampling_rate = 1 / (selected_data_spectral['time'][1] - selected_data_spectral['time'][0])
            
            if analysis_type == "Espectro de Fourier":
                # Inicializar el procesador FFT
                fft_processor = FFTProcessor(sampling_rate)
                
                # Calcular el espectro de Fourier
                frequencies, magnitudes, phase = fft_processor.compute_fft(
                    selected_data_spectral[f'{component_key}_{data_field_suffix}']
                )
                
                # Crear gráfico del espectro de Fourier
                fig_fft = go.Figure()
                fig_fft.add_trace(go.Scatter(
                    x=frequencies,
                    y=magnitudes,
                    mode='lines',
                    name='Amplitud'
                ))
                
                fig_fft.update_layout(
                    title=f"Espectro de Fourier - {component_spectral}",
                    xaxis_title="Frecuencia (Hz)",
                    yaxis_title="Amplitud",
                    xaxis_type="log",
                    yaxis_type="log",
                    showlegend=True
                )
                st.plotly_chart(fig_fft, use_container_width=True, config=graph_config)
                
            elif analysis_type == "Espectro de Potencia":
                # Calcular el espectro de potencia
                power_result = signal_processor.compute_power_spectrum(
                    selected_data_spectral[f'{component_key}_{data_field_suffix}'],
                    sampling_rate
                )
                
                # Crear gráfico del espectro de potencia
                fig_power = go.Figure()
                fig_power.add_trace(go.Scatter(
                    x=power_result['frequencies'],
                    y=power_result['power_spectrum'],
                    mode='lines',
                    name='Potencia'
                ))
                
                fig_power.update_layout(
                    title=f"Espectro de Potencia - {component_spectral}",
                    xaxis_title="Frecuencia (Hz)",
                    yaxis_title="Potencia",
                    xaxis_type="log",
                    yaxis_type="log",
                    showlegend=True
                )
                st.plotly_chart(fig_power, use_container_width=True, config=graph_config)
                
            else:  # Autocorrelación
                # Calcular la autocorrelación
                autocorr_result = signal_processor.compute_autocorrelation(
                    selected_data_spectral[f'{component_key}_{data_field_suffix}']
                )
                
                # Crear gráfico de autocorrelación
                fig_autocorr = go.Figure()
                fig_autocorr.add_trace(go.Scatter(
                    x=autocorr_result['lags'],
                    y=autocorr_result['autocorr'],
                    mode='lines',
                    name='Autocorrelación'
                ))
                
                fig_autocorr.update_layout(
                    title=f"Función de Autocorrelación - {component_spectral}",
                    xaxis_title="Desfase (muestras)",
                    yaxis_title="Coeficiente de Autocorrelación",
                    showlegend=True
                )
                st.plotly_chart(fig_autocorr, use_container_width=True, config=graph_config)
            
        with tab3:
            # Selector de registro para filtrado
            if not selected_data:
                st.info("Por favor, seleccione un registro en la vista individual primero")
                return

            # Selector de componente para filtrar
            filter_component = st.selectbox(
                "Seleccionar componente para filtrar",
                ["E (Este-Oeste)", "N (Norte-Sur)", "Z (Vertical)"],
                key="filter_component"
            )

            # Configuración del filtro
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_type = st.selectbox(
                    "Tipo de filtro",
                    ["lowpass", "highpass", "bandpass"],
                    key="filter_type"
                )
            with col2:
                if filter_type in ["lowpass", "highpass"]:
                    cutoff = st.slider(
                        "Frecuencia de corte (Hz)",
                        min_value=0.1,
                        max_value=float(metadata.get('sampling_rate', '100'))/2,
                        value=10.0,
                        step=0.1,
                        key="cutoff",
                        help="Frecuencia de corte para el filtro"
                    )
                else:
                    lowcut = st.slider(
                        "Frecuencia de corte inferior (Hz)",
                        min_value=0.1,
                        max_value=float(metadata.get('sampling_rate', '100'))/2,
                        value=1.0,
                        step=0.1,
                        key="lowcut",
                        help="Frecuencia de corte inferior para el filtro"
                    )
            with col3:
                if filter_type == "bandpass":
                    highcut = st.slider(
                        "Frecuencia de corte superior (Hz)",
                        min_value=lowcut,
                        max_value=float(metadata.get('sampling_rate', '100'))/2,
                        value=min(20.0, float(metadata.get('sampling_rate', '100'))/2),
                        step=0.1,
                        key="highcut",
                        help="Frecuencia de corte superior para el filtro"
                    )
                
                filter_order = st.slider(
                    "Orden del filtro",
                    min_value=2,
                    max_value=8,
                    value=4,
                    step=2,
                    key="filter_order",
                    help="Orden del filtro"
                )

            # Aplicar filtro
            from filters import SignalFilter
            signal_filter = SignalFilter(sampling_rate=float(metadata.get('sampling_rate', 100)))
            
            # Obtener datos de la componente seleccionada
            key = component_map[filter_component]
            data = selected_data[key]

            # Configurar parámetros del filtro
            filter_params = {'order': filter_order}
            if filter_type in ["lowpass", "highpass"]:
                filter_params['cutoff'] = cutoff
            else:
                filter_params['lowcut'] = lowcut
                filter_params['highcut'] = highcut

            # Aplicar filtro y mostrar resultados
            filtered_data = signal_filter.apply_filter(data, filter_type, **filter_params)

            # Mostrar respuesta en frecuencia del filtro
            st.subheader("Respuesta en Frecuencia del Filtro")
            freqs, response = signal_filter.get_filter_response(filter_type, **filter_params)
            
            fig_response = go.Figure()
            fig_response.add_trace(go.Scatter(
                x=freqs,
                y=response,
                mode='lines',
                name='Respuesta'
            ))
            
            fig_response.update_layout(
                title="Respuesta en Frecuencia",
                xaxis_title="Frecuencia (Hz)",
                yaxis_title="Magnitud",
                xaxis=dict(type="log"),
                showlegend=True
            )
            
            st.plotly_chart(fig_response, use_container_width=True, config=graph_config)

            # Mostrar señal original vs filtrada
            st.subheader("Comparación: Original vs Filtrada")
            fig_compare = go.Figure()
            
            fig_compare.add_trace(go.Scatter(
                x=selected_data['time'],
                y=data,
                mode='lines',
                name='Original',
                line=dict(color='blue')
            ))
            
            fig_compare.add_trace(go.Scatter(
                x=selected_data['time'],
                y=filtered_data,
                mode='lines',
                name='Filtrada',
                line=dict(color='red')
            ))
            
            fig_compare.update_layout(
                title=f"Señal Original vs Filtrada - {filter_component}",
                xaxis_title="Tiempo (s)",
                yaxis_title=f"Aceleración ({metadata.get('unit', 'm/s/s')})",
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig_compare, use_container_width=True, config=graph_config)

        with tab4:
            # Selector de registro para detección de eventos
            if not selected_data:
                st.info("Por favor, seleccione un registro en la vista individual primero")
                return

            # Selector de componente para análisis
            event_component = st.selectbox(
                "Seleccionar componente para detectar eventos",
                ["E (Este-Oeste)", "N (Norte-Sur)", "Z (Vertical)"],
                key="event_component"
            )

            # Configuración de detección
            st.subheader("Configuración de Detección")
            
            detection_method = st.radio(
                "Método de detección",
                ["STA/LTA", "Detección de picos"]
            )
            
            if detection_method == "STA/LTA":
                col1, col2, col3 = st.columns(3)
                with col1:
                    sta_window = st.slider(
                        "Ventana STA (s)",
                        min_value=0.1,
                        max_value=5.0,
                        value=1.0,
                        step=0.1,
                        key="sta_window"
                    )
                with col2:
                    lta_window = st.slider(
                        "Ventana LTA (s)",
                        min_value=1.0,
                        max_value=50.0,
                        value=10.0,
                        step=1.0,
                        key="lta_window"
                    )
                with col3:
                    trigger_ratio = st.slider(
                        "Ratio de disparo",
                        min_value=1.0,
                        max_value=10.0,
                        value=3.0,
                        step=0.1,
                        key="trigger_ratio"
                    )
            else:
                col1, col2 = st.columns(2)
                with col1:
                    peak_threshold = st.slider(
                        "Umbral (desviaciones estándar)",
                        min_value=1.0,
                        max_value=10.0,
                        value=3.0,
                        step=0.5,
                        key="peak_threshold"
                    )
                with col2:
                    min_distance = st.slider(
                        "Distancia mínima entre eventos (s)",
                        min_value=0.1,
                        max_value=10.0,
                        value=0.5,
                        step=0.1,
                        key="min_distance"
                    )

            # Realizar detección de eventos
            from event_detector import EventDetector
            detector = EventDetector(sampling_rate=float(metadata.get('sampling_rate', 100)))
            
            # Obtener datos de la componente seleccionada
            key = component_map[event_component]
            data = selected_data[key]

            try:
                # Detectar eventos
                events = []
                if detection_method == "STA/LTA":
                    events, ratio = detector.sta_lta(
                        data,
                        sta_window=sta_window,
                        lta_window=lta_window,
                        trigger_ratio=trigger_ratio
                    )
                    
                    # Visualizar ratio STA/LTA
                    fig_ratio = go.Figure()
                    fig_ratio.add_trace(go.Scatter(
                        x=selected_data['time'],
                        y=ratio,
                        mode='lines',
                        name='STA/LTA Ratio'
                    ))
                    fig_ratio.add_hline(
                        y=trigger_ratio,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Umbral de disparo"
                    )
                    fig_ratio.update_layout(
                        title="Ratio STA/LTA",
                        xaxis_title="Tiempo (s)",
                        yaxis_title="Ratio",
                        height=300
                    )
                    st.plotly_chart(fig_ratio, use_container_width=True, config=graph_config)
                else:
                    peaks, properties = detector.peak_detection(
                        data,
                        threshold=peak_threshold * np.std(data),
                        distance=min_distance
                    )
                    if len(peaks) > 0:
                        events = peaks / float(metadata.get('sampling_rate', 100))
                    else:
                        events = []

                # Visualizar resultados
                st.subheader("Eventos Detectados")
                
                fig_events = go.Figure()
                
                # Señal original
                fig_events.add_trace(go.Scatter(
                    x=selected_data['time'],
                    y=data,
                    mode='lines',
                    name='Señal',
                    line=dict(color='blue')
                ))
                
                # Configurar layout base
                fig_events.update_layout(
                    title=f"Eventos Detectados - {event_component}",
                    xaxis_title="Tiempo (s)",
                    yaxis_title=f"Aceleración ({metadata.get('unit', 'm/s/s')})",
                    showlegend=True,
                    height=500
                )
            
                # Marcar eventos si se encontraron
                if len(events) > 0:
                    for event_time in events:
                        fig_events.add_vline(
                            x=event_time,
                            line_width=1,
                            line_dash="dash",
                            line_color="red"
                        )
                        
                        # Calcular características del evento
                        features = detector.calculate_event_features(data, event_time)
                        
                        # Agregar anotación
                        fig_events.add_annotation(
                            x=event_time,
                            y=np.max(data),
                            text=f"A={features['peak_amplitude']:.2e}",
                            showarrow=True,
                            arrowhead=1
                        )
                
                # Mostrar gráfico
                st.plotly_chart(fig_events, use_container_width=True, config=graph_config)
                
                # Mostrar lista de eventos
                if len(events) > 0:
                    st.write(f"Se detectaron {len(events)} eventos:")
                    for i, event_time in enumerate(events, 1):
                        features = detector.calculate_event_features(data, event_time)
                        st.write(f"Evento {i}:")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Tiempo", f"{event_time:.2f} s")
                        with col2:
                            st.metric("Amplitud pico", f"{features['peak_amplitude']:.2e}")
                        with col3:
                            st.metric("RMS", f"{features['rms']:.2e}")
                        with col4:
                            st.metric("Energía", f"{features['energy']:.2e}")
                else:
                    st.info("No se detectaron eventos con los parámetros actuales")
                    
            except Exception as e:
                st.error(f"Error en la detección de eventos: {str(e)}")

            # Agregar información adicional
            st.markdown("### Información del Registro")
            
        with tab5:
            # Selector de registro para exportación
            if not selected_data:
                st.info("Por favor, seleccione un registro en la vista individual primero")
                return

            st.subheader("Exportar Datos")
            
            # Configuración de exportación
            col1, col2 = st.columns(2)
            with col1:
                export_type = st.selectbox(
                    "Tipo de datos a exportar",
                    ["Datos crudos", "Resultados de análisis", "Gráficos"]
                )
            with col2:
                if export_type == "Datos crudos":
                    format = st.selectbox(
                        "Formato de exportación",
                        ["csv", "excel", "json"]
                    )

            # Configurar y realizar exportación
            from data_exporter import DataExporter
            exporter = DataExporter()
            
            if st.button("Exportar"):
                try:
                    if export_type == "Datos crudos":
                        output_path = exporter.export_raw_data(
                            selected_data,
                            selected_data['name'],
                            format=format
                        )
                        st.success(f"Datos exportados a: {output_path}")
                        
                    elif export_type == "Resultados de análisis":
                        analysis_type = st.selectbox(
                            "Tipo de análisis",
                            ["fft", "filtered", "events"]
                        )
                        
                        # Preparar resultados según el tipo de análisis
                        if analysis_type == "fft":
                            # Usar los últimos resultados FFT
                            results = {
                                'frequencies': frequencies,
                                'magnitudes': magnitudes,
                                'phase': phase
                            }
                        elif analysis_type == "filtered":
                            # Usar los últimos resultados del filtrado
                            results = {
                                'component': key,
                                'filtered_data': filtered_data,
                                'filter_params': filter_params
                            }
                        elif analysis_type == "events":
                            # Usar los últimos resultados de detección
                            results = {
                                'events': [
                                    {
                                        'time': t,
                                        'features': detector.calculate_event_features(data, t)
                                    }
                                    for t in events
                                ]
                            }
                            
                        output_path = exporter.export_analysis_results(
                            selected_data,
                            analysis_type,
                            results,
                            selected_data['name']
                        )
                        st.success(f"Resultados exportados a: {output_path}")
                        
                    elif export_type == "Gráficos":
                        graph_type = st.selectbox(
                            "Tipo de gráfico",
                            ["Series de tiempo", "FFT", "Espectrograma", "Filtrado"]
                        )
                        
                        if graph_type == "Series de tiempo":
                            paths = exporter.export_plot(fig1, f"{selected_data['name']}_timeseries")
                        elif graph_type == "FFT":
                            paths = exporter.export_plot(fig_fft, f"{selected_data['name']}_fft")
                        elif graph_type == "Espectrograma":
                            paths = exporter.export_plot(fig_spec, f"{selected_data['name']}_spectrogram")
                        elif graph_type == "Filtrado":
                            paths = exporter.export_plot(fig_compare, f"{selected_data['name']}_filtered")
                            
                        st.success(f"Gráficos exportados como:")
                        for path in paths:
                            st.write(f"- {path}")
                            
                except Exception as e:
                    st.error(f"Error durante la exportación: {str(e)}")

            # Agregar información adicional
            st.markdown("### Información del Registro")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duración", f"{selected_data['time'][-1]:.2f} s")
            with col2:
                st.metric("Muestras", len(selected_data['time']))
            with col3:
                st.metric("Ganancia", metadata.get('gain', 'N/A'))
            with col4:
                st.metric("Sensibilidad", metadata.get('sens', 'N/A'))
            
            # Agregar controles de zoom/pan a fig1
            fig1.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(
                                label="Reset Zoom",
                                method="relayout",
                                args=[{"xaxis.autorange": True, "yaxis.autorange": True}]
                            )
                        ]
                    )
                ]
            )
        
        with tab6:
            st.subheader("Espectro de Respuesta")
            
            # Selector de registro para espectro de respuesta
            selected_record_response = st.selectbox(
                "Seleccionar registro para espectro de respuesta",
                [data['name'] for data in all_data],
                key="selected_record_response"
            )
            
            # Obtener los datos del registro seleccionado
            selected_data_response = next(data for data in all_data if data['name'] == selected_record_response)
            
            # Selector de tipo de análisis
            analysis_type = st.radio(
                "Tipo de análisis",
                ["Individual", "Combinado"],
                key="response_analysis_type"
            )
            
            if analysis_type == "Individual":
                # Selector de componente
                component_response = st.selectbox(
                    "Seleccionar componente",
                    ["N (Norte-Sur)", "E (Este-Oeste)", "Z (Vertical)", "Vector Suma"],
                    key="component_response"
                )
                
                # Parámetros del espectro de respuesta
                col1, col2 = st.columns(2)
                with col1:
                    damping = st.slider(
                        "Amortiguamiento (%)",
                        min_value=0.0,
                        max_value=20.0,
                        value=5.0,
                        step=0.5,
                        key="damping"
                    )
                
                with col2:
                    n_periods = st.slider(
                        "Número de períodos",
                        min_value=20,
                        max_value=200,
                        value=100,
                        step=10,
                        key="n_periods"
                    )
                
                # Calcular períodos
                periods = np.logspace(-2, 1, n_periods)  # 0.01s a 10s
                
                # Obtener el componente seleccionado
                component_key = component_map[component_response]
                
                # Calcular espectro de respuesta
                response_spectrum = signal_processor.compute_response_spectrum(
                    selected_data_response[f'{component_key}_{data_field_suffix}'],
                    selected_data_response['time'],
                    periods=periods,
                    damping_ratio=damping/100
                )
                
                # Crear gráficos
                col1, col2 = st.columns(2)
                
                with col1:
                    # Espectro de pseudo-aceleración
                    fig_sa = go.Figure()
                    fig_sa.add_trace(go.Scatter(
                        x=response_spectrum['periods'],
                        y=response_spectrum['Sa'],
                        mode='lines',
                        name='Sa'
                    ))
                    
                    fig_sa.update_layout(
                        title=f"Espectro de Pseudo-aceleración - {component_response}",
                        xaxis_title="Período (s)",
                        yaxis_title="Sa (g)",
                        xaxis_type="log",
                        yaxis_type="log",
                        showlegend=True
                    )
                    st.plotly_chart(fig_sa, use_container_width=True, config=graph_config)
                
                with col2:
                    # Espectro de pseudo-velocidad
                    fig_sv = go.Figure()
                    fig_sv.add_trace(go.Scatter(
                        x=response_spectrum['periods'],
                        y=response_spectrum['Sv'],
                        mode='lines',
                        name='Sv'
                    ))
                    
                    fig_sv.update_layout(
                        title=f"Espectro de Pseudo-velocidad - {component_response}",
                        xaxis_title="Período (s)",
                        yaxis_title="Sv (m/s)",
                        xaxis_type="log",
                        yaxis_type="log",
                        showlegend=True
                    )
                    st.plotly_chart(fig_sv, use_container_width=True, config=graph_config)
                
                # Espectro de desplazamiento
                fig_sd = go.Figure()
                fig_sd.add_trace(go.Scatter(
                    x=response_spectrum['periods'],
                    y=response_spectrum['Sd'],
                    mode='lines',
                    name='Sd'
                ))
                
                fig_sd.update_layout(
                    title=f"Espectro de Desplazamiento - {component_response}",
                    xaxis_title="Período (s)",
                    yaxis_title="Sd (m)",
                    xaxis_type="log",
                    yaxis_type="log",
                    showlegend=True
                )
                st.plotly_chart(fig_sd, use_container_width=True, config=graph_config)
                
            else:  # Análisis Combinado
                # Método de combinación
                combination_method = st.radio(
                    "Método de combinación",
                    ["SRSS", "Porcentual"],
                    key="combination_method",
                    help="SRSS: Raíz cuadrada de la suma de cuadrados\nPorcentual: Regla del 30%"
                )
                
                # Parámetros del espectro de respuesta
                col1, col2 = st.columns(2)
                with col1:
                    damping = st.slider(
                        "Amortiguamiento (%)",
                        min_value=0.0,
                        max_value=20.0,
                        value=5.0,
                        step=0.5,
                        key="damping_combined"
                    )
                
                # Calcular respuesta combinada
                combined_response = signal_processor.compute_combined_response(
                    selected_data_response[f'N_{data_field_suffix}'],
                    selected_data_response[f'E_{data_field_suffix}'],
                    selected_data_response[f'Z_{data_field_suffix}'],
                    selected_data_response['time'],
                    method=combination_method
                )
                
                # Crear gráficos
                col1, col2 = st.columns(2)
                
                with col1:
                    # Espectro de pseudo-aceleración combinado
                    fig_sa_comb = go.Figure()
                    # Agregar componentes individuales
                    fig_sa_comb.add_trace(go.Scatter(
                        x=combined_response['periods'],
                        y=combined_response['Sa_x'],
                        mode='lines',
                        name='N-S',
                        line=dict(dash='dot')
                    ))
                    fig_sa_comb.add_trace(go.Scatter(
                        x=combined_response['periods'],
                        y=combined_response['Sa_y'],
                        mode='lines',
                        name='E-O',
                        line=dict(dash='dot')
                    ))
                    fig_sa_comb.add_trace(go.Scatter(
                        x=combined_response['periods'],
                        y=combined_response['Sa_z'],
                        mode='lines',
                        name='Z',
                        line=dict(dash='dot')
                    ))
                    # Agregar respuesta combinada
                    fig_sa_comb.add_trace(go.Scatter(
                        x=combined_response['periods'],
                        y=combined_response['Sa_combined'],
                        mode='lines',
                        name='Combinada',
                        line=dict(width=3)
                    ))
                    
                    fig_sa_comb.update_layout(
                        title=f"Espectro de Pseudo-aceleración Combinado ({combination_method})",
                        xaxis_title="Período (s)",
                        yaxis_title="Sa (g)",
                        xaxis_type="log",
                        yaxis_type="log",
                        showlegend=True
                    )
                    st.plotly_chart(fig_sa_comb, use_container_width=True, config=graph_config)
                
                with col2:
                    # Espectro de pseudo-velocidad combinado
                    fig_sv_comb = go.Figure()
                    # Agregar componentes individuales
                    fig_sv_comb.add_trace(go.Scatter(
                        x=combined_response['periods'],
                        y=combined_response['Sv_x'],
                        mode='lines',
                        name='N-S',
                        line=dict(dash='dot')
                    ))
                    fig_sv_comb.add_trace(go.Scatter(
                        x=combined_response['periods'],
                        y=combined_response['Sv_y'],
                        mode='lines',
                        name='E-O',
                        line=dict(dash='dot')
                    ))
                    fig_sv_comb.add_trace(go.Scatter(
                        x=combined_response['periods'],
                        y=combined_response['Sv_z'],
                        mode='lines',
                        name='Z',
                        line=dict(dash='dot')
                    ))
                    # Agregar respuesta combinada
                    fig_sv_comb.add_trace(go.Scatter(
                        x=combined_response['periods'],
                        y=combined_response['Sv_combined'],
                        mode='lines',
                        name='Combinada',
                        line=dict(width=3)
                    ))
                    
                    fig_sv_comb.update_layout(
                        title=f"Espectro de Pseudo-velocidad Combinado ({combination_method})",
                        xaxis_title="Período (s)",
                        yaxis_title="Sv (m/s)",
                        xaxis_type="log",
                        yaxis_type="log",
                        showlegend=True
                    )
                    st.plotly_chart(fig_sv_comb, use_container_width=True, config=graph_config)
                
                # Espectro de desplazamiento combinado
                fig_sd_comb = go.Figure()
                # Agregar componentes individuales
                fig_sd_comb.add_trace(go.Scatter(
                    x=combined_response['periods'],
                    y=combined_response['Sd_x'],
                    mode='lines',
                    name='N-S',
                    line=dict(dash='dot')
                ))
                fig_sd_comb.add_trace(go.Scatter(
                    x=combined_response['periods'],
                    y=combined_response['Sd_y'],
                    mode='lines',
                    name='E-O',
                    line=dict(dash='dot')
                ))
                fig_sd_comb.add_trace(go.Scatter(
                    x=combined_response['periods'],
                    y=combined_response['Sd_z'],
                    mode='lines',
                    name='Z',
                    line=dict(dash='dot')
                ))
                # Agregar respuesta combinada
                fig_sd_comb.add_trace(go.Scatter(
                    x=combined_response['periods'],
                    y=combined_response['Sd_combined'],
                    mode='lines',
                    name='Combinada',
                    line=dict(width=3)
                ))
                
                fig_sd_comb.update_layout(
                    title=f"Espectro de Desplazamiento Combinado ({combination_method})",
                    xaxis_title="Período (s)",
                    yaxis_title="Sd (m)",
                    xaxis_type="log",
                    yaxis_type="log",
                    showlegend=True
                )
                st.plotly_chart(fig_sd_comb, use_container_width=True, config=graph_config)
            
    except Exception as e:
        st.error(f"Error al procesar los archivos: {str(e)}")

if __name__ == "__main__":
    main()
