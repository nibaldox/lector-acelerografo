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

st.set_page_config(
    page_title="Visor de Acelerógrafos",
    layout="wide"
)

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
            all_data.append(data)
        
        # Crear pestañas para diferentes vistas
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Vista Individual", "Comparación", "Análisis Espectral", "Filtros", "Detección de Eventos", "Exportar"])
        
        with tab1:
            # Selector de registro para vista individual
            selected_index = st.selectbox(
                "Seleccionar registro para visualizar",
                range(len(all_data)),
                format_func=lambda i: all_data[i]['name']
            )
            
            selected_data = all_data[selected_index]
            metadata = selected_data['metadata']
        
            # Mostrar información relevante
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Frecuencia de muestreo", f"{metadata.get('sampling_rate', 'N/A')} Hz")
            with col2:
                st.metric("Sensor", metadata.get('sensor_name', 'N/A'))
            with col3:
                st.metric("Unidades", metadata.get('unit', 'm/s/s'))
            
            # Selector de unidades de visualización
            st.sidebar.subheader("Unidades de Visualización")
            display_unit = st.sidebar.radio(
                "Seleccionar unidad:",
                ["m/s²", "g (9.81 m/s²)"],
                key="display_unit_tab1"
            )
            
            # Factor de conversión según la unidad seleccionada
            conversion_factor = 1.0 if display_unit == "m/s²" else 1.0/9.81
            unit_label = "m/s²" if display_unit == "m/s²" else "g"
            
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
                "Z (Vertical)": "#d62728"     # Rojo
            }
            
            # Crear tres gráficos separados para cada componente
            st.subheader("Componentes de Aceleración")
            
            # Componente Norte-Sur
            fig_ns = go.Figure()
            fig_ns.add_trace(go.Scatter(
                x=selected_data['time'],
                y=selected_data['N'] * conversion_factor,
                mode='lines',
                name="N (Norte-Sur)",
                line=dict(color=colors["N (Norte-Sur)"])
            ))
            
            # Configuración del gráfico N-S
            max_val_ns = abs(selected_data['N']).max() * conversion_factor * 1.2
            fig_ns.update_layout(
                title="Componente Norte-Sur",
                xaxis_title="Tiempo (s)",
                yaxis_title=f"Aceleración ({unit_label})",
                height=350,
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis=dict(range=[-max_val_ns, max_val_ns])
            )
            st.plotly_chart(fig_ns, use_container_width=True)
            
            # Componente Este-Oeste
            fig_eo = go.Figure()
            fig_eo.add_trace(go.Scatter(
                x=selected_data['time'],
                y=selected_data['E'] * conversion_factor,
                mode='lines',
                name="E (Este-Oeste)",
                line=dict(color=colors["E (Este-Oeste)"])
            ))
            
            # Configuración del gráfico E-O
            max_val_eo = abs(selected_data['E']).max() * conversion_factor * 1.2
            fig_eo.update_layout(
                title="Componente Este-Oeste",
                xaxis_title="Tiempo (s)",
                yaxis_title=f"Aceleración ({unit_label})",
                height=350,
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis=dict(range=[-max_val_eo, max_val_eo])
            )
            st.plotly_chart(fig_eo, use_container_width=True)
            
            # Componente Vertical
            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(
                x=selected_data['time'],
                y=selected_data['Z'] * conversion_factor,
                mode='lines',
                name="Z (Vertical)",
                line=dict(color=colors["Z (Vertical)"])
            ))
            
            # Configuración del gráfico Z
            max_val_z = abs(selected_data['Z']).max() * conversion_factor * 1.2
            fig_z.update_layout(
                title="Componente Vertical",
                xaxis_title="Tiempo (s)",
                yaxis_title=f"Aceleración ({unit_label})",
                height=350,
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis=dict(range=[-max_val_z, max_val_z])
            )
            st.plotly_chart(fig_z, use_container_width=True)
            
            # Opciones adicionales para análisis
            st.subheader("Opciones adicionales para análisis")
            
            # Selector de componentes
            components = st.multiselect(
                "Seleccionar componentes",
                ["E (Este-Oeste)", "N (Norte-Sur)", "Z (Vertical)"],
                default=["E (Este-Oeste)"]
            )
            
            # Crear gráfico con Plotly para vista individual
            fig1 = go.Figure()
            
            # Agregar cada componente seleccionado
            for component in components:
                key = component_map[component]
                fig1.add_trace(go.Scatter(
                    x=selected_data['time'],
                    y=selected_data[key] * conversion_factor,
                    mode='lines',
                    name=component,
                    line=dict(color=colors[component])
                ))
        
            # Calcular el rango del eje Y basado en el máximo valor absoluto
            max_vals = []
            for component in components:
                key = component_map[component]
                max_vals.append(abs(selected_data[key]).max() * conversion_factor)
            y_max = max(max_vals) * 2  # Duplicar el valor máximo para el rango

            # Configuración del gráfico individual
            fig1.update_layout(
                title=f"Registro de Aceleración - {selected_data['name']}",
                xaxis_title="Tiempo (s)",
                yaxis_title=f"Aceleración ({unit_label})",
                showlegend=True,
                height=600,
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type="linear"
                ),
                yaxis=dict(
                    title=f"Aceleración ({unit_label})",
                    exponentformat='e',
                    showexponent='all',
                    tickformat='.2e',
                    range=[-y_max, y_max]  # Rango simétrico
                )
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
        with tab2:
            # Selector de unidades de visualización para comparación
            st.sidebar.subheader("Unidades de Visualización (Comparación)")
            display_unit_comp = st.sidebar.radio(
                "Seleccionar unidad:",
                ["m/s²", "g (9.81 m/s²)"],
                key="display_unit_tab2"
            )
            
            # Factor de conversión según la unidad seleccionada
            conversion_factor_comp = 1.0 if display_unit_comp == "m/s²" else 1.0/9.81
            unit_label_comp = "m/s²" if display_unit_comp == "m/s²" else "g"
            
            # Crear tres gráficos separados para cada componente (comparación)
            st.subheader("Comparación de Componentes")
            
            # Componente Norte-Sur (Comparación)
            fig_ns_comp = go.Figure()
            
            # Agregar cada registro seleccionado
            for data in all_data:
                fig_ns_comp.add_trace(go.Scatter(
                    x=data['time'],
                    y=data['N'] * conversion_factor_comp,
                    mode='lines',
                    name=data['name']
                ))
            
            # Calcular el rango del eje Y para la comparación
            max_val_ns_comp = max([abs(data['N']).max() * conversion_factor_comp for data in all_data]) * 1.2
            
            # Configuración del gráfico N-S (Comparación)
            fig_ns_comp.update_layout(
                title="Comparación de Componentes Norte-Sur",
                xaxis_title="Tiempo (s)",
                yaxis_title=f"Aceleración ({unit_label_comp})",
                height=350,
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis=dict(range=[-max_val_ns_comp, max_val_ns_comp]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_ns_comp, use_container_width=True)
            
            # Componente Este-Oeste (Comparación)
            fig_eo_comp = go.Figure()
            
            # Agregar cada registro seleccionado
            for data in all_data:
                fig_eo_comp.add_trace(go.Scatter(
                    x=data['time'],
                    y=data['E'] * conversion_factor_comp,
                    mode='lines',
                    name=data['name']
                ))
            
            # Calcular el rango del eje Y para la comparación
            max_val_eo_comp = max([abs(data['E']).max() * conversion_factor_comp for data in all_data]) * 1.2
            
            # Configuración del gráfico E-O (Comparación)
            fig_eo_comp.update_layout(
                title="Comparación de Componentes Este-Oeste",
                xaxis_title="Tiempo (s)",
                yaxis_title=f"Aceleración ({unit_label_comp})",
                height=350,
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis=dict(range=[-max_val_eo_comp, max_val_eo_comp]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_eo_comp, use_container_width=True)
            
            # Componente Vertical (Comparación)
            fig_z_comp = go.Figure()
            
            # Agregar cada registro seleccionado
            for data in all_data:
                fig_z_comp.add_trace(go.Scatter(
                    x=data['time'],
                    y=data['Z'] * conversion_factor_comp,
                    mode='lines',
                    name=data['name']
                ))
            
            # Calcular el rango del eje Y para la comparación
            max_val_z_comp = max([abs(data['Z']).max() * conversion_factor_comp for data in all_data]) * 1.2
            
            # Configuración del gráfico Z (Comparación)
            fig_z_comp.update_layout(
                title="Comparación de Componentes Verticales",
                xaxis_title="Tiempo (s)",
                yaxis_title=f"Aceleración ({unit_label_comp})",
                height=350,
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis=dict(range=[-max_val_z_comp, max_val_z_comp]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_z_comp, use_container_width=True)

        with tab3:
            # Selector de registro para análisis espectral
            if not selected_data:
                st.info("Por favor, seleccione un registro en la vista individual primero")
                return

            # Selector de componente para análisis
            fft_component = st.selectbox(
                "Seleccionar componente para análisis",
                ["E (Este-Oeste)", "N (Norte-Sur)", "Z (Vertical)"],
                key="fft_component"
            )

            # Obtener datos de la componente seleccionada
            key = component_map[fft_component]
            data = selected_data[key]

            # Configuración del análisis FFT
            col1, col2, col3 = st.columns(3)
            with col1:
                window_type = st.selectbox(
                    "Tipo de ventana",
                    ["hann", "hamming", "blackman", "boxcar"],
                    key="window_type"
                )
            with col2:
                # Calcular valor máximo basado en la longitud de los datos
                max_nperseg = min(4096, len(data))
                default_nperseg = min(1024, max_nperseg)
                
                nperseg = st.slider(
                    "Puntos por segmento",
                    min_value=128,
                    max_value=max_nperseg,
                    value=default_nperseg,
                    step=128,
                    key="nperseg",
                    help="Número de puntos para cada segmento de análisis"
                )
            with col3:
                max_overlap = 90
                if nperseg >= len(data):
                    max_overlap = 0
                    
                noverlap = st.slider(
                    "Superposición (%)",
                    min_value=0,
                    max_value=max_overlap,
                    value=min(50, max_overlap),
                    key="noverlap",
                    help="Porcentaje de superposición entre segmentos"
                )
                
                if nperseg >= len(data):
                    st.warning("El tamaño del segmento es igual o mayor que la longitud de los datos. Se analizará la señal completa.")

            # Realizar análisis FFT
            from fft_processor import FFTProcessor
            fft_proc = FFTProcessor(sampling_rate=float(metadata.get('sampling_rate', 100)))

            # Calcular y mostrar FFT
            st.subheader("Espectro de Frecuencias")
            try:
                frequencies, magnitudes, phase = fft_proc.compute_fft(
                    data, 
                    window=window_type,
                    nperseg=nperseg
                )
                
                # Mostrar estadísticas básicas
                col1, col2 = st.columns(2)
                with col1:
                    peak_freq = frequencies[np.argmax(magnitudes)]
                    st.metric("Frecuencia dominante", f"{peak_freq:.2f} Hz")
                with col2:
                    peak_mag = np.max(magnitudes)
                    st.metric("Magnitud máxima", f"{peak_mag:.2e}")
                    
                fig_fft = fft_proc.plot_spectrum(
                    frequencies, 
                    magnitudes,
                    title=f"Espectro de Frecuencias - {fft_component}"
                )
                st.plotly_chart(fig_fft, use_container_width=True)
            except Exception as e:
                st.error(f"Error en el análisis FFT: {str(e)}")
                return

            # Calcular y mostrar espectrograma
            st.subheader("Espectrograma")
            freq_spec, times_spec, Sxx = fft_proc.compute_spectrogram(
                data,
                window=window_type,
                nperseg=nperseg,
                noverlap=int(nperseg * noverlap/100)
            )
            fig_spec = fft_proc.plot_spectrogram(
                freq_spec,
                times_spec,
                Sxx,
                title=f"Espectrograma - {fft_component}"
            )
            st.plotly_chart(fig_spec, use_container_width=True)

        with tab4:
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
            
            st.plotly_chart(fig_response, use_container_width=True)

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
            
            st.plotly_chart(fig_compare, use_container_width=True)

        with tab5:
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
                    st.plotly_chart(fig_ratio, use_container_width=True)
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
                st.plotly_chart(fig_events, use_container_width=True)
                
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
            
        with tab6:
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
        
    except Exception as e:
        st.error(f"Error al procesar los archivos: {str(e)}")

if __name__ == "__main__":
    main()
