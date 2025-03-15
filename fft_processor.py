import numpy as np
from scipy import signal
import plotly.graph_objects as go

class FFTProcessor:
    def __init__(self, sampling_rate):
        """
        Inicializa el procesador FFT
        Args:
            sampling_rate: Frecuencia de muestreo en Hz
        """
        self.sampling_rate = sampling_rate
    
    def compute_fft(self, data, window='hann', nperseg=1024):
        """
        Calcula la FFT de los datos usando ventana
        Args:
            data: Array de datos de entrada
            window: Tipo de ventana ('hann', 'hamming', 'blackman', etc.)
            nperseg: Número de puntos por segmento (por defecto 1024)
        Returns:
            frequencies: Array de frecuencias
            magnitudes: Array de magnitudes promediadas
            phase: Array de fases promediadas
        """
        # Dividir datos en segmentos
        num_segments = len(data) // nperseg
        if num_segments == 0:
            nperseg = len(data)
            num_segments = 1
            
        # Inicializar arrays para promediar
        fft_avg = np.zeros(nperseg//2 + 1, dtype=complex)
        
        # Procesar cada segmento
        for i in range(num_segments):
            start = i * nperseg
            end = start + nperseg
            segment = data[start:end]
            
            # Aplicar ventana
            win = getattr(signal.windows, window)(nperseg)
            windowed_data = segment * win
            
            # Calcular FFT del segmento
            fft_result = np.fft.rfft(windowed_data)
            fft_avg += fft_result
            
        # Promediar resultados
        fft_avg /= num_segments
        
        # Calcular frecuencias
        frequencies = np.fft.rfftfreq(nperseg, d=1/self.sampling_rate)
        
        # Calcular magnitud y fase del promedio
        magnitudes = np.abs(fft_avg)
        phase = np.angle(fft_avg)
        
        return frequencies, magnitudes, phase
    
    def plot_spectrum(self, frequencies, magnitudes, title="Espectro de Frecuencias"):
        """
        Crea una figura de Plotly con el espectro de frecuencias
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=frequencies,
            y=magnitudes,
            mode='lines',
            name='Magnitud'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Frecuencia (Hz)",
            yaxis_title="Magnitud",
            showlegend=True,
            xaxis=dict(
                type="log",  # Escala logarítmica para frecuencias
                exponentformat='power'
            ),
            yaxis=dict(
                type="log",  # Escala logarítmica para magnitudes
                exponentformat='power'
            )
        )
        
        return fig
    
    def compute_spectrogram(self, data, window='hann', nperseg=256, noverlap=None):
        """
        Calcula el espectrograma de los datos
        Args:
            data: Array de datos de entrada
            window: Tipo de ventana
            nperseg: Número de puntos por segmento
            noverlap: Número de puntos de superposición
        Returns:
            frequencies: Array de frecuencias
            times: Array de tiempos
            Sxx: Matriz de potencia espectral
        """
        # Asegurar que nperseg no sea mayor que la longitud de los datos
        nperseg = min(nperseg, len(data))
        
        if noverlap is None:
            noverlap = nperseg // 2
        else:
            noverlap = min(noverlap, nperseg - 1)
            
        # Remover tendencia lineal antes del análisis
        detrended_data = signal.detrend(data)
        
        # Aplicar ventana y calcular espectrograma
        frequencies, times, Sxx = signal.spectrogram(
            detrended_data,
            fs=self.sampling_rate,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            scaling='density'
        )
        
        # Suavizar el espectrograma si es necesario
        if Sxx.shape[1] > 100:  # Si hay muchos segmentos temporales
            # Promedio móvil en el tiempo
            kernel_size = min(5, Sxx.shape[1] // 10)
            kernel = np.ones(kernel_size) / kernel_size
            Sxx = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, Sxx)
        
        return frequencies, times, Sxx
    
    def plot_spectrogram(self, frequencies, times, Sxx, title="Espectrograma"):
        """
        Crea una figura de Plotly con el espectrograma
        """
        fig = go.Figure(data=go.Heatmap(
            x=times,
            y=frequencies,
            z=10 * np.log10(Sxx),  # Convertir a dB
            colorscale='Jet',
            colorbar=dict(title='Potencia (dB)')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Tiempo (s)",
            yaxis_title="Frecuencia (Hz)",
            yaxis=dict(
                type="log",  # Escala logarítmica para frecuencias
                exponentformat='power'
            )
        )
        
        return fig
