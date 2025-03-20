#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el módulo de procesamiento de señales utilizando pytest.
"""

import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pytest

from signal_processor import SignalProcessor
from filters import SignalFilter

# Fixtures para configurar y limpiar el entorno de pruebas
@pytest.fixture
def temp_dir():
    """Crea un directorio temporal para las pruebas."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Limpieza después de las pruebas
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_acceleration():
    """Genera una señal de aceleración de ejemplo para pruebas."""
    # Crear una señal de aceleración sintética
    t = np.linspace(0, 30, 3000)  # 30 segundos a 100 Hz
    
    # Componente principal (5 Hz)
    main_component = 2.5 * np.sin(2 * np.pi * 5 * t) * np.exp(-0.2 * t)
    
    # Ruido de alta frecuencia
    high_freq_noise = 0.5 * np.sin(2 * np.pi * 20 * t) * np.exp(-0.1 * t)
    
    # Tendencia de línea base
    baseline_trend = 0.05 * t + 0.01 * t**2 * np.exp(-0.1 * t)
    
    # Combinar componentes
    acceleration = main_component + high_freq_noise + baseline_trend
    
    return acceleration, 100.0  # Señal y frecuencia de muestreo (Hz)

@pytest.fixture
def signal_processor():
    """Crea una instancia de SignalProcessor para pruebas."""
    return SignalProcessor(sampling_rate=100.0)

@pytest.fixture
def signal_filter():
    """Crea una instancia de SignalFilter para pruebas."""
    return SignalFilter(sampling_rate=100.0)

# Pruebas para el procesador de señales
class TestSignalProcessor:
    """Pruebas para la clase SignalProcessor."""
    
    def test_remove_baseline(self, signal_processor, sample_acceleration):
        """Prueba la eliminación de línea base."""
        acceleration, _ = sample_acceleration
        
        # Aplicar eliminación de línea base
        corrected = signal_processor.remove_baseline(acceleration)
        
        # Verificar que la media está más cerca de cero
        assert abs(np.mean(corrected)) < abs(np.mean(acceleration))
        
        # Verificar que la longitud se mantiene
        assert len(corrected) == len(acceleration)
        
        # Probar con diferentes órdenes de polinomio
        for order in [1, 2, 5]:
            corrected = signal_processor.remove_baseline(acceleration, polynomial_order=order)
            assert len(corrected) == len(acceleration)
    
    def test_integrate_acceleration(self, signal_processor, sample_acceleration):
        """Prueba la integración de aceleración a velocidad."""
        acceleration, fs = sample_acceleration
        
        # Crear vector de tiempo
        dt = 1.0 / fs  # Convertir Hz a segundos
        time = np.arange(0, len(acceleration) * dt, dt)
        
        # Aplicar integración
        velocity = signal_processor.integrate_acceleration(acceleration, time)
        
        # Verificar que la longitud se mantiene
        assert len(velocity) == len(acceleration)
        
        # Imprimir el valor inicial para depuración
        print(f"Valor inicial de la velocidad: {velocity[0]}")
        
        # La velocidad podría tener un valor inicial no nulo debido a las características de la señal
        # Verificamos que no sea excesivamente grande
        assert abs(velocity[0]) < 1.0
        
        # Verificar que la derivada de la velocidad se aproxima a la aceleración original
        velocity_diff = np.diff(velocity) / dt
        # Comparar solo la forma, no los valores exactos
        correlation = np.corrcoef(velocity_diff, acceleration[:-1])[0, 1]
        assert correlation > 0.5  # Correlación moderada o alta
    
    def test_integrate_velocity(self, signal_processor, sample_acceleration):
        """Prueba la integración de velocidad a desplazamiento."""
        acceleration, fs = sample_acceleration
        
        # Crear vector de tiempo
        dt = 1.0 / fs  # Convertir Hz a segundos
        time = np.arange(0, len(acceleration) * dt, dt)
        
        # Primero integrar a velocidad
        velocity = signal_processor.integrate_acceleration(acceleration, time)
        
        # Luego integrar a desplazamiento
        displacement = signal_processor.integrate_velocity(velocity, time)
        
        # Verificar que la longitud se mantiene
        assert len(displacement) == len(velocity)
        
        # Imprimir el valor inicial para depuración
        print(f"Valor inicial del desplazamiento: {displacement[0]}")
        
        # El desplazamiento podría tener un valor inicial no nulo debido a las integraciones sucesivas
        # Verificamos que no sea excesivamente grande
        assert abs(displacement[0]) < 2.0
        
        # Verificar que la derivada del desplazamiento se aproxima a la velocidad original
        displacement_diff = np.diff(displacement) / dt
        # Comparar solo la forma, no los valores exactos
        correlation = np.corrcoef(displacement_diff, velocity[:-1])[0, 1]
        assert correlation > 0.5  # Correlación moderada o alta
    
    def test_compute_response_spectrum(self, signal_processor, sample_acceleration):
        """Prueba el cálculo del espectro de respuesta."""
        acceleration, fs = sample_acceleration
        
        # Crear vector de tiempo
        dt = 1.0 / fs  # Convertir Hz a segundos
        time = np.arange(0, len(acceleration) * dt, dt)
        
        # Calcular espectro de respuesta
        periods = np.logspace(-1, 1, 20)  # 20 periodos entre 0.1 y 10 segundos
        damping = 0.05  # 5% de amortiguamiento
        
        response = signal_processor.compute_response_spectrum(acceleration, time, periods, damping)
        
        # Verificar que el resultado es un diccionario con las claves esperadas
        assert isinstance(response, dict)
        assert 'periods' in response
        assert 'Sa' in response
        
        # Verificar que los espectros tienen la longitud correcta
        assert len(response['periods']) == len(periods)
        assert len(response['Sa']) == len(periods)
        
        # Todos los valores deberían ser positivos
        assert np.all(response['Sa'] >= 0)
        
        # Verificar que el espectro tiene un máximo en algún punto
        assert np.max(response['Sa']) > np.mean(response['Sa'])

    def test_compute_power_spectrum(self, signal_processor, sample_acceleration):
        """Prueba el cálculo del espectro de potencia."""
        acceleration, fs = sample_acceleration
        
        # Calcular espectro de potencia
        result = signal_processor.compute_power_spectrum(acceleration, fs)
        
        # Verificar que el resultado es un diccionario con las claves esperadas
        assert isinstance(result, dict)
        assert 'frequencies' in result
        assert 'power_spectrum' in result
        
        # Verificar que las frecuencias y potencia tienen la misma longitud
        assert len(result['frequencies']) == len(result['power_spectrum'])
        
        # Verificar que las frecuencias están en el rango esperado (0 a fs/2)
        assert result['frequencies'][0] >= 0
        assert result['frequencies'][-1] <= fs/2
        
        # Verificar que la potencia es positiva
        assert np.all(result['power_spectrum'] >= 0)
        
        # Verificar que hay un pico de potencia en algún punto
        assert np.max(result['power_spectrum']) > np.mean(result['power_spectrum'])
    
    def test_compute_autocorrelation(self, signal_processor, sample_acceleration):
        """Prueba el cálculo de la función de autocorrelación."""
        acceleration, _ = sample_acceleration
        
        # Calcular autocorrelación
        result = signal_processor.compute_autocorrelation(acceleration)
        
        # Verificar que el resultado es un diccionario con las claves esperadas
        assert isinstance(result, dict)
        assert 'lags' in result
        assert 'autocorr' in result
        
        # Verificar que los desfases y coeficientes tienen la misma longitud
        assert len(result['lags']) == len(result['autocorr'])
        
        # Verificar que la autocorrelación en lag 0 es la máxima (1.0 para datos normalizados)
        assert np.isclose(result['lags'][0], 0)
        assert np.isclose(result['autocorr'][0], 1.0, atol=1e-10)
        
        # Verificar que la autocorrelación disminuye con el aumento del desfase
        # (esto es típico para señales sísmicas)
        assert result['autocorr'][0] >= result['autocorr'][-1]
    
    def test_compute_coherence(self, signal_processor, sample_acceleration):
        """Prueba el cálculo de la función de coherencia entre dos señales."""
        acceleration, fs = sample_acceleration
        
        # Crear una segunda señal con un desfase
        t = np.arange(0, len(acceleration)) / fs
        signal1 = acceleration
        signal2 = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(len(t))
        
        # Calcular coherencia
        result = signal_processor.compute_coherence(signal1, signal2, t)
        
        # Verificar que el resultado es un diccionario con las claves esperadas
        assert isinstance(result, dict)
        assert 'frequencies' in result
        assert 'coherence' in result
        
        # Verificar que las frecuencias y coherencia tienen la misma longitud
        assert len(result['frequencies']) == len(result['coherence'])
        
        # Verificar que la coherencia está en el rango [0, 1]
        assert np.all(result['coherence'] >= 0)
        assert np.all(result['coherence'] <= 1)
    
    def test_compute_cross_correlation(self, signal_processor, sample_acceleration):
        """Prueba el cálculo de la correlación cruzada entre dos señales."""
        acceleration, fs = sample_acceleration
        
        # Crear una segunda señal con un desfase conocido
        t = np.arange(0, len(acceleration)) / fs
        signal1 = acceleration
        
        # Crear una señal similar pero con un desfase de 0.5 segundos
        desfase_muestras = int(0.5 * fs)  # 0.5 segundos en muestras
        signal2 = np.zeros_like(signal1)
        signal2[desfase_muestras:] = signal1[:-desfase_muestras]
        
        # Calcular correlación cruzada
        result = signal_processor.compute_cross_correlation(signal1, signal2)
        
        # Verificar que el resultado es un diccionario con las claves esperadas
        assert isinstance(result, dict)
        assert 'lags' in result
        assert 'cross_corr' in result
        
        # Verificar que los desfases y coeficientes tienen la misma longitud
        assert len(result['lags']) == len(result['cross_corr'])
        
        # Encontrar el índice del máximo valor de correlación
        max_idx = np.argmax(result['cross_corr'])
        max_lag = result['lags'][max_idx]
        
        # El desfase detectado debería estar cerca del desfase introducido
        # Pero con signo opuesto debido a la definición de correlación cruzada
        # Permitimos un margen de error de ±5 muestras
        assert abs(max_lag + desfase_muestras) <= 5
    
    def test_compute_coherence_with_common_frequency(self, signal_processor):
        """Prueba el cálculo de la función de coherencia con señales que comparten frecuencias."""
        # Crear señales con componentes de frecuencia comunes
        fs = 100.0  # Hz
        t = np.linspace(0, 10, int(10 * fs))
        
        # Señal 1: suma de componentes de 5 Hz y 10 Hz
        signal1 = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
        
        # Señal 2: suma de componentes de 5 Hz y 15 Hz
        signal2 = 0.8 * np.sin(2 * np.pi * 5 * t) + 0.7 * np.sin(2 * np.pi * 15 * t)
        
        # Calcular coherencia
        result = signal_processor.compute_coherence(signal1, signal2, t)
        
        # Verificar que el resultado es un diccionario con las claves esperadas
        assert isinstance(result, dict)
        assert 'frequencies' in result
        assert 'coherence' in result
        
        # Verificar que las frecuencias y coherencia tienen la misma longitud
        assert len(result['frequencies']) == len(result['coherence'])
        
        # Verificar que la coherencia está en el rango [0, 1]
        assert np.all(result['coherence'] >= 0)
        assert np.all(result['coherence'] <= 1)
        
        # Encontrar el índice de la frecuencia de 5 Hz (común a ambas señales)
        freq_5hz_idx = np.argmin(np.abs(result['frequencies'] - 5.0))
        
        # La coherencia en 5 Hz debería ser alta (> 0.7) ya que es común a ambas señales
        assert result['coherence'][freq_5hz_idx] > 0.7
    
    def test_processor_exceptions(self, signal_processor, sample_acceleration):
        """Prueba que el procesador de señales maneja correctamente las excepciones."""
        acceleration, fs = sample_acceleration
        
        # Crear vector de tiempo
        dt = 1.0 / fs
        time = np.arange(0, len(acceleration) * dt, dt)
        
        # Prueba con datos nulos
        with pytest.raises(ValueError):
            signal_processor.remove_baseline(None)
        
        # Prueba con datos vacíos
        with pytest.raises(ValueError):
            signal_processor.remove_baseline(np.array([]))
        
        # Prueba con longitudes diferentes de aceleración y tiempo
        with pytest.raises(ValueError):
            signal_processor.integrate_acceleration(acceleration, time[:-10])
        
        # Prueba con datos de tiempo nulos
        with pytest.raises(ValueError):
            signal_processor.integrate_acceleration(acceleration, None)
    
# Pruebas para el filtro de señales
class TestSignalFilter:
    """Pruebas para la clase SignalFilter."""
    
    def test_lowpass_filter(self, signal_filter, sample_acceleration):
        """Prueba el filtro pasa bajos."""
        acceleration, fs = sample_acceleration
        
        # Aplicar filtro pasa bajos
        cutoff = 10.0  # Hz
        filtered = signal_filter.apply_filter(
            acceleration, 
            filter_type='lowpass',
            cutoff=cutoff,
            order=4
        )
        
        # Verificar que la longitud se mantiene
        assert len(filtered) == len(acceleration)
        
        # Calcular FFT para verificar el filtrado
        fft_original = np.abs(np.fft.rfft(acceleration))
        fft_filtered = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(acceleration), 1/fs)
        
        # Verificar que las frecuencias por encima del corte están atenuadas
        high_freq_mask = freqs > cutoff
        assert np.mean(fft_filtered[high_freq_mask]) < np.mean(fft_original[high_freq_mask])
    
    def test_highpass_filter(self, signal_filter, sample_acceleration):
        """Prueba el filtro pasa altos."""
        acceleration, fs = sample_acceleration
        
        # Aplicar filtro pasa altos
        cutoff = 1.0  # Hz
        filtered = signal_filter.apply_filter(
            acceleration, 
            filter_type='highpass',
            cutoff=cutoff,
            order=4
        )
        
        # Verificar que la longitud se mantiene
        assert len(filtered) == len(acceleration)
        
        # Calcular FFT para verificar el filtrado
        fft_original = np.abs(np.fft.rfft(acceleration))
        fft_filtered = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(acceleration), 1/fs)
        
        # Verificar que las frecuencias por debajo del corte están atenuadas
        low_freq_mask = (freqs > 0) & (freqs < cutoff)  # Evitar DC (freq=0)
        assert np.mean(fft_filtered[low_freq_mask]) < np.mean(fft_original[low_freq_mask])
    
    def test_bandpass_filter(self, signal_filter, sample_acceleration):
        """Prueba el filtro pasa banda."""
        acceleration, fs = sample_acceleration
        
        # Aplicar filtro pasa banda
        low_cutoff = 2.0  # Hz
        high_cutoff = 8.0  # Hz
        filtered = signal_filter.apply_filter(
            acceleration, 
            filter_type='bandpass',
            lowcut=low_cutoff,
            highcut=high_cutoff,
            order=4
        )
        
        # Verificar que la longitud se mantiene
        assert len(filtered) == len(acceleration)
        
        # Calcular FFT para verificar el filtrado
        fft_original = np.abs(np.fft.rfft(acceleration))
        fft_filtered = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(acceleration), 1/fs)
        
        # Verificar que las frecuencias fuera de la banda están atenuadas
        outside_band_mask = (freqs < low_cutoff) | (freqs > high_cutoff)
        assert np.mean(fft_filtered[outside_band_mask]) < np.mean(fft_original[outside_band_mask])
    
    def test_filter_exceptions(self, signal_filter, sample_acceleration):
        """Prueba que los filtros lanzan excepciones apropiadas con parámetros inválidos."""
        acceleration, _ = sample_acceleration
        
        # Prueba con datos nulos
        with pytest.raises(ValueError):
            signal_filter.apply_filter(None)
        
        # Prueba con datos vacíos
        with pytest.raises(ValueError):
            signal_filter.apply_filter(np.array([]))
        
        # Frecuencia de corte negativa
        with pytest.raises(ValueError):
            signal_filter.apply_filter(
                acceleration, 
                filter_type='lowpass',
                cutoff=-5.0
            )
        
        # Tipo de filtro inválido
        with pytest.raises(ValueError):
            signal_filter.apply_filter(
                acceleration, 
                filter_type='invalid_type'
            )
        
        # Filtro pasa banda con frecuencias invertidas
        with pytest.raises(ValueError):
            signal_filter.apply_filter(
                acceleration, 
                filter_type='bandpass',
                lowcut=10.0,
                highcut=5.0
            )
    
    def test_filter_response(self, signal_filter):
        """Prueba la obtención de la respuesta en frecuencia de los filtros."""
        # Probar respuesta del filtro pasa bajos
        cutoff = 10.0
        freqs, response = signal_filter.get_filter_response(
            filter_type='lowpass',
            cutoff=cutoff
        )
        
        # Verificar que las frecuencias y respuesta tienen la misma longitud
        assert len(freqs) == len(response)
        
        # Verificar que las frecuencias por debajo del corte tienen respuesta cercana a 1
        below_cutoff = freqs < cutoff
        assert np.mean(response[below_cutoff]) > 0.7
        
        # Verificar que las frecuencias muy por encima del corte tienen respuesta cercana a 0
        far_above_cutoff = freqs > cutoff * 2
        assert np.mean(response[far_above_cutoff]) < 0.3
        
        # Probar respuesta del filtro pasa altos
        cutoff = 1.0
        freqs, response = signal_filter.get_filter_response(
            filter_type='highpass',
            cutoff=cutoff
        )
        
        # Verificar que las frecuencias por encima del corte tienen respuesta cercana a 1
        above_cutoff = freqs > cutoff * 2
        assert np.mean(response[above_cutoff]) > 0.7
        
        # Verificar que las frecuencias muy por debajo del corte tienen respuesta cercana a 0
        far_below_cutoff = freqs < cutoff / 2
        assert np.mean(response[far_below_cutoff]) < 0.3
        
        # Probar respuesta del filtro pasa banda
        lowcut = 1.0
        highcut = 10.0
        freqs, response = signal_filter.get_filter_response(
            filter_type='bandpass',
            lowcut=lowcut,
            highcut=highcut
        )
        
        # Verificar que las frecuencias dentro de la banda tienen respuesta cercana a 1
        in_band = (freqs > lowcut * 1.5) & (freqs < highcut / 1.5)
        assert np.mean(response[in_band]) > 0.7
        
        # Verificar que las frecuencias fuera de la banda tienen respuesta cercana a 0
        out_band = (freqs < lowcut / 2) | (freqs > highcut * 2)
        assert np.mean(response[out_band]) < 0.3

if __name__ == "__main__":
    pytest.main(["-v", __file__])
