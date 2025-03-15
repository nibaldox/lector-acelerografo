import unittest
import numpy as np
import os
import tempfile
import shutil
import streamlit as st
from ms_reader import MSReader
from fft_processor import FFTProcessor
from filters import SignalFilter
from event_detector import EventDetector
from data_exporter import DataExporter

class TestFullWorkflow(unittest.TestCase):
    def setUp(self):
        # Crear directorio temporal para pruebas
        self.test_dir = tempfile.mkdtemp()
        self.export_dir = os.path.join(self.test_dir, "exports")
        os.makedirs(self.export_dir)
        
        # Crear datos de prueba
        self.sampling_rate = 100
        self.duration = 60  # segundos
        t = np.linspace(0, self.duration, self.sampling_rate * self.duration)
        
        # Señal con eventos y ruido
        background = np.random.normal(0, 0.1, len(t))
        events = np.zeros_like(t)
        event_times = [10, 30, 50]
        for time in event_times:
            idx = int(time * self.sampling_rate)
            events[idx:idx+100] = 2.0 * np.sin(2 * np.pi * 5 * t[:100])
        
        # Agregar ruido de alta frecuencia
        noise = 0.1 * np.sin(2 * np.pi * 40 * t)
        
        # Señal final
        self.test_signal = background + events + noise
        
        # Crear archivos de prueba
        self.ms_file = os.path.join(self.test_dir, "test.ms")
        self.ss_file = os.path.join(self.test_dir, "test.ss")
        
        # Escribir datos en formato .ms
        with open(self.ms_file, 'wb') as f:
            header = bytes([0] * 32)
            f.write(header)
            # Repetir la señal para los tres canales
            np.concatenate([self.test_signal, self.test_signal, self.test_signal]).astype(np.int32).tofile(f)
        
        # Escribir archivo de configuración .ss
        with open(self.ss_file, 'w') as f:
            f.write('sampling_rate="100"\n')
            f.write('sens="1.0"\n')
            f.write('gain="1.0"\n')
            
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_complete_workflow(self):
        """Prueba el flujo completo de trabajo"""
        
        # 1. Leer datos
        reader = MSReader(self.ms_file)
        data = reader.read_data()
        self.assertIn('time', data)
        self.assertIn('E', data)
        self.assertIn('N', data)
        self.assertIn('Z', data)
        
        # 2. Procesar FFT
        fft_proc = FFTProcessor(self.sampling_rate)
        frequencies, magnitudes, phase = fft_proc.compute_fft(data['E'])
        
        # Verificar que detecta la frecuencia de los eventos (5 Hz)
        peak_freqs = frequencies[np.argsort(magnitudes)[-5:]]
        self.assertTrue(np.any(np.isclose(peak_freqs, 5, atol=1)))
        
        # 3. Aplicar filtro
        signal_filter = SignalFilter(self.sampling_rate)
        filtered_data = signal_filter.apply_filter(
            data['E'],
            filter_type='lowpass',
            cutoff=20
        )
        
        # Verificar que el filtro reduce el ruido de alta frecuencia
        fft_orig = np.fft.rfft(data['E'])
        fft_filt = np.fft.rfft(filtered_data)
        freqs = np.fft.rfftfreq(len(data['E']), d=1/self.sampling_rate)
        freq_40hz_idx = np.abs(freqs - 40).argmin()
        self.assertLess(np.abs(fft_filt[freq_40hz_idx]), np.abs(fft_orig[freq_40hz_idx]))
        
        # 4. Detectar eventos
        detector = EventDetector(self.sampling_rate)
        events, ratio = detector.sta_lta(
            filtered_data,
            sta_window=1.0,
            lta_window=10.0,
            trigger_ratio=3.0
        )
        
        # Verificar que detecta aproximadamente 3 eventos
        self.assertTrue(2 <= len(events) <= 4)
        
        # 5. Exportar resultados
        exporter = DataExporter(self.export_dir)
        
        # Exportar datos crudos
        export_data = {
            'time': data['time'],
            'E': data['E'],
            'N': data['N'],
            'Z': data['Z'],
            'name': 'test_data'
        }
        csv_path = exporter.export_raw_data(export_data, 'test_export', 'csv')
        self.assertTrue(os.path.exists(csv_path))
        
        # Exportar resultados del análisis
        fft_results = {
            'frequencies': frequencies,
            'magnitudes': magnitudes,
            'phase': phase
        }
        json_path = exporter.export_analysis_results(
            data,
            'fft',
            fft_results,
            'test_fft'
        )
        self.assertTrue(os.path.exists(json_path))

class TestStreamlitIntegration(unittest.TestCase):
    def setUp(self):
        # Configurar entorno de prueba similar a TestFullWorkflow
        self.test_dir = tempfile.mkdtemp()
        # ... (configuración similar a TestFullWorkflow)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_streamlit_components(self):
        """
        Verificar que los componentes de Streamlit funcionan correctamente
        Nota: Estas pruebas son más limitadas debido a la naturaleza de Streamlit
        """
        try:
            # Probar creación de gráficos
            fft_proc = FFTProcessor(100)
            test_signal = np.random.randn(1000)
            frequencies, magnitudes, _ = fft_proc.compute_fft(test_signal)
            
            fig = fft_proc.plot_spectrum(frequencies, magnitudes)
            self.assertIsNotNone(fig)
            
            # Verificar que la figura tiene los componentes esperados
            self.assertIn('data', fig.to_dict())
            self.assertIn('layout', fig.to_dict())
            
        except Exception as e:
            self.fail(f"Error en la integración con Streamlit: {str(e)}")

if __name__ == '__main__':
    unittest.main()
