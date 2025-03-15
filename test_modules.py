import unittest
import numpy as np
from ms_reader import MSReader
from fft_processor import FFTProcessor
from filters import SignalFilter
from event_detector import EventDetector
from data_exporter import DataExporter
import os
import tempfile
import shutil

class TestMSReader(unittest.TestCase):
    def setUp(self):
        # Crear datos de prueba
        self.test_data = np.random.randn(3000)  # 1000 muestras por canal
        self.sampling_rate = 100
        
        # Crear archivo .ms temporal
        self.temp_dir = tempfile.mkdtemp()
        self.ms_file = os.path.join(self.temp_dir, "test.ms")
        self.ss_file = os.path.join(self.temp_dir, "test.ss")
        
        # Escribir datos de prueba
        with open(self.ms_file, 'wb') as f:
            header = bytes([0] * 32)  # Header vacío
            f.write(header)
            self.test_data.astype(np.int32).tofile(f)
            
        # Escribir archivo .ss
        with open(self.ss_file, 'w') as f:
            f.write('sampling_rate="100"\n')
            f.write('sens="1.0"\n')
            
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_read_data(self):
        reader = MSReader(self.ms_file)
        data = reader.read_data()
        
        self.assertIn('time', data)
        self.assertIn('E', data)
        self.assertIn('N', data)
        self.assertIn('Z', data)
        self.assertEqual(len(data['time']), 1000)

class TestFFTProcessor(unittest.TestCase):
    def setUp(self):
        self.sampling_rate = 100
        self.duration = 10  # segundos
        t = np.linspace(0, self.duration, self.sampling_rate * self.duration)
        self.test_signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
        self.fft_proc = FFTProcessor(self.sampling_rate)
        
    def test_compute_fft(self):
        freqs, mags, phase = self.fft_proc.compute_fft(self.test_signal)
        
        # Verificar picos en 5 Hz y 10 Hz
        peak_freqs = freqs[np.argsort(mags)[-2:]]
        self.assertTrue(np.any(np.isclose(peak_freqs, 5, atol=0.5)))
        self.assertTrue(np.any(np.isclose(peak_freqs, 10, atol=0.5)))

class TestSignalFilter(unittest.TestCase):
    def setUp(self):
        self.sampling_rate = 100
        self.duration = 10
        t = np.linspace(0, self.duration, self.sampling_rate * self.duration)
        self.test_signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)
        self.filter = SignalFilter(self.sampling_rate)
        
    def test_lowpass_filter(self):
        filtered = self.filter.apply_filter(
            self.test_signal,
            filter_type='lowpass',
            cutoff=10
        )
        
        # Verificar que se atenuaron las frecuencias altas
        fft_orig = np.fft.rfft(self.test_signal)
        fft_filt = np.fft.rfft(filtered)
        freqs = np.fft.rfftfreq(len(self.test_signal), d=1/self.sampling_rate)
        
        # La amplitud a 20 Hz debe ser menor en la señal filtrada
        freq_20hz_idx = np.abs(freqs - 20).argmin()
        self.assertLess(np.abs(fft_filt[freq_20hz_idx]), np.abs(fft_orig[freq_20hz_idx]))

class TestEventDetector(unittest.TestCase):
    def setUp(self):
        self.sampling_rate = 100
        self.duration = 60
        t = np.linspace(0, self.duration, self.sampling_rate * self.duration)
        
        # Crear señal con eventos
        self.background = np.random.normal(0, 0.1, len(t))
        self.events = np.zeros_like(t)
        event_times = [10, 30, 50]
        for time in event_times:
            idx = int(time * self.sampling_rate)
            self.events[idx:idx+100] = 2.0 * np.sin(2 * np.pi * 5 * t[:100])
            
        self.test_signal = self.background + self.events
        self.detector = EventDetector(self.sampling_rate)
        
    def test_sta_lta_detection(self):
        events, ratio = self.detector.sta_lta(
            self.test_signal,
            sta_window=1.0,
            lta_window=10.0,
            trigger_ratio=3.0
        )
        
        # Debe detectar aproximadamente 3 eventos
        self.assertTrue(2 <= len(events) <= 4)

class TestDataExporter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter(self.temp_dir)
        
        # Crear datos de prueba
        t = np.linspace(0, 10, 1000)
        self.test_data = {
            'time': t,
            'E': np.sin(2 * np.pi * t),
            'N': np.cos(2 * np.pi * t),
            'Z': np.sin(4 * np.pi * t),
            'name': 'test_data'
        }
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_export_raw_data(self):
        # Probar exportación CSV
        csv_path = self.exporter.export_raw_data(self.test_data, 'test', 'csv')
        self.assertTrue(os.path.exists(csv_path))
        
        # Probar exportación Excel
        xlsx_path = self.exporter.export_raw_data(self.test_data, 'test', 'excel')
        self.assertTrue(os.path.exists(xlsx_path))

if __name__ == '__main__':
    unittest.main()
