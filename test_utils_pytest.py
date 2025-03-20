#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el módulo de utilidades utilizando pytest.
"""

import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pytest

from utils import (
    ensure_directory_exists,
    safe_file_path,
    generate_unique_filename,
    validate_numeric_array,
    resample_signal,
    calculate_signal_statistics,
    save_plot,
    format_time_axis,
    format_frequency_axis
)

# Fixtures para configurar y limpiar el entorno de pruebas
@pytest.fixture
def temp_dir():
    """Crea un directorio temporal para las pruebas."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Limpieza después de las pruebas
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_signal():
    """Genera una señal de ejemplo para pruebas."""
    # Crear una señal sinusoidal simple
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    return signal

@pytest.fixture
def sample_plot():
    """Crea una figura de matplotlib para pruebas."""
    fig, ax = plt.subplots()
    t = np.linspace(0, 1, 1000)
    ax.plot(t, np.sin(2 * np.pi * 5 * t))
    ax.set_title("Gráfico de prueba")
    return fig, ax

# Pruebas para funciones de gestión de archivos y directorios
class TestFileManagement:
    """Pruebas para las funciones de gestión de archivos y directorios."""

    def test_ensure_directory_exists(self, temp_dir):
        """Prueba que ensure_directory_exists crea directorios correctamente."""
        # Probar con un directorio que no existe
        test_dir = Path(temp_dir) / "test_dir"
        result = ensure_directory_exists(test_dir)
        assert result.exists()
        assert result.is_dir()
        
        # Probar con un directorio que ya existe
        result2 = ensure_directory_exists(test_dir)
        assert result2 == result
        
        # Probar con un directorio anidado
        nested_dir = Path(temp_dir) / "parent" / "child" / "grandchild"
        result3 = ensure_directory_exists(nested_dir)
        assert result3.exists()
        assert result3.is_dir()

    def test_safe_file_path(self, temp_dir):
        """Prueba que safe_file_path crea rutas seguras."""
        # Caso normal
        result = safe_file_path(temp_dir, "archivo.txt")
        assert result == Path(temp_dir) / "archivo.txt"
        
        # Intento de inyección de ruta
        result = safe_file_path(temp_dir, "../../../etc/passwd")
        assert result.parent == Path(temp_dir)
        assert "passwd" in result.name
        assert "../" not in str(result)

    def test_generate_unique_filename(self, temp_dir):
        """Prueba que generate_unique_filename genera nombres únicos."""
        # Sin extensión
        result1 = generate_unique_filename(temp_dir)
        assert result1.parent == Path(temp_dir)
        
        # Con extensión sin punto
        result2 = generate_unique_filename(temp_dir, "txt")
        assert result2.suffix == ".txt"
        
        # Con extensión con punto
        result3 = generate_unique_filename(temp_dir, ".csv")
        assert result3.suffix == ".csv"
        
        # Verificar que los nombres son diferentes
        assert result1 != result2 != result3

# Pruebas para funciones de validación y procesamiento de datos
class TestDataProcessing:
    """Pruebas para las funciones de validación y procesamiento de datos."""

    def test_validate_numeric_array(self):
        """Prueba que validate_numeric_array valida correctamente los arrays."""
        # Array válido
        valid_array = np.array([1.0, 2.0, 3.0])
        assert validate_numeric_array(valid_array) is True
        
        # Array con longitud insuficiente
        short_array = np.array([1.0])
        assert validate_numeric_array(short_array, min_length=2) is False
        
        # Array con NaN
        nan_array = np.array([1.0, np.nan, 3.0])
        assert validate_numeric_array(nan_array) is False
        assert validate_numeric_array(nan_array, allow_nan=True) is True
        
        # Array con infinito
        inf_array = np.array([1.0, np.inf, 3.0])
        assert validate_numeric_array(inf_array) is False
        assert validate_numeric_array(inf_array, allow_inf=True) is True
        
        # Array no numérico
        non_numeric = np.array(["a", "b", "c"])
        assert validate_numeric_array(non_numeric) is False

    def test_resample_signal(self, sample_signal):
        """Prueba que resample_signal remuestrea correctamente las señales."""
        # Remuestrear a una frecuencia más baja
        original_fs = 1000
        target_fs = 500
        resampled, new_fs = resample_signal(sample_signal, original_fs, target_fs)
        
        # Verificar la nueva frecuencia
        assert new_fs == target_fs
        
        # Verificar la longitud de la señal remuestreada
        expected_length = int(len(sample_signal) * (target_fs / original_fs))
        assert len(resampled) == expected_length
        
        # Remuestrear a una frecuencia más alta
        target_fs = 2000
        resampled, new_fs = resample_signal(sample_signal, original_fs, target_fs)
        assert new_fs == target_fs
        expected_length = int(len(sample_signal) * (target_fs / original_fs))
        assert len(resampled) == expected_length

    def test_calculate_signal_statistics(self, sample_signal):
        """Prueba que calculate_signal_statistics calcula estadísticas correctamente."""
        stats = calculate_signal_statistics(sample_signal)
        
        # Verificar que todas las estadísticas esperadas están presentes
        expected_keys = ['min', 'max', 'mean', 'std', 'rms', 'peak_to_peak', 'abs_max']
        for key in expected_keys:
            assert key in stats
        
        # Verificar algunos valores específicos
        assert stats['min'] == pytest.approx(np.min(sample_signal))
        assert stats['max'] == pytest.approx(np.max(sample_signal))
        assert stats['mean'] == pytest.approx(np.mean(sample_signal))
        assert stats['peak_to_peak'] == pytest.approx(np.max(sample_signal) - np.min(sample_signal))
        assert stats['abs_max'] == pytest.approx(max(abs(np.min(sample_signal)), abs(np.max(sample_signal))))

# Pruebas para funciones de visualización y gráficos
class TestVisualization:
    """Pruebas para las funciones de visualización y gráficos."""

    def test_save_plot(self, sample_plot, temp_dir):
        """Prueba que save_plot guarda figuras correctamente."""
        fig, _ = sample_plot
        
        # Guardar en un solo formato
        saved_files = save_plot(fig, "test_plot", temp_dir, formats=["png"])
        assert len(saved_files) == 1
        assert saved_files[0].exists()
        assert saved_files[0].suffix == ".png"
        
        # Guardar en múltiples formatos
        saved_files = save_plot(fig, "test_plot_multi", temp_dir, formats=["png", "pdf"])
        assert len(saved_files) == 2
        assert all(f.exists() for f in saved_files)
        assert any(f.suffix == ".png" for f in saved_files)
        assert any(f.suffix == ".pdf" for f in saved_files)

    def test_format_time_axis(self, sample_plot):
        """Prueba que format_time_axis formatea correctamente los ejes de tiempo."""
        _, ax = sample_plot
        
        # Formatear con unidades de segundos
        format_time_axis(ax, time_unit="s")
        assert "Tiempo" in ax.get_xlabel()
        assert "s" in ax.get_xlabel()
        
        # Formatear con unidades de milisegundos
        format_time_axis(ax, time_unit="ms")
        assert "Tiempo" in ax.get_xlabel()
        assert "ms" in ax.get_xlabel()

    def test_format_frequency_axis(self, sample_plot):
        """Prueba que format_frequency_axis formatea correctamente los ejes de frecuencia."""
        _, ax = sample_plot
        
        # Formatear con unidades de Hz
        format_frequency_axis(ax, freq_unit="Hz")
        assert "Frecuencia" in ax.get_xlabel()
        assert "Hz" in ax.get_xlabel()
        
        # Formatear con unidades de kHz
        format_frequency_axis(ax, freq_unit="kHz")
        assert "Frecuencia" in ax.get_xlabel()
        assert "kHz" in ax.get_xlabel()

if __name__ == "__main__":
    pytest.main(["-v", __file__])
