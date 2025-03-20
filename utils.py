"""
Módulo de utilidades para funciones compartidas en la aplicación.

Este módulo contiene funciones de utilidad que son utilizadas por diferentes
componentes de la aplicación, promoviendo la reutilización de código y
evitando la duplicación.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional, Any
from logger import get_logger

# Configurar logger
logger = get_logger("Utils")

def ensure_directory_exists(directory_path: Union[str, Path]) -> Path:
    """
    Asegura que un directorio exista, creándolo si es necesario.
    
    Args:
        directory_path (Union[str, Path]): Ruta del directorio a verificar/crear
        
    Returns:
        Path: Objeto Path del directorio
        
    Raises:
        PermissionError: Si no se tienen permisos para crear el directorio
    """
    path = Path(directory_path)
    if not path.exists():
        logger.info(f"Creando directorio: {path}")
        try:
            path.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            logger.error(f"Error de permisos al crear directorio {path}: {str(e)}")
            raise
    return path

def safe_file_path(base_dir: Union[str, Path], filename: str) -> Path:
    """
    Crea una ruta de archivo segura, evitando inyecciones de ruta.
    
    Args:
        base_dir (Union[str, Path]): Directorio base
        filename (str): Nombre del archivo
        
    Returns:
        Path: Ruta segura al archivo
    """
    # Sanitizar el nombre del archivo
    safe_name = os.path.basename(filename)
    return Path(base_dir) / safe_name

def generate_unique_filename(base_path: Union[str, Path], extension: str = "") -> Path:
    """
    Genera un nombre de archivo único basado en la fecha/hora actual.
    
    Args:
        base_path (Union[str, Path]): Directorio base
        extension (str): Extensión del archivo (con o sin punto)
        
    Returns:
        Path: Ruta al archivo con nombre único
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Asegurar que la extensión tenga un punto
    if extension and not extension.startswith("."):
        extension = f".{extension}"
        
    filename = f"data_{timestamp}{extension}"
    return Path(base_path) / filename

def validate_numeric_array(array: np.ndarray, 
                          min_length: int = 1, 
                          allow_nan: bool = False,
                          allow_inf: bool = False) -> bool:
    """
    Valida que un array numpy sea numérico y cumpla con ciertos criterios.
    
    Args:
        array (numpy.ndarray): Array a validar
        min_length (int): Longitud mínima requerida
        allow_nan (bool): Si se permiten valores NaN
        allow_inf (bool): Si se permiten valores infinitos
        
    Returns:
        bool: True si el array es válido, False en caso contrario
    """
    if array is None or not isinstance(array, np.ndarray):
        logger.warning("El array proporcionado no es un numpy.ndarray válido")
        return False
        
    if len(array) < min_length:
        logger.warning(f"El array tiene longitud {len(array)}, menor que el mínimo requerido {min_length}")
        return False
        
    if not np.issubdtype(array.dtype, np.number):
        logger.warning(f"El array no es de tipo numérico (dtype: {array.dtype})")
        return False
        
    if not allow_nan and np.isnan(array).any():
        logger.warning("El array contiene valores NaN")
        return False
        
    if not allow_inf and np.isinf(array).any():
        logger.warning("El array contiene valores infinitos")
        return False
        
    return True

def resample_signal(signal: np.ndarray, 
                   original_fs: float, 
                   target_fs: float) -> Tuple[np.ndarray, float]:
    """
    Remuestrea una señal a una nueva frecuencia de muestreo.
    
    Args:
        signal (numpy.ndarray): Señal original
        original_fs (float): Frecuencia de muestreo original (Hz)
        target_fs (float): Frecuencia de muestreo objetivo (Hz)
        
    Returns:
        Tuple[numpy.ndarray, float]: Señal remuestreada y nueva frecuencia de muestreo
        
    Raises:
        ValueError: Si los parámetros de entrada no son válidos
    """
    if not validate_numeric_array(signal):
        raise ValueError("La señal de entrada no es válida")
        
    if original_fs <= 0 or target_fs <= 0:
        raise ValueError("Las frecuencias de muestreo deben ser positivas")
    
    # Si las frecuencias son iguales, no es necesario remuestrear
    if np.isclose(original_fs, target_fs):
        return signal, original_fs
    
    # Calcular el número de muestras en la señal remuestreada
    original_duration = len(signal) / original_fs
    n_samples = int(original_duration * target_fs)
    
    # Crear vector de tiempo original y nuevo
    t_original = np.linspace(0, original_duration, len(signal))
    t_new = np.linspace(0, original_duration, n_samples)
    
    # Interpolar la señal
    resampled_signal = np.interp(t_new, t_original, signal)
    
    logger.info(f"Señal remuestreada de {original_fs} Hz a {target_fs} Hz")
    return resampled_signal, target_fs

def calculate_signal_statistics(signal: np.ndarray) -> Dict[str, float]:
    """
    Calcula estadísticas básicas de una señal.
    
    Args:
        signal (numpy.ndarray): Señal de entrada
        
    Returns:
        Dict[str, float]: Diccionario con estadísticas (min, max, mean, std, rms)
        
    Raises:
        ValueError: Si la señal de entrada no es válida
    """
    if not validate_numeric_array(signal):
        raise ValueError("La señal de entrada no es válida")
    
    # Calcular estadísticas
    min_val = np.min(signal)
    max_val = np.max(signal)
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    rms_val = np.sqrt(np.mean(np.square(signal)))
    peak_to_peak = max_val - min_val
    
    return {
        'min': min_val,
        'max': max_val,
        'mean': mean_val,
        'std': std_val,
        'rms': rms_val,
        'peak_to_peak': peak_to_peak,
        'abs_max': max(abs(min_val), abs(max_val))
    }

def save_plot(fig: plt.Figure, 
             filename: str, 
             output_dir: Union[str, Path] = "output",
             dpi: int = 300,
             formats: List[str] = ["png", "pdf"]) -> List[Path]:
    """
    Guarda una figura de matplotlib en múltiples formatos.
    
    Args:
        fig (matplotlib.figure.Figure): Figura a guardar
        filename (str): Nombre base del archivo (sin extensión)
        output_dir (Union[str, Path]): Directorio de salida
        dpi (int): Resolución en puntos por pulgada
        formats (List[str]): Lista de formatos a guardar
        
    Returns:
        List[Path]: Lista de rutas a los archivos guardados
        
    Raises:
        ValueError: Si los parámetros de entrada no son válidos
    """
    if not isinstance(fig, plt.Figure):
        raise ValueError("El parámetro fig debe ser una instancia de matplotlib.figure.Figure")
    
    # Asegurar que el directorio de salida exista
    output_path = ensure_directory_exists(output_dir)
    
    # Sanitizar el nombre del archivo
    safe_name = os.path.basename(filename)
    
    saved_files = []
    for fmt in formats:
        # Eliminar el punto si está presente
        fmt = fmt.lstrip('.')
        
        # Crear ruta completa
        file_path = output_path / f"{safe_name}.{fmt}"
        
        try:
            fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
            saved_files.append(file_path)
            logger.info(f"Figura guardada como {file_path}")
        except Exception as e:
            logger.error(f"Error al guardar figura como {file_path}: {str(e)}")
    
    return saved_files

def format_time_axis(ax: plt.Axes, time_unit: str = "s") -> None:
    """
    Formatea el eje X de tiempo en una gráfica.
    
    Args:
        ax (matplotlib.axes.Axes): Ejes a formatear
        time_unit (str): Unidad de tiempo (s, ms, min, h)
    """
    if time_unit not in ["s", "ms", "min", "h"]:
        logger.warning(f"Unidad de tiempo no reconocida: {time_unit}. Usando 's'.")
        time_unit = "s"
    
    unit_labels = {
        "s": "Tiempo (s)",
        "ms": "Tiempo (ms)",
        "min": "Tiempo (min)",
        "h": "Tiempo (h)"
    }
    
    ax.set_xlabel(unit_labels[time_unit])
    ax.grid(True, linestyle='--', alpha=0.7)
    
def format_frequency_axis(ax: plt.Axes, freq_unit: str = "Hz") -> None:
    """
    Formatea el eje X de frecuencia en una gráfica.
    
    Args:
        ax (matplotlib.axes.Axes): Ejes a formatear
        freq_unit (str): Unidad de frecuencia (Hz, kHz)
    """
    if freq_unit not in ["Hz", "kHz"]:
        logger.warning(f"Unidad de frecuencia no reconocida: {freq_unit}. Usando 'Hz'.")
        freq_unit = "Hz"
    
    unit_labels = {
        "Hz": "Frecuencia (Hz)",
        "kHz": "Frecuencia (kHz)"
    }
    
    ax.set_xlabel(unit_labels[freq_unit])
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Si es un eje logarítmico, ajustar la cuadrícula
    if ax.get_xscale() == 'log':
        ax.grid(True, which='minor', linestyle=':', alpha=0.4)
