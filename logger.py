"""
Módulo de logging para la aplicación de análisis sísmico.

Este módulo proporciona funciones para configurar y utilizar un sistema
de logging consistente en toda la aplicación.
"""

import logging
import os
from pathlib import Path
from datetime import datetime

# Configuración global del logger
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Directorio para los archivos de log
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def get_logger(name):
    """
    Obtiene un logger configurado para el módulo especificado.
    
    Args:
        name (str): Nombre del módulo o componente que utilizará el logger.
        
    Returns:
        logging.Logger: Logger configurado.
    """
    # Crear logger
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Evitar duplicación de handlers
    if not logger.handlers:
        # Crear handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)
        
        # Crear handler para archivo
        log_file = LOG_DIR / f"{datetime.now().strftime('%Y%m%d')}_{name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(LOG_LEVEL)
        
        # Crear formatter y añadirlo a los handlers
        formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Añadir handlers al logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

def set_log_level(level):
    """
    Establece el nivel de logging global.
    
    Args:
        level (int): Nivel de logging (e.g., logging.DEBUG, logging.INFO)
    """
    global LOG_LEVEL
    LOG_LEVEL = level
    
    # Actualizar nivel en root logger
    logging.getLogger().setLevel(level)
    
    # Actualizar nivel en todos los handlers existentes
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)

# Configuración inicial del root logger
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
