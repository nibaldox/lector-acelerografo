# Resumen de Pruebas Implementadas

## Pruebas para el Procesador de Señales

### Funcionalidades Básicas
1. **test_remove_baseline**: Verifica la corrección de línea base en señales sísmicas.
2. **test_integrate_acceleration**: Comprueba la integración de aceleración a velocidad.
3. **test_integrate_velocity**: Valida la integración de velocidad a desplazamiento.
4. **test_compute_response_spectrum**: Verifica el cálculo del espectro de respuesta.

### Análisis Espectral
5. **test_compute_power_spectrum**: Prueba el cálculo del espectro de potencia.
6. **test_compute_autocorrelation**: Valida el cálculo de la función de autocorrelación.

### Análisis de Múltiples Componentes
7. **test_compute_cross_correlation**: Verifica la correlación cruzada entre dos señales.
8. **test_compute_coherence**: Prueba el cálculo de la función de coherencia.
9. **test_compute_coherence_with_common_frequency**: Valida la coherencia con señales que comparten frecuencias.

### Manejo de Excepciones
10. **test_processor_exceptions**: Verifica el manejo de excepciones para entradas inválidas.

## Pruebas para el Filtro de Señales

1. **test_filter_lowpass**: Verifica el filtro pasa bajos.
2. **test_filter_highpass**: Prueba el filtro pasa altos.
3. **test_filter_bandpass**: Valida el filtro pasa banda.
4. **test_filter_response**: Verifica la respuesta en frecuencia de los filtros.

## Mejoras Implementadas

1. **Validación de Datos**: Se han añadido pruebas para verificar que los métodos validan correctamente los datos de entrada.
2. **Manejo de Excepciones**: Se han implementado pruebas para verificar que se lanzan las excepciones adecuadas cuando los datos son inválidos.
3. **Verificación de Respuesta**: Se han añadido pruebas para verificar que los filtros tienen la respuesta en frecuencia esperada.
4. **Análisis Espectral**: Se han implementado pruebas para verificar el cálculo correcto de espectros de potencia y funciones de correlación.

## Próximos Pasos

1. Implementar pruebas para la función de coherencia con diferentes parámetros.
2. Añadir pruebas para el cálculo de la respuesta combinada de múltiples componentes.
3. Implementar pruebas para el cálculo de la órbita de partículas.
4. Añadir pruebas para la diferencia de fase entre componentes.
