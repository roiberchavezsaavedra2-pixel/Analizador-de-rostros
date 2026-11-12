# Analizador de rostro

Compara las proporciones geométricas de dos fotos usando MediaPipe Face Landmarker. El resultado es una estimación aproximada y no verifica identidad.

## Archivos

- `main.py`: programa principal.
- `face_landmarker.task`: modelo que necesita MediaPipe.
- `requirements.txt`: paquetes de Python necesarios.

## Ejecutarlo en Windows

1. Instala Python.
2. Descarga el repositorio con **Code → Download ZIP** y extrae la carpeta.
3. Abre PowerShell en esa carpeta y ejecuta:

   ```powershell
   python -m pip install -r requirements.txt
   python main.py
   ```

4. Coloca las dos fotos que quieras comparar en la misma carpeta que `main.py` y escribe sus nombres cuando el programa los solicite.

Las fotos y los reportes generados no se incluyen en este repositorio.
