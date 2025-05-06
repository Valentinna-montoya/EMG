import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import sounddevice as sd
from matplotlib import cm
from PIL import Image
import cv2  # <-- ¡Este es el que faltaba!
import time

# Configuración
DURATION = 0.1  # segundos por bloque
FS = 44100
IMG_HEIGHT = 480
IMG_WIDTH = 1280
SIGNAL_HEIGHT = 80  # altura de la franja de señal
NFFT = 1024
NOVERLAP = 1000

# Imagen del espectrograma deslizante
scroll_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

# Número de ciclos
n_cycles = 500 
for _ in range(n_cycles):
    # Captura de audio
    audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    signal = audio[:, 0]

    # Calcular espectrograma
    f, t, Sxx = spectrogram(signal, fs=FS, nperseg=NFFT, noverlap=NOVERLAP)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)
    Sxx_norm = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min() + 1e-10)

    # Convertir espectrograma en imagen de color
    colormap = cm.inferno(Sxx_norm)
    colormap_img = (colormap[:, :, :3] * 255).astype(np.uint8)
    new_column = np.array(Image.fromarray(colormap_img).resize((10, IMG_HEIGHT - SIGNAL_HEIGHT)))

    # Desplazar espectrograma hacia la izquierda
    scroll_img[SIGNAL_HEIGHT:, :-10] = scroll_img[SIGNAL_HEIGHT:, 10:]
    scroll_img[SIGNAL_HEIGHT:, -10:] = new_column

    # Dibujar la forma de onda (parte superior)
    wave_section = np.zeros((SIGNAL_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    normalized_signal = ((signal - signal.min()) / (signal.max() - signal.min() + 1e-10))  # Normalizar
    waveform = (1 - normalized_signal) * (SIGNAL_HEIGHT - 1)
    waveform = waveform.astype(np.int32)
    
    # Dibujar la forma de onda
    for i in range(len(waveform) - 1):
        x1 = int(i * IMG_WIDTH / len(waveform))
        x2 = int((i + 1) * IMG_WIDTH / len(waveform))
        y1 = waveform[i]
        y2 = waveform[i + 1]
        cv_color = (255, 255, 255)  # blanco
        cv2.line(wave_section, (x1, y1), (x2, y2), cv_color, 1)

    # Unir la forma de onda con el espectrograma
    combined_img = np.vstack((wave_section, scroll_img[SIGNAL_HEIGHT:]))

    # Mostrar en pantalla con OpenCV
    cv2.imshow("Espectrograma y Señal", combined_img)

    # Esperar una pequeña cantidad de tiempo para la animación
    time.sleep(0.01)

    # Salir si se presiona la tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cerrar la ventana de OpenCV
cv2.destroyAllWindows()
