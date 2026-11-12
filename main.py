"""
Analizador de rostro (versión 2)
--------------------------------
Carga una foto, detecta el rostro con MediaPipe (Face Landmarker), calcula
medidas y proporciones faciales y muestra un reporte ordenado por secciones.

Novedades de esta versión:
    - Verifica si la foto es de frente (giro e inclinación de la cabeza).
    - Reporte organizado: calidad de la foto, resumen, rasgos, colores y
      tabla de medidas con una explicación de cada valor.
    - Permite guardar el reporte en un archivo de texto.

Requisitos:
    python -m pip install opencv-python mediapipe

Archivos necesarios en la misma carpeta que este programa:
    - face_landmarker.task  (modelo de MediaPipe)
    - la foto a analizar (por ejemplo rostro1.jpg)
"""

import math
import os

import cv2
import mediapipe as mp
import numpy as np

# Rutas relativas a la carpeta donde está este archivo
CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(CARPETA, "face_landmarker.task")

ANCHO_REPORTE = 80

# Pares de puntos (izquierda, derecha) para medir la simetría del rostro
PARES_SIMETRIA = [
    (33, 263),    # esquinas externas de los ojos
    (133, 362),   # esquinas internas de los ojos
    (129, 358),   # aletas de la nariz
    (61, 291),    # comisuras de la boca
    (234, 454),   # mejillas
    (172, 397),   # mandíbula
]

# Puntos de cada ojo para calcular la apertura (EAR): 6 puntos por ojo
OJO_IZQ = (33, 160, 158, 133, 153, 144)
OJO_DER = (362, 385, 387, 263, 373, 380)

# Cómo se muestra cada medida en la tabla:
# clave -> (nombre legible, formato del valor, explicación breve)
ETIQUETAS = {
    "relacion_alto_ancho": ("Alto / ancho del rostro", "{:.2f}", "más alto = rostro más alargado"),
    "frente_vs_mejillas": ("Frente vs. mejillas", "{:.0%}", "ancho de la frente / ancho de mejillas"),
    "mandibula_vs_mejillas": ("Mandíbula vs. mejillas", "{:.0%}", "cerca de 100% = mandíbula marcada"),
    "menton_vs_mejillas": ("Ancho del mentón", "{:.0%}", "respecto al ancho de las mejillas"),
    "tercio_superior": ("Tercio superior (frente)", "{:.0%}", "ideal aprox. 33%"),
    "tercio_medio": ("Tercio medio (nariz)", "{:.0%}", "ideal aprox. 33%"),
    "tercio_inferior": ("Tercio inferior (boca-mentón)", "{:.0%}", "ideal aprox. 33%"),
    "asimetria_pct": ("Asimetría izquierda-derecha", "{:.1f}%", "0% = perfectamente simétrico"),
    "separacion_ojos": ("Separación entre ojos", "{:.2f}", "1.00 = distancia de un ojo"),
    "apertura_ojos": ("Apertura de los ojos", "{:.2f}", "bajo = entrecerrados, alto = abiertos"),
    "inclinacion_ojos_grados": ("Inclinación de los ojos", "{:+.1f}°", "positivo = esquina externa más alta"),
    "arco_cejas": ("Arco de las cejas", "{:.2f}", "bajo = rectas, alto = arqueadas"),
    "ceja_a_ojo": ("Distancia ceja-ojo", "{:.1%}", "respecto al alto del rostro"),
    "nariz_vs_rostro": ("Ancho de la nariz", "{:.0%}", "respecto al ancho del rostro"),
    "largo_nariz": ("Largo de la nariz", "{:.0%}", "respecto al alto del rostro"),
    "boca_vs_rostro": ("Ancho de la boca", "{:.0%}", "respecto al ancho del rostro"),
    "grosor_labios_vs_boca": ("Grosor de los labios", "{:.0%}", "respecto al ancho de la boca"),
    "labio_inf_vs_sup": ("Labio inferior / superior", "{:.2f}", "mayor a 1 = inferior más grueso"),
    "apertura_boca": ("Apertura de la boca", "{:.1%}", "respecto al alto del rostro"),
    "elevacion_comisuras": ("Elevación de comisuras", "{:.3f}", "mayor = sonrisa más marcada"),
    "inclinacion_grados": ("Inclinación de la cabeza", "{:+.1f}°", "0° = cabeza recta"),
    "giro_cabeza_pct": ("Giro de la cabeza", "{:.1f}%", "0% = mirando de frente"),
}

# Cómo se agrupan las medidas en la tabla
GRUPOS = [
    ("Forma y proporciones", ["relacion_alto_ancho", "frente_vs_mejillas",
                              "mandibula_vs_mejillas", "menton_vs_mejillas",
                              "tercio_superior", "tercio_medio", "tercio_inferior",
                              "asimetria_pct"]),
    ("Ojos y cejas", ["separacion_ojos", "apertura_ojos", "inclinacion_ojos_grados",
                      "arco_cejas", "ceja_a_ojo"]),
    ("Nariz", ["nariz_vs_rostro", "largo_nariz"]),
    ("Boca y labios", ["boca_vs_rostro", "grosor_labios_vs_boca",
                       "labio_inf_vs_sup", "apertura_boca", "elevacion_comisuras"]),
    ("Postura de la cabeza", ["inclinacion_grados", "giro_cabeza_pct"]),
]


# ---------------------------------------------------------------------------
# Entrada y detección
# ---------------------------------------------------------------------------

def pedir_ruta_foto():
    """Pide el nombre de la foto y devuelve su ruta completa."""
    nombre = input("Nombre de la foto (con extensión): ").strip()
    return os.path.join(CARPETA, nombre)


def cargar_imagen(ruta):
    """Carga la imagen con OpenCV. Devuelve None si no existe o no se puede leer."""
    if not os.path.exists(ruta):
        print(f"No se encontró la imagen: {ruta}")
        return None
    imagen = cv2.imread(ruta)
    if imagen is None:
        print("El archivo existe pero no se pudo leer como imagen.")
    return imagen


def detectar_rostro(ruta_imagen):
    """Detecta rostros en la imagen y devuelve el resultado de MediaPipe."""
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    opciones = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=RUTA_MODELO),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=2,  # pedimos hasta 2 para poder avisar si hay más de uno
    )
    imagen_mp = mp.Image.create_from_file(ruta_imagen)
    with FaceLandmarker.create_from_options(opciones) as detector:
        return detector.detect(imagen_mp)


# ---------------------------------------------------------------------------
# Funciones geométricas
# ---------------------------------------------------------------------------

def distancia(p1, p2, ancho, alto):
    """Distancia en píxeles entre dos puntos faciales."""
    dx = (p1.x - p2.x) * ancho
    dy = (p1.y - p2.y) * alto
    return math.hypot(dx, dy)


def distancia_a_recta(p, a, b, ancho, alto):
    """Distancia en píxeles del punto p a la recta que pasa por a y b."""
    px, py = p.x * ancho, p.y * alto
    ax, ay = a.x * ancho, a.y * alto
    bx, by = b.x * ancho, b.y * alto
    base = math.hypot(bx - ax, by - ay)
    if base == 0:
        return 0.0
    return abs((bx - ax) * (ay - py) - (ax - px) * (by - ay)) / base


def apertura_ojo(puntos, indices, ancho, alto):
    """EAR (eye aspect ratio): alto del ojo / ancho del ojo.
    Valores bajos = ojo cerrado o entrecerrado."""
    p1, p2, p3, p4, p5, p6 = (puntos[i] for i in indices)
    vertical = distancia(p2, p6, ancho, alto) + distancia(p3, p5, ancho, alto)
    horizontal = 2 * distancia(p1, p4, ancho, alto)
    return vertical / horizontal


def inclinacion_ojo(puntos, interno, externo, ancho, alto):
    """Ángulo (grados) de la línea esquina interna -> esquina externa del ojo.
    Positivo = la esquina externa está más arriba que la interna."""
    dx = abs(puntos[externo].x - puntos[interno].x) * ancho
    dy = (puntos[interno].y - puntos[externo].y) * alto
    return math.degrees(math.atan2(dy, dx))


def arco_ceja(puntos, pico, extremo1, extremo2, ancho, alto):
    """Qué tan arqueada es la ceja: altura del arco / largo de la ceja."""
    largo = distancia(puntos[extremo1], puntos[extremo2], ancho, alto)
    altura = distancia_a_recta(
        puntos[pico], puntos[extremo1], puntos[extremo2], ancho, alto
    )
    return altura / largo


def asimetria_rostro(puntos, ancho, alto):
    """Asimetría promedio (en %) comparando las distancias de cada par de
    puntos al eje vertical del rostro (frente -> mentón). 0 = simétrico."""
    eje_a, eje_b = puntos[10], puntos[152]
    diferencias = []
    for izq, der in PARES_SIMETRIA:
        di = distancia_a_recta(puntos[izq], eje_a, eje_b, ancho, alto)
        dd = distancia_a_recta(puntos[der], eje_a, eje_b, ancho, alto)
        if di + dd > 0:
            diferencias.append(abs(di - dd) / (di + dd))
    return 100 * sum(diferencias) / len(diferencias)


def giro_cabeza(puntos, ancho, alto):
    """Giro de la cabeza a izquierda o derecha (en %).
    Compara la distancia de la punta de la nariz a cada mejilla: si la
    persona mira de frente, las dos distancias son casi iguales."""
    izq = distancia(puntos[1], puntos[234], ancho, alto)
    der = distancia(puntos[1], puntos[454], ancho, alto)
    return 100 * abs(izq - der) / (izq + der)


def calcular_proporciones(puntos, ancho, alto):
    """Calcula medidas y proporciones del rostro a partir de los puntos."""

    def d(a, b):
        return distancia(puntos[a], puntos[b], ancho, alto)

    alto_rostro = d(10, 152)        # frente -> mentón
    ancho_rostro = d(234, 454)      # de mejilla a mejilla
    ancho_frente = d(54, 284)
    ancho_mandibula = d(172, 397)
    ancho_ojo = d(33, 133)          # un ojo, de esquina a esquina
    separacion_ojos = d(133, 362)   # entre las esquinas internas
    ancho_nariz = d(129, 358)
    ancho_boca = d(61, 291)
    apertura_boca = d(13, 14)       # labio superior -> inferior

    # Inclinación de la cabeza según la línea de los ojos
    izq, der = puntos[33], puntos[263]
    dx = (der.x - izq.x) * ancho
    dy = (der.y - izq.y) * alto
    inclinacion = math.degrees(math.atan2(dy, dx))

    # Sonrisa: cuánto suben las comisuras (esquinas de la boca) respecto al
    # centro de los labios, en proporción al alto del rostro.
    # En el eje Y de la imagen, "más arriba" significa un valor menor.
    centro_labios_y = (puntos[13].y + puntos[14].y) / 2
    comisuras_y = (puntos[61].y + puntos[291].y) / 2
    elevacion_comisuras = (centro_labios_y - comisuras_y) * alto / alto_rostro

    # Tercios del rostro: 10 (arriba de la frente), 9 (entre las cejas),
    # 2 (base de la nariz), 152 (mentón)
    t1, t2, t3 = d(10, 9), d(9, 2), d(2, 152)
    suma_tercios = t1 + t2 + t3

    # Ojos: se promedian ambos; así se compensa si la cabeza está ladeada.
    apertura_ojos = (
        apertura_ojo(puntos, OJO_IZQ, ancho, alto)
        + apertura_ojo(puntos, OJO_DER, ancho, alto)
    ) / 2
    inclinacion_ojos = (
        inclinacion_ojo(puntos, 133, 33, ancho, alto)
        + inclinacion_ojo(puntos, 362, 263, ancho, alto)
    ) / 2

    # Cejas
    arco_cejas = (
        arco_ceja(puntos, 105, 70, 107, ancho, alto)
        + arco_ceja(puntos, 334, 300, 336, ancho, alto)
    ) / 2
    ceja_a_ojo = (d(105, 159) + d(334, 386)) / 2 / alto_rostro

    # Labios y mentón
    grosor_sup = d(0, 13)
    grosor_inf = d(14, 17)
    ancho_menton = d(176, 400)

    return {
        "relacion_alto_ancho": alto_rostro / ancho_rostro,
        "frente_vs_mejillas": ancho_frente / ancho_rostro,
        "mandibula_vs_mejillas": ancho_mandibula / ancho_rostro,
        "menton_vs_mejillas": ancho_menton / ancho_rostro,
        "tercio_superior": t1 / suma_tercios,
        "tercio_medio": t2 / suma_tercios,
        "tercio_inferior": t3 / suma_tercios,
        "asimetria_pct": asimetria_rostro(puntos, ancho, alto),
        "separacion_ojos": separacion_ojos / ancho_ojo,
        "apertura_ojos": apertura_ojos,
        "inclinacion_ojos_grados": inclinacion_ojos,
        "arco_cejas": arco_cejas,
        "ceja_a_ojo": ceja_a_ojo,
        "nariz_vs_rostro": ancho_nariz / ancho_rostro,
        "largo_nariz": d(168, 2) / alto_rostro,
        "boca_vs_rostro": ancho_boca / ancho_rostro,
        "grosor_labios_vs_boca": (grosor_sup + grosor_inf) / ancho_boca,
        "labio_inf_vs_sup": grosor_inf / grosor_sup if grosor_sup else 0.0,
        "apertura_boca": apertura_boca / alto_rostro,
        "elevacion_comisuras": elevacion_comisuras,
        "inclinacion_grados": inclinacion,
        "giro_cabeza_pct": giro_cabeza(puntos, ancho, alto),
    }


# ---------------------------------------------------------------------------
# Verificación de la pose
# ---------------------------------------------------------------------------

def evaluar_pose(m):
    """Revisa si la foto es de frente. Devuelve (nivel, avisos):
    nivel 0 = buena, 1 = aceptable con reservas, 2 = poco fiable."""
    nivel = 0
    avisos = []

    giro = m["giro_cabeza_pct"]
    if giro >= 15:
        nivel = 2
        avisos.append("La cabeza está muy girada hacia un lado.")
    elif giro >= 8:
        nivel = max(nivel, 1)
        avisos.append("La cabeza está un poco girada hacia un lado.")

    inclinacion = abs(m["inclinacion_grados"])
    if inclinacion > 10:
        nivel = 2
        avisos.append("La cabeza está muy inclinada (ladeada).")
    elif inclinacion > 5:
        nivel = max(nivel, 1)
        avisos.append("La cabeza está algo inclinada (ladeada).")

    return nivel, avisos


# ---------------------------------------------------------------------------
# Clasificaciones
# Umbrales iniciales: son aproximaciones y conviene calibrarlos con varias
# fotos (mira los valores de la tabla y ajusta los números a tu gusto).
# ---------------------------------------------------------------------------

def clasificar_forma(m):
    """Decide la forma del rostro con reglas simples."""
    relacion = m["relacion_alto_ancho"]
    frente = m["frente_vs_mejillas"]
    mandibula = m["mandibula_vs_mejillas"]

    if relacion >= 1.45:
        return "alargado"
    if frente - mandibula >= 0.15:
        return "en forma de corazón"
    if relacion < 1.25:
        return "cuadrado" if mandibula >= 0.85 else "redondo"
    return "ovalado"


def clasificar_expresion(m):
    """Decide si el rostro está sonriente o serio según las comisuras."""
    elevacion = m["elevacion_comisuras"]
    if elevacion > 0.025:
        return "sonriente"
    if elevacion > 0.012:
        return "sonrisa leve"
    return "expresión seria"


def clasificar_tercios(m):
    """Describe si los tres tercios del rostro están equilibrados."""
    tercios = {
        "la frente": m["tercio_superior"],
        "la zona de la nariz": m["tercio_medio"],
        "la parte baja del rostro": m["tercio_inferior"],
    }
    mayor = max(tercios, key=tercios.get)
    menor = min(tercios, key=tercios.get)
    if tercios[mayor] - tercios[menor] < 0.06:
        return "equilibrados"
    return f"desiguales (predomina {mayor})"


def clasificar_simetria(m):
    """Describe qué tan simétrico es el rostro en la foto."""
    asimetria = m["asimetria_pct"]
    if asimetria < 3:
        return "muy simétrico"
    if asimetria < 6:
        return "simetría normal"
    return "asimetría notable (revisa que la foto sea de frente)"


def rasgos_ojos(m):
    rasgos = []
    if m["separacion_ojos"] > 1.15:
        rasgos.append("separados")
    elif m["separacion_ojos"] < 0.90:
        rasgos.append("juntos")
    else:
        rasgos.append("con separación promedio")

    apertura = m["apertura_ojos"]
    if apertura < 0.15:
        rasgos.append("cerrados o casi cerrados")
    elif apertura < 0.22:
        rasgos.append("entrecerrados o de apertura pequeña")
    elif apertura > 0.32:
        rasgos.append("muy abiertos")
    else:
        rasgos.append("de apertura media")

    inclinacion = m["inclinacion_ojos_grados"]
    if inclinacion > 7:
        rasgos.append("esquinas externas elevadas")
    elif inclinacion < 1:
        rasgos.append("esquinas externas descendentes")
    else:
        rasgos.append("línea casi nivelada")
    return rasgos


def rasgos_cejas(m):
    rasgos = []
    if m["arco_cejas"] > 0.14:
        rasgos.append("arqueadas")
    elif m["arco_cejas"] < 0.07:
        rasgos.append("rectas")
    else:
        rasgos.append("con arco suave")

    if m["ceja_a_ojo"] > 0.095:
        rasgos.append("altas")
    elif m["ceja_a_ojo"] < 0.065:
        rasgos.append("cercanas a los ojos")
    else:
        rasgos.append("a distancia media de los ojos")
    return rasgos


def rasgos_nariz(m):
    rasgos = []
    if m["nariz_vs_rostro"] > 0.34:
        rasgos.append("ancha")
    elif m["nariz_vs_rostro"] < 0.28:
        rasgos.append("delgada")
    else:
        rasgos.append("de ancho medio")

    if m["largo_nariz"] > 0.27:
        rasgos.append("larga")
    elif m["largo_nariz"] < 0.22:
        rasgos.append("corta")
    else:
        rasgos.append("de largo medio")
    return rasgos


def rasgos_boca(m):
    rasgos = []
    if m["boca_vs_rostro"] > 0.46:
        rasgos.append("boca grande")
    elif m["boca_vs_rostro"] < 0.36:
        rasgos.append("boca pequeña")
    else:
        rasgos.append("boca de tamaño medio")

    rasgos.append("abierta" if m["apertura_boca"] > 0.03 else "cerrada")

    if m["grosor_labios_vs_boca"] > 0.45:
        rasgos.append("labios gruesos")
    elif m["grosor_labios_vs_boca"] < 0.30:
        rasgos.append("labios delgados")
    else:
        rasgos.append("labios de grosor medio")

    if m["labio_inf_vs_sup"] > 1.8:
        rasgos.append("labio inferior notablemente más grueso")
    return rasgos


def rasgos_menton(m):
    if m["menton_vs_mejillas"] > 0.38:
        return ["ancho"]
    if m["menton_vs_mejillas"] < 0.28:
        return ["angosto"]
    return ["de ancho medio"]


def rasgos_cabeza(m):
    rasgos = ["inclinada" if abs(m["inclinacion_grados"]) > 5 else "recta"]
    rasgos.append(clasificar_expresion(m))
    return rasgos


# ---------------------------------------------------------------------------
# Colores de ojos y cabello
# ---------------------------------------------------------------------------

def a_hsv(bgr):
    """Convierte un color BGR a HSV (H: 0-179, S: 0-255, V: 0-255)."""
    pixel = np.array([[bgr]], dtype=np.uint8)
    h, s, v = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    return int(h), int(s), int(v)


def color_iris(imagen, puntos, centro, borde):
    """Color mediano del iris (sin la pupila). Devuelve None si no se puede medir."""
    alto, ancho = imagen.shape[:2]
    cx, cy = puntos[centro].x * ancho, puntos[centro].y * alto
    bx, by = puntos[borde].x * ancho, puntos[borde].y * alto
    radio = math.hypot(bx - cx, by - cy)
    if radio < 3:
        return None

    mascara = np.zeros((alto, ancho), dtype=np.uint8)
    cv2.circle(mascara, (int(cx), int(cy)), int(radio * 0.8), 255, -1)
    cv2.circle(mascara, (int(cx), int(cy)), int(radio * 0.35), 0, -1)  # quita la pupila
    pixeles = imagen[mascara == 255]
    if len(pixeles) == 0:
        return None
    return np.median(pixeles, axis=0)


def clasificar_color_ojos(bgr):
    """Nombra el color de ojos según H, S, V (umbrales aproximados)."""
    h, s, v = a_hsv(bgr)
    if v < 60:
        return "negro o marrón muy oscuro"
    if s < 45:
        return "gris"
    if 85 <= h <= 130:
        return "azul"
    if 35 <= h < 85:
        return "verde"
    if 20 <= h < 35:
        return "avellana o ámbar"
    return "marrón oscuro" if v < 110 else "marrón claro"


def color_cabello(imagen, puntos):
    """Color mediano de una franja justo encima de la frente."""
    alto, ancho = imagen.shape[:2]
    alto_rostro = distancia(puntos[10], puntos[152], ancho, alto)
    cx = int(puntos[10].x * ancho)
    cy = int(puntos[10].y * alto)

    mitad = int(0.25 * alto_rostro)
    x1, x2 = max(cx - mitad, 0), min(cx + mitad, ancho)
    y1 = max(int(cy - 0.25 * alto_rostro), 0)
    y2 = max(int(cy - 0.05 * alto_rostro), 0)
    if (y2 - y1) < 3 or (x2 - x1) < 3:
        return None  # el cabello queda fuera de la foto

    region = imagen[y1:y2, x1:x2].reshape(-1, 3)
    return np.median(region, axis=0)


def clasificar_color_cabello(bgr):
    """Nombra el color de cabello según H, S, V (umbrales aproximados)."""
    h, s, v = a_hsv(bgr)
    if v < 55:
        return "negro"
    if s < 40 and v > 170:
        return "muy claro o canoso"
    if v < 110:
        return "castaño oscuro"
    if (h < 10 or h > 165) and s > 110:
        return "rojizo"
    if 15 <= h <= 35 and v >= 170 and s >= 60:
        return "rubio"
    return "castaño"


def analizar_colores(imagen, puntos, medidas):
    """Devuelve las líneas del reporte con el color estimado de ojos y cabello."""
    lineas = []

    if medidas["apertura_ojos"] < 0.15:
        lineas.append("  Ojos:    no se pudo medir (ojos cerrados o casi cerrados)")
    else:
        ojo1 = color_iris(imagen, puntos, 468, 469)
        ojo2 = color_iris(imagen, puntos, 473, 474)
        ojos = [c for c in (ojo1, ojo2) if c is not None]
        if ojos:
            promedio = np.mean(ojos, axis=0)
            lineas.append(f"  Ojos:    {clasificar_color_ojos(promedio)}")
        else:
            lineas.append("  Ojos:    no se pudo medir (rostro muy pequeño)")

    cabello = color_cabello(imagen, puntos)
    if cabello is not None:
        lineas.append(f"  Cabello: {clasificar_color_cabello(cabello)}")
    else:
        lineas.append("  Cabello: no se pudo medir (la frente está en el borde de la foto)")

    lineas.append("  (Son estimaciones: dependen mucho de la luz y la calidad de la foto.)")
    return lineas


# ---------------------------------------------------------------------------
# Construcción del reporte
# ---------------------------------------------------------------------------

def seccion(titulo):
    """Encabezado de sección del reporte."""
    return ["", titulo, "-" * ANCHO_REPORTE]


def tabla_medidas(m):
    """Tabla de medidas agrupadas, con una explicación de cada valor."""
    lineas = []
    for titulo, claves in GRUPOS:
        lineas.append(f"  {titulo}")
        for clave in claves:
            nombre, formato, explicacion = ETIQUETAS[clave]
            valor = formato.format(m[clave])
            lineas.append(f"    {nombre:<30}{valor:>8}   {explicacion}")
        lineas.append("")
    return lineas


def construir_reporte(nombre_foto, m, lineas_colores):
    """Arma todo el reporte como una lista de líneas de texto."""
    lineas = ["=" * ANCHO_REPORTE,
              f"  ANÁLISIS FACIAL - {nombre_foto}",
              "=" * ANCHO_REPORTE]

    # 1. Calidad de la foto
    nivel, avisos = evaluar_pose(m)
    lineas += seccion("1. CALIDAD DE LA FOTO")
    if nivel == 0:
        lineas.append("  [OK] Foto de frente: las medidas son confiables.")
    else:
        estado = "con reservas" if nivel == 1 else "poco fiables"
        lineas.append(f"  [!] Las medidas pueden ser {estado}.")
        for aviso in avisos:
            lineas.append(f"      - {aviso}")
        lineas.append("      Recomendación: usa una foto de frente, con la cabeza recta.")

    # 2. Resumen
    lineas += seccion("2. RESUMEN")
    lineas.append(f"  Forma del rostro : {clasificar_forma(m)}")
    lineas.append(f"  Expresión        : {clasificar_expresion(m)}")
    lineas.append(f"  Simetría         : {clasificar_simetria(m)}")
    lineas.append(
        f"  Tercios          : {clasificar_tercios(m)} "
        f"(frente {m['tercio_superior']:.0%}, nariz {m['tercio_medio']:.0%}, "
        f"parte baja {m['tercio_inferior']:.0%})"
    )

    # 3. Rasgos
    lineas += seccion("3. RASGOS DEL ROSTRO")
    grupos_rasgos = [
        ("Ojos", rasgos_ojos(m)),
        ("Cejas", rasgos_cejas(m)),
        ("Nariz", rasgos_nariz(m)),
        ("Boca y labios", rasgos_boca(m)),
        ("Mentón", rasgos_menton(m)),
        ("Cabeza", rasgos_cabeza(m)),
    ]
    for nombre, rasgos in grupos_rasgos:
        lineas.append(f"  {nombre:<14}: " + ", ".join(rasgos))

    # 4. Colores
    lineas += seccion("4. COLORES (ESTIMADOS)")
    lineas += lineas_colores

    return lineas


def guardar_reporte(lineas, nombre_foto):
    """Guarda el reporte en un archivo de texto junto al programa."""
    base = os.path.splitext(nombre_foto)[0]
    ruta = os.path.join(CARPETA, f"reporte_{base}.txt")
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas) + "\n")
    print(f"\nReporte guardado en: {ruta}")


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def dibujar_puntos(imagen, puntos):
    """Dibuja los puntos faciales sobre la imagen."""
    alto, ancho = imagen.shape[:2]
    for p in puntos:
        x = int(p.x * ancho)
        y = int(p.y * alto)
        cv2.circle(imagen, (x, y), 1, (0, 255, 0), -1)


def mostrar_imagen(imagen):
    """Muestra la imagen en una ventana, reducida si es muy grande."""
    alto, ancho = imagen.shape[:2]
    if alto > 800:
        escala = 800 / alto
        imagen = cv2.resize(imagen, (int(ancho * escala), 800))
    cv2.imshow("Rostro con puntos faciales", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calcular_similitud(m1, m2):
    """Compara proporciones faciales y devuelve un porcentaje aproximado."""
    tolerancias = {
        "relacion_alto_ancho": 0.30,
        "frente_vs_mejillas": 0.15,
        "mandibula_vs_mejillas": 0.18,
        "menton_vs_mejillas": 0.15,
        "tercio_superior": 0.12,
        "tercio_medio": 0.12,
        "tercio_inferior": 0.12,
        "separacion_ojos": 0.25,
        "nariz_vs_rostro": 0.15,
        "largo_nariz": 0.15,
        "boca_vs_rostro": 0.18,
    }
    puntuaciones = []
    for clave, tolerancia in tolerancias.items():
        diferencia = abs(m1[clave] - m2[clave])
        puntuaciones.append(max(0.0, 1.0 - diferencia / tolerancia))
    return 100 * sum(puntuaciones) / len(puntuaciones)


def analizar_foto(ruta):
    """Detecta un rostro y devuelve imagen, puntos faciales y medidas."""
    imagen = cargar_imagen(ruta)
    if imagen is None:
        return None
    resultado = detectar_rostro(ruta)
    cantidad = len(resultado.face_landmarks)
    if cantidad != 1:
        if cantidad == 0:
            print(f"No se detectó ningún rostro en: {ruta}")
        else:
            print(f"Se detectaron varios rostros en: {ruta}. Usa una foto con una sola persona.")
        return None
    puntos = resultado.face_landmarks[0]
    alto, ancho = imagen.shape[:2]
    medidas = calcular_proporciones(puntos, ancho, alto)
    return imagen, puntos, medidas

def main():
    if not os.path.exists(RUTA_MODELO):
        print("Falta el archivo del modelo: face_landmarker.task")
        print(f"Guárdalo en esta carpeta: {CARPETA}")
        return

    print("Compara las proporciones faciales de dos fotos.")
    ruta1 = os.path.join(CARPETA, input("Nombre de la primera foto: ").strip())
    ruta2 = os.path.join(CARPETA, input("Nombre de la segunda foto: ").strip())

    analisis1 = analizar_foto(ruta1)
    if analisis1 is None:
        return
    analisis2 = analizar_foto(ruta2)
    if analisis2 is None:
        return

    similitud = calcular_similitud(analisis1[2], analisis2[2])
    print(f"\nSimilitud de proporciones faciales: {similitud:.1f}%")
    print("Es una comparación aproximada de rasgos geométricos, no una verificación de identidad.")

    for ruta, (imagen, puntos, medidas) in ((ruta1, analisis1), (ruta2, analisis2)):
        colores = analizar_colores(imagen, puntos, medidas)
        reporte = construir_reporte(os.path.basename(ruta), medidas, colores)
        print("\n" + "\n".join(reporte))
        respuesta = input(f"¿Guardar el reporte de {os.path.basename(ruta)}? (s/n): ").strip().lower()
        if respuesta == "s":
            guardar_reporte(reporte, os.path.basename(ruta))
        dibujar_puntos(imagen, puntos)
        mostrar_imagen(imagen)


if __name__ == "__main__":
    main()