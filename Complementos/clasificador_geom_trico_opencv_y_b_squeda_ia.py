"""
SISTEMA DE RECONOCIMIENTO DE FIGURAS GEOMÉTRICAS MEDIANTE BÚSQUEDA EN ESPACIO DE ESTADOS Y OPENCV
=================================================================================================
Entregable en Python que implementa:
1. Lectura de imagen o generación sintética de las 6 figuras (Círculo, Triángulo, Cuadrado, Trapecio, Estrella, Null).
2. Extracción de rasgos geométricos con OpenCV (Vértices, Circularidad, Relación de Aspecto, Paralelismo, Concavidad).
3. Motores de Búsqueda:
   - Búsqueda a Ciegas (BFS en anchura sin información adicional).
   - Búsqueda Heurística A* (f(n) = g(n) + h(n) basada en rasgos geométricos).
   - Algoritmo Genético (Cromosoma de 5 genes, función Fitness, Cruce de 1 punto y Mutación).
4. Visualización gráfica mostrando: Imagen Original, Rasgos Detectados, Figura Identificada y Confianza.
"""

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# =====================================================================
# 1. ESTRUCTURA DE DATOS DEL ESTADO Y RASGOS GEOMÉTRICOS
# =====================================================================
@dataclass
class RasgosGeometricos:
    """Representa los rasgos extraídos de un contorno mediante OpenCV."""
    num_vertices: int = 0
    circularidad: float = 0.0
    relacion_aspecto: float = 1.0
    es_convexo: bool = True
    defectos_convexidad: int = 0
    pares_paralelos: int = 0
    area: float = 0.0
    perimetro: float = 0.0


@dataclass
class NodoGrafo:
    """Nodo del Grafo Representativo del Sistema de Reconocimiento."""
    id_nodo: str
    nombre: str
    es_terminal: bool = False
    costo_g: int = 0
    camino: List[str] = field(default_factory=list)


# =====================================================================
# 2. MÓDULO OPENCV: LEER Y EXTRAER RASGOS GEOMÉTRICOS
# =====================================================================
def generar_lienzo_prueba() -> np.ndarray:
    """
    Genera una imagen sintética de 900x620 con las 6 figuras del grafo:
    Círculo, Triángulo, Cuadrado, Trapecio, Estrella y Objeto Irregular (Null).
    """
    lienzo = np.full((620, 900, 3), 24, dtype=np.uint8)

    # 1. Círculo
    cv2.circle(lienzo, (150, 155), 68, (129, 185, 16), -1)

    # 2. Triángulo
    pts_tri = np.array([[450, 80], [375, 225], [525, 225]], np.int32)
    cv2.fillPoly(lienzo, [pts_tri], (248, 189, 56))

    # 3. Cuadrado
    cv2.rectangle(lienzo, (675, 85), (815, 225), (241, 102, 99), -1)

    # 4. Trapecio
    pts_trap = np.array([[100, 370], [200, 370], [245, 495], [55, 495]], np.int32)
    cv2.fillPoly(lienzo, [pts_trap], (11, 158, 245))

    # 5. Estrella (10 vértices: 5 puntas externas + 5 valles internos)
    pts_est = []
    cx, cy, r_ext, r_int = 450, 435, 78, 33
    for i in range(10):
        r = r_ext if i % 2 == 0 else r_int
        ang = i * math.pi / 5 - math.pi / 2
        pts_est.append([int(cx + r * math.cos(ang)), int(cy + r * math.sin(ang))])
    cv2.fillPoly(lienzo, [np.array(pts_est, np.int32)], (247, 85, 168))

    # 6. Objeto Irregular / Ruido -> Nodo Terminal NULL
    pts_null = np.array([
        [680, 380], [760, 360], [825, 415],
        [785, 505], [720, 480], [665, 445]
    ], np.int32)
    cv2.fillPoly(lienzo, [pts_null], (110, 110, 235))

    return lienzo


def contar_pares_paralelos(aprox: np.ndarray, tolerancia_grados: float = 10.0) -> int:
    """Cuenta cuántos pares de lados opuestos son paralelos en un polígono de 4 vértices."""
    pts = aprox.reshape(-1, 2)
    if len(pts) != 4:
        return 0

    angulos = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        ang = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180.0
        angulos.append(ang)

    pares = 0
    # Comparar lado 0 con lado 2, y lado 1 con lado 3
    for i in range(2):
        diff = abs(angulos[i] - angulos[i + 2])
        diff = min(diff, 180.0 - diff)
        if diff <= tolerancia_grados:
            pares += 1
    return pares


def extraer_rasgos_contorno(contorno: np.ndarray) -> Tuple[RasgosGeometricos, np.ndarray]:
    """
    Aplica operadores de OpenCV sobre un contorno para construir el estado intermedio s.
    """
    area = float(cv2.contourArea(contorno))
    perimetro = float(cv2.arcLength(contorno, True))
    circularidad = (4.0 * math.pi * area) / (perimetro ** 2) if perimetro > 0 else 0.0

    # Aproximación poligonal de Douglas-Peucker
    epsilon = 0.022 * perimetro
    aprox = cv2.approxPolyDP(contorno, epsilon, True)
    num_vertices = len(aprox)

    # Bounding Box y Relación de Aspecto (AR)
    _, _, w, h = cv2.boundingRect(aprox)
    relacion_aspecto = float(w) / float(h) if h > 0 else 1.0

    # Evaluación de Convexidad y Defectos de Convexidad (para Estrella vs Círculo)
    es_convexo = bool(cv2.isContourConvex(aprox))
    defectos_profundos = 0
    hull_indices = cv2.convexHull(contorno, returnPoints=False)
    if hull_indices is not None and len(hull_indices) > 3:
        try:
            defects = cv2.convexityDefects(contorno, hull_indices)
            if defects is not None:
                for i in range(defects.shape[0]):
                    _, _, _, depth = defects[i, 0]
                    if (depth / 256.0) > 12.0:
                        defectos_profundos += 1
        except cv2.error:
            pass

    if defectos_profundos >= 3:
        es_convexo = False

    pares_paralelos = contar_pares_paralelos(aprox) if num_vertices == 4 else 0

    rasgos = RasgosGeometricos(
        num_vertices=num_vertices,
        circularidad=round(circularidad, 3),
        relacion_aspecto=round(relacion_aspecto, 3),
        es_convexo=es_convexo,
        defectos_convexidad=defectos_profundos,
        pares_paralelos=pares_paralelos,
        area=area,
        perimetro=perimetro
    )
    return rasgos, aprox


# =====================================================================
# 3. ALGORITMOS DE BÚSQUEDA (CIEGA, HEURÍSTICA Y GENÉTICO)
# =====================================================================
def busqueda_a_ciegas_bfs(r: RasgosGeometricos) -> Tuple[str, float, int, List[str]]:
    """
    1. BÚSQUEDA A CIEGAS (BFS - Anchura):
    Explora el grafo nivel por nivel sin usar heurística, acumulando todas las comparaciones.
    """
    cola = deque([NodoGrafo("S0", "Imagen_Sin_Procesar", False, 0, ["S0"])])
    comparaciones = 0

    while cola:
        actual = cola.popleft()

        if actual.id_nodo == "S0":
            comparaciones += 1
            cola.append(NodoGrafo("S1", "Contorno_Detectado", False, comparaciones, actual.camino + ["S1"]))

        elif actual.id_nodo == "S1":
            # En búsqueda ciega se encolan todas las ramas de nivel 2
            for rama in ["N_V3", "N_V4", "N_V8+", "N_IRREGULAR"]:
                comparaciones += 1
                cola.append(NodoGrafo(rama, rama, False, comparaciones, actual.camino + [rama]))

        elif actual.id_nodo == "N_V3":
            comparaciones += 1
            if r.num_vertices == 3 and r.es_convexo:
                return "Triangulo", 0.98, comparaciones, actual.camino + ["Triangulo"]

        elif actual.id_nodo == "N_V4":
            comparaciones += 1
            if r.num_vertices == 4:
                cola.append(NodoGrafo("N_PARALELOS", "Evaluar_Paralelismo", False, comparaciones, actual.camino + ["N_PARALELOS"]))

        elif actual.id_nodo == "N_PARALELOS":
            comparaciones += 1
            if r.pares_paralelos == 2 and 0.85 <= r.relacion_aspecto <= 1.15:
                return "Cuadrado", 0.99, comparaciones, actual.camino + ["Cuadrado"]
            comparaciones += 1
            if r.pares_paralelos >= 1 and not (0.85 <= r.relacion_aspecto <= 1.15):
                return "Trapecio", 0.95, comparaciones, actual.camino + ["Trapecio"]

        elif actual.id_nodo == "N_V8+":
            comparaciones += 1
            if r.num_vertices >= 8:
                cola.append(NodoGrafo("N_CONCAVIDAD", "Evaluar_Concavidad", False, comparaciones, actual.camino + ["N_CONCAVIDAD"]))

        elif actual.id_nodo == "N_CONCAVIDAD":
            comparaciones += 1
            if r.es_convexo and r.circularidad >= 0.80:
                return "Circulo", 0.97, comparaciones, actual.camino + ["Circulo"]
            comparaciones += 1
            if not r.es_convexo and r.defectos_convexidad >= 4:
                return "Estrella", 0.96, comparaciones, actual.camino + ["Estrella"]

    return "Null", 0.20, comparaciones + 1, ["S0", "S1", "N_IRREGULAR", "Null"]


def calcular_heuristica_rasgos(r: RasgosGeometricos, rama: str) -> float:
    """
    Función Heurística h(n) basada en la discrepancia entre los rasgos observados
    y el prototipo geométrico esperado en cada rama del grafo.
    """
    if rama == "N_V3":
        return abs(r.num_vertices - 3) * 0.4 + (0.0 if r.es_convexo else 0.5)
    if rama == "N_V4":
        return abs(r.num_vertices - 4) * 0.4
    if rama == "N_V8+":
        return 0.05 if r.num_vertices >= 8 else abs(8 - r.num_vertices) * 0.35
    return 0.65


def busqueda_heuristica_a_estrella(r: RasgosGeometricos) -> Tuple[str, float, int, List[str]]:
    """
    2. BÚSQUEDA HEURÍSTICA A* (f(n) = g(n) + h(n)):
    Prioriza la rama con menor valor f(n), ahorrando comparaciones innecesarias.
    """
    camino = ["S0", "S1"]
    comparaciones = 1  # Paso S0 -> S1

    # Evaluar heurística h(n) para las ramas del Nivel 2 y ordenar por f(n) = g(n) + h(n)
    ramas_nivel_2 = ["N_V3", "N_V4", "N_V8+", "N_IRREGULAR"]
    ramas_ordenadas = sorted(ramas_nivel_2, key=lambda rm: 1 + calcular_heuristica_rasgos(r, rm))

    mejor_rama = ramas_ordenadas[0]
    comparaciones += 1
    camino.append(mejor_rama)

    if mejor_rama == "N_V3" and r.num_vertices == 3:
        comparaciones += 1
        if r.es_convexo:
            camino.append("Triangulo")
            return "Triangulo", 0.98, comparaciones, camino

    elif mejor_rama == "N_V4" and r.num_vertices == 4:
        comparaciones += 1
        camino.append("N_PARALELOS")
        comparaciones += 1
        if r.pares_paralelos == 2 and 0.85 <= r.relacion_aspecto <= 1.15:
            camino.append("Cuadrado")
            return "Cuadrado", 0.99, comparaciones, camino
        if r.pares_paralelos >= 1:
            camino.append("Trapecio")
            return "Trapecio", 0.95, comparaciones, camino

    elif mejor_rama == "N_V8+" and r.num_vertices >= 8:
        comparaciones += 1
        camino.append("N_CONCAVIDAD")
        comparaciones += 1
        if r.es_convexo and r.circularidad >= 0.80:
            camino.append("Circulo")
            return "Circulo", round(min(0.99, r.circularidad + 0.06), 2), comparaciones, camino
        if not r.es_convexo and r.defectos_convexidad >= 4:
            camino.append("Estrella")
            return "Estrella", 0.96, comparaciones, camino

    camino.append("Null")
    return "Null", 0.18, comparaciones + 1, camino


def clasificador_algoritmo_genetico(r: RasgosGeometricos, generaciones: int = 25) -> Tuple[str, float]:
    """
    3. ALGORITMO GENÉTICO:
    - Cromosoma: [num_vertices, circularidad, relacion_aspecto, es_convexo, pares_paralelos]
    - Fitness: Similitud inversa a la distancia euclidiana ponderada respecto al objeto observado.
    - Cruce: Un punto entre los mejores individuos.
    - Mutación: Perturbación estocástica de genes.
    """
    prototipos: Dict[str, List[float]] = {
        "Circulo":   [12.0, 0.90, 1.00, 1.0, 0.0],
        "Triangulo": [3.0,  0.60, 1.00, 1.0, 0.0],
        "Cuadrado":  [4.0,  0.78, 1.00, 1.0, 2.0],
        "Trapecio":  [4.0,  0.67, 1.45, 1.0, 1.0],
        "Estrella":  [10.0, 0.29, 1.00, 0.0, 0.0],
        "Null":      [6.0,  0.50, 1.30, 0.0, 0.0],
    }

    vector_obs = [
        float(min(r.num_vertices, 12)),
        r.circularidad,
        r.relacion_aspecto,
        1.0 if r.es_convexo else 0.0,
        float(r.pares_paralelos)
    ]

    def fitness(cromosoma: List[float]) -> float:
        pesos = [0.25, 1.2, 0.8, 1.0, 0.6]
        dist_sq = sum(w * ((o - c) ** 2) for w, o, c in zip(pesos, vector_obs, cromosoma))
        return 1.0 / (1.0 + math.sqrt(dist_sq))

    # Inicializar población combinando prototipos y variaciones
    poblacion = [list(vals) for vals in prototipos.values()]
    for _ in range(6):
        poblacion.append([
            float(random.randint(3, 12)),
            round(random.uniform(0.2, 0.95), 2),
            round(random.uniform(0.8, 1.6), 2),
            float(random.choice([0, 1])),
            float(random.choice([0, 1, 2]))
        ])

    for _ in range(generaciones):
        poblacion.sort(key=fitness, reverse=True)
        padres = poblacion[:4]
        nueva_pob = padres[:]

        while len(nueva_pob) < len(poblacion):
            p1, p2 = random.sample(padres, 2)
            # Cruce de 1 punto
            punto = random.randint(1, 4)
            hijo = p1[:punto] + p2[punto:]

            # Mutación suave
            if random.random() < 0.25:
                idx_gen = random.randint(0, 4)
                hijo[idx_gen] += random.gauss(0, 0.08)
            nueva_pob.append(hijo)

        poblacion = nueva_pob

    mejor_individuo = max(poblacion, key=fitness)
    mejor_clase = min(
        prototipos.keys(),
        key=lambda k: sum((a - b) ** 2 for a, b in zip(mejor_individuo, prototipos[k]))
    )
    return mejor_clase, round(fitness(mejor_individuo), 3)


# =====================================================================
# 4. PIPELINE PRINCIPAL: PROCESAR, CLASIFICAR Y MOSTRAR RESULTADOS
# =====================================================================
def ejecutar_sistema_reconocimiento(ruta_imagen: Optional[str] = None) -> None:
    """
    Ejecuta el flujo completo:
    1. Lee la imagen (o genera el conjunto de prueba si no se especifica archivo).
    2. Detecta contornos y extrae rasgos geométricos.
    3. Clasifica cada figura mediante Búsqueda Ciega, Heurística A* y Algoritmo Genético.
    4. Muestra por consola y en ventana gráfica la imagen original, rasgos, clase y confianza.
    """
    if ruta_imagen:
        img_original = cv2.imread(ruta_imagen)
        if img_original is None:
            print(f"[Aviso] No se encontró '{ruta_imagen}'. Generando imagen sintética con las 6 figuras...")
            img_original = generar_lienzo_prueba()
    else:
        img_original = generar_lienzo_prueba()

    img_resultado = img_original.copy()
    gris = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    _, binaria = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("=" * 95)
    print("RESULTADOS DEL SISTEMA DE RECONOCIMIENTO DE FIGURAS (OPENCV + BÚSQUEDA EN ESPACIO DE ESTADOS)")
    print("=" * 95)

    for idx, cnt in enumerate(contornos, start=1):
        if cv2.contourArea(cnt) < 400:
            continue

        rasgos, aprox = extraer_rasgos_contorno(cnt)

        # Ejecutar los 3 algoritmos
        clase_bfs, conf_bfs, comp_bfs, _ = busqueda_a_ciegas_bfs(rasgos)
        clase_astar, conf_astar, comp_astar, ruta_astar = busqueda_heuristica_a_estrella(rasgos)
        clase_ag, fit_ag = clasificador_algoritmo_genetico(rasgos)

        print(f"\n[Objeto #{idx}] Rasgos Extraídos -> V={rasgos.num_vertices}, "
              f"Circ={rasgos.circularidad:.2f}, AR={rasgos.relacion_aspecto:.2f}, "
              f"Convexo={rasgos.es_convexo}, Defectos={rasgos.defectos_convexidad}, Paralelos={rasgos.pares_paralelos}")
        print(f"  • Búsqueda Ciega (BFS)   : {clase_bfs:<10} | Confianza: {conf_bfs*100:5.1f}% | Costo g(n): {comp_bfs} comparaciones")
        print(f"  • Búsqueda Heurística A* : {clase_astar:<10} | Confianza: {conf_astar*100:5.1f}% | Costo g(n): {comp_astar} comparaciones | Ruta: {' -> '.join(ruta_astar)}")
        print(f"  • Algoritmo Genético     : {clase_ag:<10} | Fitness  : {fit_ag*100:5.1f}%")

        # Dibujar contorno, vértices y etiquetas sobre la imagen de salida
        color_borde = (16, 185, 129) if clase_astar != "Null" else (68, 68, 239)
        cv2.drawContours(img_resultado, [cnt], -1, color_borde, 2)

        for pt in aprox:
            px, py = pt[0]
            cv2.circle(img_resultado, (int(px), int(py)), 4, (0, 0, 255), -1)

        x, y, w, h = cv2.boundingRect(cnt)
        etiqueta_1 = f"{clase_astar.upper()} ({conf_astar*100:.0f}%)"
        etiqueta_2 = f"V:{rasgos.num_vertices} C:{rasgos.circularidad:.2f} AR:{rasgos.relacion_aspecto:.2f}"

        cv2.putText(img_resultado, etiqueta_1, (x, max(22, y - 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        cv2.putText(img_resultado, etiqueta_2, (x, max(38, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (56, 189, 248), 1)

    # Guardar imagen procesada para adjuntar en reportes y mostrar en ventana si hay entorno gráfico
    cv2.imwrite("resultado_reconocimiento_opencv.png", img_resultado)
    print("\n[OK] Imagen con resultados guardada en 'resultado_reconocimiento_opencv.png'.")


if __name__ == "__main__":
    ejecutar_sistema_reconocimiento()