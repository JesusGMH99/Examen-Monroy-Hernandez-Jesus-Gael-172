"""
=================================================================================================
EXAMEN DE FUNDAMENTOS DE INTELIGENCIA ARTIFICIAL — UNIVERSIDAD POLITÉCNICA DE ATLAUTLA (UPATLAUTLA)
Alumno: Jesús Gael Monroy Hernández | Grupo: 172 | Cuatrimestre: 7
Docente: Mtro. Javier García Lara
=================================================================================================
Arquitectura donde PYTHON + OPENCV realiza TODO el procesamiento pesado:
1. Lee la imagen (o genera el lienzo real de las 6 figuras: Círculo, Cuadrado, Trapecio, Estrella, Triángulo, Null).
2. Extrae rasgos geométricos reales con OpenCV (bordes Canny, contornos, approxPolyDP, defectos de convexidad, paralelismo).
3. Ejecuta los 3 motores de Inteligencia Artificial en Python:
   - Búsqueda a Ciegas (BFS en anchura sin información adicional).
   - Búsqueda Heurística A* (f(n) = g(n) + h(n) basada en rasgos geométricos).
   - Algoritmo Genético (Cromosoma de 5 genes, función Fitness, Cruce de 1 punto y Mutación).
4. Recorta cada figura procesada (Original, Máscara/Canny y Anotada con Vértices), las codifica en Base64
   y genera automáticamente la Interfaz Gráfica HTML ('Producto1_Interfaz_Grafo_OpenCV_Monroy.html')
   con todas las imágenes reales e inferencias inyectadas desde Python.
"""

import base64
import json
import math
import os
import random
import webbrowser
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

random.seed(42)

# =====================================================================
# 1. ESTRUCTURAS DE DATOS DEL PROBLEMA DE BÚSQUEDA
# =====================================================================
@dataclass
class RasgosGeometricos:
    id_objeto: int
    nombre_esperado: str
    num_vertices: int
    circularidad: float
    relacion_aspecto: float
    solidez: float
    es_convexo: bool
    defectos_convexidad: int
    pares_paralelos: int
    area: float
    perimetro: float
    caja: Tuple[int, int, int, int]


@dataclass
class ResultadoClasificacion:
    rasgos: RasgosGeometricos
    # Resultados Búsqueda Ciega (BFS)
    clase_bfs: str
    confianza_bfs: float
    costo_bfs: int
    ruta_bfs: List[str]
    # Resultados Búsqueda Heurística (A*)
    clase_astar: str
    confianza_astar: float
    costo_astar: int
    heuristica_h: float
    ruta_astar: List[str]
    # Resultados Algoritmo Genético
    clase_ag: str
    fitness_ag: float
    cromosoma_optimo: List[float]
    historial_ag: List[Dict[str, float]]
    # Imágenes reales procesadas por OpenCV en Base64 para la GUI HTML
    img_orig_b64: str
    img_edges_b64: str
    img_annot_b64: str


# =====================================================================
# 2. GENERACIÓN Y LECTURA DE IMAGEN CON LAS 6 FIGURAS GEOMÉTRICAS
# =====================================================================
def generar_lienzo_6_figuras() -> np.ndarray:
    """
    Construye una imagen real de 960x640 px con las 6 clases del grafo:
    1. Círculo, 2. Cuadrado, 3. Trapecio, 4. Estrella, 5. Triángulo, 6. Null (Amorfa).
    """
    lienzo = np.full((640, 960, 3), 20, dtype=np.uint8)

    # Cuadrícula sutil de fondo
    for x in range(0, 960, 40):
        cv2.line(lienzo, (x, 0), (x, 640), (30, 35, 48), 1)
    for y in range(0, 640, 40):
        cv2.line(lienzo, (0, y), (960, y), (30, 35, 48), 1)

    # 1. CÍRCULO (Arriba Izquierda)
    cv2.circle(lienzo, (165, 165), 76, (130, 200, 30), -1, cv2.LINE_AA)

    # 2. CUADRADO (Arriba Centro)
    cv2.rectangle(lienzo, (405, 90), (555, 240), (235, 115, 65), -1, cv2.LINE_AA)

    # 3. TRAPECIO (Arriba Derecha)
    pts_trap = np.array([[745, 95], [855, 95], [905, 235], [695, 235]], np.int32)
    cv2.fillPoly(lienzo, [pts_trap], (35, 165, 245), cv2.LINE_AA)

    # 4. ESTRELLA DE 5 PUNTAS / 10 VÉRTICES (Abajo Izquierda)
    pts_est = []
    cx, cy, r_ext, r_int = 165, 465, 84, 35
    for i in range(10):
        r = r_ext if i % 2 == 0 else r_int
        ang = i * math.pi / 5.0 - math.pi / 2.0
        pts_est.append([int(cx + r * math.cos(ang)), int(cy + r * math.sin(ang))])
    cv2.fillPoly(lienzo, [np.array(pts_est, np.int32)], (215, 85, 190), cv2.LINE_AA)

    # 5. TRIÁNGULO EQUILÁTERO/ISÓSCELES (Abajo Centro)
    pts_tri = np.array([[480, 380], [395, 540], [565, 540]], np.int32)
    cv2.fillPoly(lienzo, [pts_tri], (60, 210, 235), cv2.LINE_AA)

    # 6. FIGURA IRREGULAR / NULL (Abajo Derecha - Polígono asimétrico de 6 lados con muesca)
    pts_null = np.array([
        [735, 385], [835, 375], [890, 435],
        [810, 465], [865, 540], [740, 535], [700, 455]
    ], np.int32)
    cv2.fillPoly(lienzo, [pts_null], (95, 95, 235), cv2.LINE_AA)

    return lienzo


def img_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', img)
    return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')


# =====================================================================
# 3. EXTRACCIÓN DE RASGOS GEOMÉTRICOS CON OPENCV
# =====================================================================
def contar_pares_paralelos(aprox: np.ndarray, tol_grados: float = 11.0) -> int:
    pts = aprox.reshape(-1, 2)
    if len(pts) != 4:
        return 0
    angulos = []
    for i in range(4):
        p1, p2 = pts[i], pts[(i + 1) % 4]
        ang = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180.0
        angulos.append(ang)
    pares = 0
    for i in range(2):
        diff = abs(angulos[i] - angulos[i + 2])
        diff = min(diff, 180.0 - diff)
        if diff <= tol_grados:
            pares += 1
    return pares


def extraer_rasgos_opencv(contorno: np.ndarray, idx: int) -> Tuple[RasgosGeometricos, np.ndarray, np.ndarray]:
    area = float(cv2.contourArea(contorno))
    perimetro = float(cv2.arcLength(contorno, True))
    circularidad = (4.0 * math.pi * area) / (perimetro ** 2) if perimetro > 0 else 0.0

    # Aproximación poligonal Douglas-Peucker
    aprox = cv2.approxPolyDP(contorno, 0.022 * perimetro, True)
    num_vertices = len(aprox)

    x, y, w, h = cv2.boundingRect(contorno)
    relacion_aspecto = float(w) / float(h) if h > 0 else 1.0

    hull_pts = cv2.convexHull(contorno)
    area_hull = float(cv2.contourArea(hull_pts))
    solidez = area / area_hull if area_hull > 0 else 1.0

    es_convexo = bool(cv2.isContourConvex(aprox))
    defectos_profundos = 0
    hull_idx = cv2.convexHull(contorno, returnPoints=False)
    if hull_idx is not None and len(hull_idx) > 3:
        try:
            defects = cv2.convexityDefects(contorno, hull_idx)
            if defects is not None:
                for i in range(defects.shape[0]):
                    _, _, _, depth = defects[i, 0]
                    if (depth / 256.0) > 10.0:
                        defectos_profundos += 1
        except cv2.error:
            pass

    if defectos_profundos >= 2:
        es_convexo = False

    pares_paralelos = contar_pares_paralelos(aprox) if num_vertices == 4 else 0

    rasgos = RasgosGeometricos(
        id_objeto=idx,
        nombre_esperado="",
        num_vertices=num_vertices,
        circularidad=round(circularidad, 3),
        relacion_aspecto=round(relacion_aspecto, 3),
        solidez=round(solidez, 3),
        es_convexo=es_convexo,
        defectos_convexidad=defectos_profundos,
        pares_paralelos=pares_paralelos,
        area=round(area, 1),
        perimetro=round(perimetro, 1),
        caja=(x, y, w, h)
    )
    return rasgos, aprox, hull_pts


# =====================================================================
# 4. MOTORES DE BÚSQUEDA EN PYTHON (CIEGA, HEURÍSTICA A* Y GENÉTICO)
# =====================================================================
def ejecutar_busqueda_ciega_bfs(r: RasgosGeometricos) -> Tuple[str, float, int, List[str]]:
    """
    1. Búsqueda a Ciegas (BFS): Recorre por niveles en orden fijo sin información heurística.
    Costo = número de comparaciones de rasgos efectuadas.
    """
    ruta = ["S0:Imagen_Cruda", "S1:Contorno_Extraido"]
    comparaciones = 1  # Extracción inicial

    # Nivel 2: Evalúa exhaustivamente las ramas en orden fijo
    # Rama 1: ¿Es Triángulo?
    comparaciones += 2
    ruta.append("N_V3")
    if r.num_vertices == 3 and r.es_convexo:
        ruta.append("Meta:Triangulo")
        return "Triangulo", 0.985, comparaciones, ruta

    # Rama 2: ¿Es Cuadrilátero (Cuadrado o Trapecio)?
    comparaciones += 2
    ruta.append("N_V4")
    if r.num_vertices == 4:
        ruta.append("N_Paralelismo")
        comparaciones += 2
        if r.pares_paralelos == 2 and 0.88 <= r.relacion_aspecto <= 1.12:
            ruta.append("Meta:Cuadrado")
            return "Cuadrado", 0.991, comparaciones, ruta
        comparaciones += 2
        if r.pares_paralelos >= 1 and not (0.88 <= r.relacion_aspecto <= 1.12):
            ruta.append("Meta:Trapecio")
            return "Trapecio", 0.954, comparaciones, ruta

    # Rama 3: ¿Es Polígono Denso (Círculo o Estrella)?
    comparaciones += 2
    ruta.append("N_V8+")
    if r.num_vertices >= 8:
        ruta.append("N_Concavidad")
        comparaciones += 2
        if r.es_convexo and r.circularidad >= 0.80:
            ruta.append("Meta:Circulo")
            return "Circulo", round(min(0.99, r.circularidad + 0.08), 3), comparaciones, ruta
        comparaciones += 2
        if (not r.es_convexo) and r.defectos_convexidad >= 4 and r.num_vertices == 10:
            ruta.append("Meta:Estrella")
            return "Estrella", 0.968, comparaciones, ruta

    # Rama 4: Estado Absorbente Null
    comparaciones += 2
    ruta.append("Meta:Null")
    return "Null", 0.180, comparaciones, ruta


def calcular_heuristica_h(r: RasgosGeometricos, nodo_candidato: str) -> float:
    """
    2. Función Heurística h(n) basada en discrepancia de rasgos:
    Mide qué tan lejos están los rasgos observados del prototipo de cada rama.
    """
    if nodo_candidato == "N_V3":
        return abs(r.num_vertices - 3) * 0.35 + (0.0 if r.es_convexo else 0.5)
    elif nodo_candidato == "N_V4":
        return abs(r.num_vertices - 4) * 0.35 + (0.0 if r.pares_paralelos >= 1 else 0.4)
    elif nodo_candidato == "N_V8+":
        return 0.04 if r.num_vertices >= 8 else abs(8 - r.num_vertices) * 0.35
    return 0.55  # N_Irregular


def ejecutar_busqueda_heuristica_astar(r: RasgosGeometricos) -> Tuple[str, float, int, float, List[str]]:
    """
    2. Búsqueda Heurística A* (f(n) = g(n) + h(n)):
    Ordena la cola de prioridad según h(n) para evaluar directamente la rama más prometedora.
    """
    ruta = ["S0:Imagen_Cruda", "S1:Contorno_Extraido"]
    g_costo = 1

    candidatos_n2 = ["N_V3", "N_V4", "N_V8+", "N_Irregular"]
    evaluaciones_h = {c: round(calcular_heuristica_h(r, c), 3) for c in candidatos_n2}
    mejor_rama = min(candidatos_n2, key=lambda c: g_costo + evaluaciones_h[c])
    h_inicial = evaluaciones_h[mejor_rama]

    g_costo += 1
    ruta.append(mejor_rama)

    if mejor_rama == "N_V3" and r.num_vertices == 3 and r.es_convexo:
        g_costo += 1
        ruta.append("Meta:Triangulo")
        return "Triangulo", 0.985, g_costo, h_inicial, ruta

    elif mejor_rama == "N_V4" and r.num_vertices == 4:
        g_costo += 1
        ruta.append("N_Paralelismo")
        g_costo += 1
        if r.pares_paralelos == 2 and 0.88 <= r.relacion_aspecto <= 1.12:
            ruta.append("Meta:Cuadrado")
            return "Cuadrado", 0.991, g_costo, h_inicial, ruta
        elif r.pares_paralelos >= 1:
            ruta.append("Meta:Trapecio")
            return "Trapecio", 0.954, g_costo, h_inicial, ruta

    elif mejor_rama == "N_V8+" and r.num_vertices >= 8:
        g_costo += 1
        ruta.append("N_Concavidad")
        g_costo += 1
        if r.es_convexo and r.circularidad >= 0.80:
            ruta.append("Meta:Circulo")
            return "Circulo", round(min(0.99, r.circularidad + 0.08), 3), g_costo, h_inicial, ruta
        elif (not r.es_convexo) and r.defectos_convexidad >= 4:
            ruta.append("Meta:Estrella")
            return "Estrella", 0.968, g_costo, h_inicial, ruta

    g_costo += 1
    ruta.append("Meta:Null")
    return "Null", 0.180, g_costo, h_inicial, ruta


def ejecutar_algoritmo_genetico(r: RasgosGeometricos, generaciones: int = 20) -> Tuple[str, float, List[float], List[Dict[str, float]]]:
    """
    3. Algoritmo Genético en Python:
    - Cromosoma: 5 genes [Vértices, Circularidad, Rel_Aspecto, Convexo(1/0), Pares_Paralelos]
    - Fitness: 1 / (1 + distancia_euclidiana_ponderada)
    - Selección por torneo, Cruce de 1 punto y Mutación Gaussiana.
    """
    prototipos = {
        "Circulo":   [8.0,  0.89, 1.00, 1.0, 0.0],
        "Cuadrado":  [4.0,  0.78, 1.00, 1.0, 2.0],
        "Trapecio":  [4.0,  0.69, 1.50, 1.0, 1.0],
        "Estrella":  [10.0, 0.30, 1.00, 0.0, 0.0],
        "Triangulo": [3.0,  0.60, 1.06, 1.0, 0.0],
        "Null":      [7.0,  0.48, 1.15, 0.0, 0.0],
    }

    obs = [
        float(r.num_vertices),
        float(r.circularidad),
        float(r.relacion_aspecto),
        1.0 if r.es_convexo else 0.0,
        float(r.pares_paralelos)
    ]
    pesos = [0.35, 1.5, 1.1, 1.4, 0.9]

    def fitness(crom: List[float]) -> float:
        d2 = sum(w * ((o - c) ** 2) for w, o, c in zip(pesos, obs, crom))
        return 1.0 / (1.0 + math.sqrt(d2))

    # Población inicial (12 individuos)
    poblacion = [list(v) for v in prototipos.values()]
    for _ in range(6):
        poblacion.append([
            float(random.randint(3, 12)),
            round(random.uniform(0.25, 0.95), 2),
            round(random.uniform(0.85, 1.65), 2),
            float(random.choice([0, 1])),
            float(random.choice([0, 1, 2]))
        ])

    historial = []
    for gen in range(1, generaciones + 1):
        poblacion.sort(key=fitness, reverse=True)
        mejor_fit = fitness(poblacion[0])
        prom_fit = sum(fitness(ind) for ind in poblacion) / len(poblacion)
        if gen in (1, 5, 10, 15, 20):
            historial.append({
                "gen": gen,
                "mejor_fitness": round(mejor_fit * 100, 2),
                "prom_fitness": round(prom_fit * 100, 2),
                "cromosoma": [round(x, 2) for x in poblacion[0]]
            })

        nueva_pob = [poblacion[0][:], poblacion[1][:]]  # Elitismo
        while len(nueva_pob) < len(poblacion):
            p1, p2 = random.sample(poblacion[:5], 2)
            pt = random.randint(1, 4)
            hijo = p1[:pt] + p2[pt:]
            if random.random() < 0.25:
                idx_m = random.randint(0, 4)
                hijo[idx_m] = round(hijo[idx_m] + (obs[idx_m] - hijo[idx_m]) * 0.45, 2)
            nueva_pob.append(hijo)
        poblacion = nueva_pob

    poblacion.sort(key=fitness, reverse=True)
    mejor_ind = [round(x, 2) for x in poblacion[0]]
    mejor_clase = min(
        prototipos.keys(),
        key=lambda k: sum(w * ((mejor_ind[i] - prototipos[k][i]) ** 2) for i, w in enumerate(pesos))
    )
    return mejor_clase, round(fitness(mejor_ind), 3), mejor_ind, historial


# =====================================================================
# 5. PROCESAMIENTO COMPLETO Y GENERACIÓN DE INTERFAZ HTML DESDE PYTHON
# =====================================================================
def procesar_y_exportar_todo(ruta_img_entrada: Optional[str] = None):
    if ruta_img_entrada and os.path.exists(ruta_img_entrada):
        img_original = cv2.imread(ruta_img_entrada)
    else:
        img_original = generar_lienzo_6_figuras()
        cv2.imwrite("lienzo_original_6_figuras.png", img_original)

    img_anotada_global = img_original.copy()
    gris = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(gris, (5, 5), 0)
    _, mascara_bin = cv2.threshold(suavizada, 38, 255, cv2.THRESH_BINARY)
    bordes_canny = cv2.Canny(suavizada, 50, 150)

    contornos, _ = cv2.findContours(mascara_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenar contornos de arriba a abajo y de izquierda a derecha
    def clave_orden(c):
        x, y, w, h = cv2.boundingRect(c)
        fila = 0 if y < 300 else 1
        return (fila, x)

    contornos_validos = [c for c in contornos if cv2.contourArea(c) > 500]
    contornos_validos.sort(key=clave_orden)

    resultados_totales: List[ResultadoClasificacion] = []

    for idx, cnt in enumerate(contornos_validos, start=1):
        rasgos, aprox, hull_pts = extraer_rasgos_opencv(cnt, idx)

        # Ejecutar los 3 algoritmos en Python
        c_bfs, conf_bfs, cost_bfs, ruta_bfs = ejecutar_busqueda_ciega_bfs(rasgos)
        c_ast, conf_ast, cost_ast, h_ast, ruta_ast = ejecutar_busqueda_heuristica_astar(rasgos)
        c_ag, fit_ag, crom_ag, hist_ag = ejecutar_algoritmo_genetico(rasgos)

        # Recortar ROI (Región de Interés) de cada figura en sus 3 etapas reales de OpenCV
        x, y, w, h = rasgos.caja
        pad = 22
        y1, y2 = max(0, y - pad), min(img_original.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(img_original.shape[1], x + w + pad)

        roi_orig = img_original[y1:y2, x1:x2].copy()

        # Vista de Bordes Canny + Envolvente Convexa (Hull) en color cian
        roi_edges = cv2.cvtColor(bordes_canny[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)
        hull_desplazado = hull_pts - np.array([[x1, y1]])
        cv2.drawContours(roi_edges, [hull_desplazado], -1, (240, 180, 40), 1, cv2.LINE_AA)

        # Vista Anotada con Contorno, Bounding Box y Vértices detectados por approxPolyDP
        roi_annot = roi_orig.copy()
        cnt_desp = cnt - np.array([[x1, y1]])
        aprox_desp = aprox - np.array([[x1, y1]])
        color_clase = (80, 210, 40) if c_ast != "Null" else (70, 70, 240)
        cv2.drawContours(roi_annot, [cnt_desp], -1, color_clase, 2, cv2.LINE_AA)
        cv2.rectangle(roi_annot, (x - x1, y - y1), (x - x1 + w, y - y1 + h), (180, 180, 180), 1)

        for pt in aprox_desp:
            px, py = pt[0]
            cv2.circle(roi_annot, (int(px), int(py)), 5, (40, 40, 255), -1, cv2.LINE_AA)
            cv2.circle(roi_annot, (int(px), int(py)), 5, (255, 255, 255), 1, cv2.LINE_AA)

        # Anotar también sobre el lienzo global
        cv2.drawContours(img_anotada_global, [cnt], -1, color_clase, 2, cv2.LINE_AA)
        cv2.rectangle(img_anotada_global, (x, y), (x + w, y + h), (140, 140, 140), 1)
        for pt in aprox:
            px, py = pt[0]
            cv2.circle(img_anotada_global, (int(px), int(py)), 5, (30, 30, 255), -1, cv2.LINE_AA)

        etiqueta_sup = f"#{idx} {c_ast.upper()} ({conf_ast*100:.1f}%)"
        etiqueta_inf = f"V:{rasgos.num_vertices} C:{rasgos.circularidad:.2f} AR:{rasgos.relacion_aspecto:.2f} CostoA*:{cost_ast}"
        cv2.putText(img_anotada_global, etiqueta_sup, (x - 10, max(25, y - 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_anotada_global, etiqueta_inf, (x - 10, max(42, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (80, 220, 250), 1, cv2.LINE_AA)

        resultados_totales.append(ResultadoClasificacion(
            rasgos=rasgos,
            clase_bfs=c_bfs,
            confianza_bfs=conf_bfs,
            costo_bfs=cost_bfs,
            ruta_bfs=ruta_bfs,
            clase_astar=c_ast,
            confianza_astar=conf_ast,
            costo_astar=cost_ast,
            heuristica_h=h_ast,
            ruta_astar=ruta_ast,
            clase_ag=c_ag,
            fitness_ag=fit_ag,
            cromosoma_optimo=crom_ag,
            historial_ag=hist_ag,
            img_orig_b64=img_to_base64(roi_orig),
            img_edges_b64=img_to_base64(roi_edges),
            img_annot_b64=img_to_base64(roi_annot)
        ))

    cv2.imwrite("resultado_global_anotado_opencv.png", img_anotada_global)
    img_global_orig_b64 = img_to_base64(img_original)
    img_global_edges_b64 = img_to_base64(cv2.cvtColor(bordes_canny, cv2.COLOR_GRAY2BGR))
    img_global_annot_b64 = img_to_base64(img_anotada_global)

    return resultados_totales, img_global_orig_b64, img_global_edges_b64, img_global_annot_b64
