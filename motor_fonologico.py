# -*- coding: utf-8 -*-
"""
Motor de lectura fonologica de la Lengua de Señas Venezolana.

Materializa el sistema de transcripcion de Alejandro Oviedo ("Un estudio sobre la
estructura de las senas de la LSV", Universidad de Los Andes, Merida, 2000,
rev. 2003). Toma un video, extrae por cuadro los cinco componentes de cada
segmento y devuelve la lectura en el orden canonico de la Parte 5:

    estructura segmental (D / M / T)
    configuracion manual (CM)
    superficie de la mano
    locacion
    proximidad
    plano horizontal
    rotacion del antebrazo
    rasgos no manuales

Uso:
    python motor_fonologico.py "C:\\ruta\\video.mp4"
    python motor_fonologico.py video.mp4 --paso 2 --guardar-frames cuadros
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

AQUI = os.path.dirname(os.path.abspath(__file__))
RUTA_CATALOGO = os.path.join(AQUI, "datos", "oviedo_cm_114.json")
MODELO_MANOS = os.path.join(AQUI, "modelos", "hand_landmarker.task")
MODELO_POSE = os.path.join(AQUI, "modelos", "pose_landmarker_lite.task")

# ---------------------------------------------------------------- anatomia
WRIST = 0
PULGAR = (1, 2, 3, 4)              # cmc, mcp, ip, punta
DEDOS = {1: (5, 6, 7, 8),          # indice:  mcp, pip, dip, punta
         2: (9, 10, 11, 12),       # medio
         3: (13, 14, 15, 16),      # anular
         4: (17, 18, 19, 20)}      # menique

# Pose (para la locacion y la rotacion del antebrazo)
P_NARIZ, P_OREJA_I, P_OREJA_D = 0, 7, 8
P_HOMBRO_I, P_HOMBRO_D = 11, 12
P_CODO_I, P_CODO_D = 13, 14
P_MUNECA_I, P_MUNECA_D = 15, 16
P_CADERA_I, P_CADERA_D = 23, 24

# ------------------------------------------------------------- umbrales
UMBRAL_APX = 155.0          # articulacion metacarpo-proximal extendida
UMBRAL_AMX = 150.0          # articulacion proxima-media extendida
UMBRAL_ADX = 140.0          # articulacion medio-distal extendida
UMBRAL_REDONDEADO = 168.0   # extendida pero con tension reducida => [°]
UMBRAL_SEP = 14.0           # grados entre falanges proximales vecinas => [sep]
UMBRAL_APIL = 22.0          # diferencia total de extension entre dedos => apilado
UMBRAL_PO = 0.45            # |cos| del pulgar contra la normal de la palma => opuesto
UMBRAL_CONTACTO = 0.42      # distancia pulgar-dedo, en largos de palma
UMBRAL_MOV = 0.02           # velocidad de la muneca (ancho de hombros / cuadro)
MIN_SOSTENIDO = 3           # cuadros que debe durar una CM para contar como tal
                            # (0.11 s a 28 fps: menos que eso es parpadeo)

# Las seis combinaciones de [apx][amx][adx] que ocurren en la LSV (TABLA 13 y 14)
EXT_POR_COMBO = {(1, 1, 1): "+", (0, 0, 0): "-", (1, 0, 0): "^",
                 (1, 0, 1): '"', (0, 0, 1): "-d+", (1, 1, 0): "+d-"}

GRADO = "\u00b0"
ABAJO = np.array([0.0, 1.0, 0.0])   # en la imagen, +y apunta hacia el suelo

NIVELES_CABEZA = [(0.20, "Coronilla"), (0.40, "Frente"), (0.55, "Nariz"),
                  (0.68, "Labios"), (0.82, "Barbilla"), (1.00, "Cuello")]
NIVELES_TORSO = [(0.20, "Esternon"), (0.40, "Pecho"), (0.60, "Costillas"),
                 (0.80, "Abdomen"), (1.00, "Vientre"), (99.0, "Muslos")]


# =====================================================================
# utilidades geometricas
# =====================================================================
def norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def angulo(a, b, c):
    """Angulo en grados en el vertice b, formado por b->a y b->c."""
    return angulo_entre(a - b, c - b)


def angulo_entre(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def etiqueta_extension(bits):
    """(apx, amx, adx) en bits -> simbolo de Oviedo; aproxima si el combo no ocurre."""
    bits = tuple(int(bool(b)) for b in bits)
    if bits in EXT_POR_COMBO:
        return EXT_POR_COMBO[bits]
    mas_cercano = min(EXT_POR_COMBO,
                      key=lambda c: sum(x != y for x, y in zip(c, bits)))
    return EXT_POR_COMBO[mas_cercano] + "?"


# =====================================================================
# catalogo de las 114 CM: etiqueta -> rasgos
# =====================================================================
_RX_EXT = re.compile(r'^([+\-^"](?:d[+\-])?)')
_RX_APIL = re.compile(r"^apil([RC]?)")


def normalizar_etiqueta(s):
    """Unifica glifos: el original usa U+00AD para 'cerrado' y 'º'/'°' indistintamente."""
    for raro in ("\u00ad", "\u2212", "\u2013", "\u2014"):
        s = s.replace(raro, "-")
    s = s.replace("\u00ba", GRADO).replace("*", "")
    return re.sub(r"\s+", "", s)


def leer_extension(s):
    m = _RX_EXT.match(s)
    return (m.group(1), s[m.end():]) if m else ("+", s)


def parsear_etiqueta(etiqueta):
    """Convierte la etiqueta impresa por Oviedo en un diccionario de rasgos."""
    s = normalizar_etiqueta(etiqueta)
    izq, der = s.split("/", 1)

    digitos = re.match(r"^(\d+)", izq)
    sel = tuple(int(c) for c in digitos.group(1))
    resto = izq[digitos.end():]

    ext, resto = leer_extension(resto)
    red = resto.startswith(GRADO)
    resto = resto.lstrip(GRADO)

    apil = None
    m = _RX_APIL.match(resto)
    if m:
        apil = m.group(1) or "?"
        resto = resto[m.end():]
    crz = "crz" in resto
    resto = resto.replace("crz", "")

    cabeza, _, cola_ns = resto.partition(",NSAb-")
    sep = "sep" in cabeza
    nsab = "-" if _ else "+"
    ns_sep = ("sep" in cola_ns) if _ else None

    der = der.strip(",")
    pulgar_pos = "o" if der.startswith("o") else "a"
    pulgar_ext, der = leer_extension(der[1:])

    return {"sel": sel, "ext": ext, "red": red, "apil": apil, "crz": crz,
            "sep": sep, "nsab": nsab, "ns_sep": ns_sep,
            "pulgar_pos": pulgar_pos, "pulgar_ext": pulgar_ext,
            "contacto": "c+" in der, "yema": "y+" in der,
            "ps": "ps" in der, "ds": "ds" in der}


def cargar_catalogo(ruta=RUTA_CATALOGO):
    with open(ruta, encoding="utf-8") as f:
        bruto = json.load(f)
    catalogo = []
    for fila in bruto["configuraciones"]:
        rasgos = parsear_etiqueta(fila["etiqueta"])
        rasgos.update(n=fila["n"], etiqueta=normalizar_etiqueta(fila["etiqueta"]),
                      glosa=fila["glosa"])
        catalogo.append(rasgos)
    # se canonicalizan porque la comparacion es contra lo que produce pintar_etiqueta
    pasivas = [pintar_etiqueta(parsear_etiqueta(e))
               for e in bruto["mano_pasiva_permitidas"]]
    return catalogo, pasivas


def pintar_etiqueta(f):
    """Reconstruye la etiqueta de Oviedo a partir de los rasgos observados."""
    izq = "".join(str(d) for d in f["sel"]) + f["ext"]
    if f["red"]:
        izq += GRADO
    if f["apil"]:
        izq += "apil" + ("" if f["apil"] == "?" else f["apil"])
    if f["crz"]:
        izq += "crz"
    if f["nsab"] == "-":
        izq += ",NSAb-" + ("sep" if f["ns_sep"] else "")
    elif f["sep"]:
        izq += "sep"
    der = f["pulgar_pos"] + f["pulgar_ext"]
    if f["contacto"]:
        der += "c+"
    return izq + "/" + der


# pesos del cotejo: lo que Oviedo senala como distintivo pesa mas
_PESOS = {"sel": 3.0, "ext": 3.0, "pulgar_pos": 2.5, "pulgar_ext": 2.0,
          "nsab": 2.0, "contacto": 1.5, "sep": 1.0, "crz": 1.0,
          "apil": 1.0, "red": 0.75, "ns_sep": 0.75}


def cotejar(observado, catalogo, top=3):
    """CM del catalogo mas parecidas a los rasgos observados, con su ajuste."""
    puntajes = []
    for cm in catalogo:
        mal = 0.0
        for rasgo, peso in _PESOS.items():
            a, b = observado.get(rasgo), cm.get(rasgo)
            if a is None or b is None:
                continue
            if rasgo == "sel":
                mal += peso * (1.0 - len(set(a) & set(b)) /
                               max(len(set(a) | set(b)), 1))
            elif rasgo == "apil":
                if (a or None) == (b or None):
                    continue
                mal += peso * (0.5 if b == "?" and a else 1.0)
            elif a != b:
                mal += peso
        total = sum(p for r, p in _PESOS.items() if cm.get(r) is not None)
        puntajes.append((max(0.0, 1.0 - mal / max(total, 1e-9)), cm))
    puntajes.sort(key=lambda t: -t[0])
    return puntajes[:top]


# =====================================================================
# extraccion de rasgos por cuadro
# =====================================================================
def matriz_mano(lm, mano, sin_espejo=False):
    """lm: arreglo (21,3) con los puntos de MediaPipe Hands de una mano."""
    palma = float(np.linalg.norm(lm[WRIST] - lm[DEDOS[2][0]])) or 1e-6

    # normal de la palma, apuntando hacia donde mira la palma.
    # MediaPipe asume imagen espejada; en un video sin espejar el signo se invierte.
    n = norm(np.cross(lm[DEDOS[1][0]] - lm[WRIST], lm[DEDOS[4][0]] - lm[WRIST]))
    if mano == "Right":
        n = n if sin_espejo else -n
    else:
        n = -n if sin_espejo else n

    dir_dedos = norm(lm[DEDOS[2][0]] - lm[WRIST])
    dir_radial = norm(lm[DEDOS[1][0]] - lm[DEDOS[4][0]])   # hacia el lado del pulgar

    # --- articulaciones de cada dedo: [apx] [amx] [adx]
    ext_bits, ext_ang = {}, {}
    for d, (mcp, pip, dip, punta) in DEDOS.items():
        angulos = (angulo(lm[WRIST], lm[mcp], lm[pip]),
                   angulo(lm[mcp], lm[pip], lm[dip]),
                   angulo(lm[pip], lm[dip], lm[punta]))
        ext_ang[d] = angulos
        ext_bits[d] = (angulos[0] > UMBRAL_APX,
                       angulos[1] > UMBRAL_AMX,
                       angulos[2] > UMBRAL_ADX)

    # Oviedo anota una sola extension para todos los dedos seleccionados: o la
    # mano es uniforme (puno, garra -d+, aplanado ^...) o hay un fondo de dedos
    # plegados que no participan. Un dedo plegado tiene el DIP casi recto, asi
    # que no basta con "no esta cerrado" para darlo por seleccionado.
    etiquetas = {d: etiqueta_extension(ext_bits[d]) for d in DEDOS}
    if len(set(etiquetas.values())) == 1:
        sel = (1, 2, 3, 4)
    else:
        sel = (tuple(d for d in DEDOS if ext_bits[d][0]) or
               tuple(d for d in DEDOS if etiquetas[d] != "-") or (1, 2, 3, 4))
    puno = sel == (1, 2, 3, 4) and etiquetas[1] == "-"
    no_sel = tuple(d for d in DEDOS if d not in sel)

    ext = Counter(etiquetas[d] for d in sel).most_common(1)[0][0]

    def redondeado(dedos):
        angulos = [a for d in dedos for i, a in enumerate(ext_ang[d])
                   if ext_bits[d][i]]
        return bool(angulos) and (sum(angulos) / len(angulos)) < UMBRAL_REDONDEADO

    def separacion(dedos):
        """[sep]: los dedos se separan en la articulacion metacarpo-proximal."""
        pares = list(zip(dedos, dedos[1:]))
        if not pares:
            return False
        vals = [angulo_entre(lm[DEDOS[b][1]] - lm[DEDOS[b][0]],
                             lm[DEDOS[a][1]] - lm[DEDOS[a][0]]) for a, b in pares]
        return (sum(vals) / len(vals)) > UMBRAL_SEP

    def apilado(dedos):
        """[aplR]/[aplC]: los dedos se cierran progresivamente desde un lado."""
        if len(dedos) < 3:
            return None
        puntaje = [sum(ext_ang[d]) / 3.0 for d in dedos]
        if (max(puntaje) - min(puntaje)) < UMBRAL_APIL:
            return None
        if all(puntaje[i] < puntaje[i + 1] for i in range(len(puntaje) - 1)):
            return "R"      # el indice es el mas cerrado: apilado radial
        if all(puntaje[i] > puntaje[i + 1] for i in range(len(puntaje) - 1)):
            return "C"
        return None

    def cruzado(dedos):
        """[crz]: un dedo pasa por encima de su vecino (medido sobre el eje radial)."""
        for a, b in zip(dedos, dedos[1:]):
            base = np.dot(lm[DEDOS[a][0]] - lm[DEDOS[b][0]], dir_radial)
            puntas = np.dot(lm[DEDOS[a][3]] - lm[DEDOS[b][3]], dir_radial)
            if base > 0 and puntas < 0:
                return True
        return False

    # --- pulgar: [Po] y [jmx] [jpx] [jdx]
    cmc, tmcp, tip, tpunta = PULGAR
    pulgar_pos = "o" if abs(float(np.dot(norm(lm[tmcp] - lm[cmc]), n))) > UMBRAL_PO else "a"
    # la CMC es el eje de oposicion (ya cubierto por [Po]); la extension del
    # pulgar se lee en MCP e IP: (1,1)+ (0,0)- (1,0)^ (0,1)"
    jpx = angulo(lm[cmc], lm[tmcp], lm[tip]) > UMBRAL_AMX
    jdx = angulo(lm[tmcp], lm[tip], lm[tpunta]) > UMBRAL_ADX
    pulgar_ext = {(True, True): "+", (False, False): "-",
                  (True, False): "^", (False, True): '"'}[(jpx, jdx)]

    # --- contacto fonologicamente relevante entre pulgar y dedo seleccionado
    dists = []
    for d in sel:
        _, _, dip, punta = DEDOS[d]
        dists.append(np.linalg.norm(lm[tpunta] - lm[punta]))
        dists.append(np.linalg.norm(lm[tpunta] - lm[dip]))
    contacto = bool(dists) and (min(dists) / palma) < UMBRAL_CONTACTO

    return {
        "mano": mano, "palma": palma, "puno": puno,
        "sel": sel, "ext": ext, "red": redondeado(sel),
        "apil": apilado(sel), "crz": cruzado(sel), "sep": separacion(sel),
        "nsab": ("-" if any(ext_bits[d][0] for d in no_sel) else "+") if no_sel else "+",
        "ns_sep": separacion(no_sel) if no_sel else None,
        "ns_red": redondeado(no_sel) if no_sel else None,
        "ns_abierto": all(all(ext_bits[d]) for d in no_sel) if no_sel else None,
        "pulgar_pos": pulgar_pos, "pulgar_ext": pulgar_ext, "contacto": contacto,
        "n_palma": n, "dir_dedos": dir_dedos, "dir_radial": dir_radial,
        "wrist": lm[WRIST].copy(),
    }


# rasgos que definen la CM y se deciden contra un umbral: cada uno puede
# parpadear cuadro a cuadro cuando el gesto queda justo en el limite
RASGOS_VOTABLES = ("sel", "ext", "red", "apil", "crz", "sep", "nsab", "ns_sep",
                   "ns_red", "ns_abierto", "puno", "pulgar_pos", "pulgar_ext",
                   "contacto")


def suavizar_rasgos(registros, mano_idx, radio=2):
    """Voto mayoritario de cada rasgo de la CM en una ventana de 5 cuadros.

    Un rasgo cuyo bit parpadea parte una misma CM sostenida en varias etiquetas
    (4+/a+, 4+°/a+, 4+/a"), lo que inventa cambios de CM y fragmenta el ranking
    de CM sostenidas.  Se vota rasgo por rasgo y no por etiqueta entera para que
    las variantes cercanas se reunan; en empate gana el cuadro central, que es
    el que se esta describiendo.  0.18 s a 28 fps: un cambio real de CM dura
    mas que eso y sobrevive al voto.
    """
    brutos = [reg["manos"][mano_idx] if len(reg["manos"]) > mano_idx else None
              for reg in registros]
    suaves = []
    for i, f in enumerate(brutos):
        if f is None:
            suaves.append(None)
            continue
        voto = dict(f)
        for r in RASGOS_VOTABLES:
            vals = [g[r] for g in brutos[max(0, i - radio):i + radio + 1]
                    if g is not None and r in g]
            if not vals:
                continue
            cuenta = Counter(vals).most_common()
            empate = [v for v, n in cuenta if n == cuenta[0][1]]
            if f[r] not in empate:
                voto[r] = empate[0]
        suaves.append(voto)
    return suaves


def plano_horizontal(f):
    """Cara de la mano que apunta mas hacia el suelo (5 posibilidades en la LSV)."""
    caras = {"palma": f["n_palma"], "dorso": -f["n_palma"],
             "puntaDedos": f["dir_dedos"], "base": -f["dir_dedos"],
             "cubito": -f["dir_radial"]}
    return max(caras, key=lambda c: float(np.dot(norm(caras[c]), ABAJO)))


def rotacion_antebrazo(f, pose_lm, ref):
    """neutra / prona / supina; la referencia cambia si el brazo esta elevado."""
    if pose_lm is None or ref is None:
        return "?"
    i_muneca, i_codo = (P_MUNECA_I, P_CODO_I) if f["mano"] == "Left" \
        else (P_MUNECA_D, P_CODO_D)
    muneca, codo = pose_lm[i_muneca], pose_lm[i_codo]
    if muneca is None or codo is None:
        return "?"
    if muneca[1] < codo[1]:                        # brazo elevado
        proyeccion = float(np.dot(f["n_palma"], ABAJO))
    else:                                          # brazo colgando: mira el frente
        frente = ref["frente"]
        proyeccion = float(np.dot(f["n_palma"], frente))
    if proyeccion > 0.35:
        return "prona"
    if proyeccion < -0.35:
        return "supina"
    return "neutra"


# =====================================================================
# ubicacion (Parte 2)
# =====================================================================
def referencia_corporal(pose_lm):
    """Puntos de referencia del torso, o None si Pose no detecto el cuerpo."""
    if pose_lm is None:
        return None
    claves = [P_NARIZ, P_OREJA_I, P_OREJA_D, P_HOMBRO_I, P_HOMBRO_D,
              P_CADERA_I, P_CADERA_D]
    if any(pose_lm[i] is None for i in claves):
        return None
    hombros = (pose_lm[P_HOMBRO_I] + pose_lm[P_HOMBRO_D]) / 2
    cadera = (pose_lm[P_CADERA_I] + pose_lm[P_CADERA_D]) / 2
    orejas = (pose_lm[P_OREJA_I] + pose_lm[P_OREJA_D]) / 2
    ancho = float(np.linalg.norm(pose_lm[P_HOMBRO_I] - pose_lm[P_HOMBRO_D])) or 1e-6
    frente = pose_lm[P_NARIZ] - hombros
    frente[1] = 0.0
    return {
        "pecho": hombros, "cadera": cadera, "nariz": pose_lm[P_NARIZ],
        "linea_media_x": float((pose_lm[P_NARIZ][0] + hombros[0] + cadera[0]) / 3),
        "ancho_hombros": ancho,
        "y_tope": orejas[1] - 0.8 * (hombros[1] - orejas[1]),
        "y_hombros": hombros[1], "y_cadera": cadera[1],
        "dir_izquierda": norm(pose_lm[P_HOMBRO_I] - pose_lm[P_HOMBRO_D]),
        "frente": norm(frente),
    }


def nivel_horizontal(y, ref):
    """Uno de los 12 ejes horizontales de Oviedo, segun la altura de la mano."""
    if y <= ref["y_hombros"]:
        tramo = (ref["y_hombros"] - ref["y_tope"]) or 1e-6
        for corte, nombre in NIVELES_CABEZA:
            if y <= ref["y_tope"] + corte * tramo:
                return nombre
        return "Cuello"
    tramo = (ref["y_cadera"] - ref["y_hombros"]) or 1e-6
    for corte, nombre in NIVELES_TORSO:
        if y <= ref["y_hombros"] + corte * tramo:
            return nombre
    return "Muslos"


def ubicacion(f, ref, pose_lm):
    """(superficie de la mano, locacion, proximidad) segun la Parte 2."""
    if ref is None:
        return None, None, None
    mano = f["wrist"]

    antebrazo = None
    if pose_lm is not None:
        i_muneca, i_codo = (P_MUNECA_I, P_CODO_I) if f["mano"] == "Left" \
            else (P_MUNECA_D, P_CODO_D)
        if pose_lm[i_muneca] is not None and pose_lm[i_codo] is not None:
            antebrazo = float(np.linalg.norm(pose_lm[i_muneca] - pose_lm[i_codo]))
    if not antebrazo:
        antebrazo = f["palma"] * 2.5

    # distancia al plano del torso a la altura de la mano (p / m / d de Oviedo)
    desplazamiento = np.array([mano[0] - ref["linea_media_x"], 0.0,
                               mano[2] - ref["pecho"][2]])
    proporcion = float(np.linalg.norm(desplazamiento)) / antebrazo

    if proporcion < 0.20:
        proximidad, letra = "contacto", None
    elif proporcion < 0.50:
        proximidad, letra = "proximo", "p"
    else:
        proximidad = "proximo"
        letra = "m" if proporcion < 1.30 else "d"

    # eje vertical V0..V3
    dx = mano - ref["pecho"]
    lateral = abs(float(np.dot(dx, ref["dir_izquierda"]))) / ref["ancho_hombros"]
    eje = "V0" if lateral < 0.25 else "V1" if lateral < 0.90 else \
        "V2" if lateral < 1.60 else "V3"

    # ipsilateral / contralateral
    a_la_izquierda = float(np.dot(dx, ref["dir_izquierda"])) > 0
    mismo_lado = (a_la_izquierda and f["mano"] == "Left") or \
                 (not a_la_izquierda and f["mano"] == "Right")
    lado = "Ipsilateral" if mismo_lado else "Contralateral"

    nivel = nivel_horizontal(mano[1], ref)
    if letra is None:
        locacion = nivel if eje == "V0" else nivel + lado
    else:
        locacion = letra + eje + nivel + ("Ipsi" if mismo_lado else "Contra")

    # superficie de la mano que apunta a la locacion: la parte que apunta o
    # queda mas cerca del punto del cuerpo a la altura de la mano
    if proximidad == "contacto":
        hacia = -ref["frente"]
    else:
        z_cuerpo = ref["nariz"][2] if mano[1] <= ref["y_hombros"] \
            else ref["pecho"][2]
        hacia = norm(np.array([mano[0], mano[1], z_cuerpo]) - mano)
        if float(np.linalg.norm(hacia)) < 1e-6:
            hacia = -ref["frente"]
    dir_pulgar = norm(f["dir_radial"])
    caras = {"PuntaDedos": f["dir_dedos"], "Palma": f["n_palma"],
             "Dorso": -f["n_palma"], "Base": -f["dir_dedos"],
             "Pulgar": dir_pulgar, "Cubito": -dir_pulgar}
    superficie = max(caras, key=lambda c: float(np.dot(norm(caras[c]), hacia)))
    if letra is not None and proximidad == "contacto":
        # Massone y Johnson: locacion espacial con contacto se escribe "Mano en"
        superficie = "Mano"
    return "%s %s" % (superficie, "en" if proximidad == "contacto" else "hacia"), \
        locacion, proximidad


def rasgos_no_manuales(pose_lm, ref):
    """Solo lo que Pose permite ver: inclinacion de la cabeza (Parte 4)."""
    if pose_lm is None or ref is None:
        return "\u2014"
    if pose_lm[P_OREJA_I] is None or pose_lm[P_OREJA_D] is None:
        return "\u2014"
    caida = abs(pose_lm[P_OREJA_I][1] - pose_lm[P_OREJA_D][1]) / ref["ancho_hombros"]
    return "cabeza ladeada" if caida > 0.12 else "\u2014"


# =====================================================================
# lectura del video
# =====================================================================
def leer_cuadros(ruta, paso, max_cuadros, sin_pose, sin_espejo, umbral_conf,
                 guardar_imagen):
    cap = cv2.VideoCapture(ruta, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        sys.exit("No se pudo abrir el video: %s" % ruta)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print("Video: %s" % ruta)
    print("  %d cuadros, %.1f fps; se procesa 1 de cada %d" % (total, fps, paso))

    for modelo in ([MODELO_MANOS] if sin_pose else [MODELO_MANOS, MODELO_POSE]):
        if not os.path.isfile(modelo):
            sys.exit("Falta el modelo de MediaPipe Tasks: %s\n"
                     "Descarguelo con:\n"
                     "  curl -o \"%s\" https://storage.googleapis.com/"
                     "mediapipe-models/%s" % (modelo, modelo,
                     "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                     if "hand" in modelo else
                     "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"))

    video = mp_vision.RunningMode.VIDEO
    manos = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODELO_MANOS),
            running_mode=video, num_hands=2,
            min_hand_detection_confidence=umbral_conf,
            min_tracking_confidence=0.5))
    pose = None if sin_pose else mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODELO_POSE),
            running_mode=video, min_pose_detection_confidence=umbral_conf,
            min_tracking_confidence=0.5, output_segmentation_masks=False))

    registros, indice, procesados = [], 0, 0
    while True:
        ok, cuadro = cap.read()
        if not ok:
            break
        if indice % paso == 0:
            rgb = np.ascontiguousarray(cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB))
            imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ms = int(round(indice * 1000.0 / fps))
            res_manos = manos.detect_for_video(imagen_mp, ms)
            res_pose = pose.detect_for_video(imagen_mp, ms) if pose else None

            h, w = cuadro.shape[:2]
            # MediaPipe normaliza x y z por el ancho e y por el alto: sin volver a
            # pixeles, en un video vertical todos los angulos salen torcidos.
            esc = np.array([w, h, w], dtype=float)

            pose_lm = None
            if res_pose and res_pose.pose_landmarks:
                persona = res_pose.pose_landmarks[0]
                pose_lm = [np.array([p.x, p.y, p.z]) * esc
                           if getattr(p, "visibility", 1.0) > 0.5 else None
                           for p in persona]

            detectadas = []
            for k, lm in enumerate(res_manos.hand_landmarks):
                puntos = np.array([[p.x, p.y, p.z] for p in lm]) * esc
                lado = "Right"
                if res_manos.handedness and res_manos.handedness[k]:
                    lado = res_manos.handedness[k][0].category_name
                detectadas.append(matriz_mano(puntos, lado, sin_espejo))
            detectadas.sort(key=lambda f: (f["mano"], f["wrist"][0]))

            ref = referencia_corporal(pose_lm)
            if ref is not None:
                # una mano que cuelga a un lado con los dedos al suelo no
                # articula: es reposo, no la mano pasiva de una sena bimanual
                corte = (ref["y_hombros"] + ref["y_cadera"]) / 2
                detectadas = [f for f in detectadas
                              if not (float(np.dot(f["dir_dedos"], ABAJO)) > 0.8
                                      and f["wrist"][1] > corte)]

            imagen = None
            if guardar_imagen:
                imagen = cv2.imencode(".jpg", cuadro,
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
            registros.append({"n": indice, "t": indice / fps, "manos": detectadas,
                              "pose": pose_lm, "ref": ref, "imagen": imagen})
            procesados += 1
            if max_cuadros and procesados >= max_cuadros:
                break
        indice += 1

    cap.release()
    manos.close()
    if pose:
        pose.close()
    print("  %d cuadros analizados" % procesados)
    return registros, fps


def pos_normalizada(reg, mano_idx):
    """Muneca referida al pecho y al ancho de hombros: invariante a la distancia."""
    f = reg["manos"][mano_idx]
    ref = reg["ref"]
    if ref is not None:
        return (f["wrist"] - ref["pecho"]) / ref["ancho_hombros"]
    return f["wrist"] / max(f["palma"], 1e-6)


def segmentar(registros, umbral_mov, mano_idx, rasgos):
    """Segmentos D (detencion), M (movimiento) y T (transicion) de la Parte 3."""
    tray, etiquetas = [], []
    for i, reg in enumerate(registros):
        if rasgos[i] is None:
            tray.append(None)
            etiquetas.append(None)
            continue
        tray.append(pos_normalizada(reg, mano_idx))
        etiquetas.append(pintar_etiqueta(rasgos[i]))

    velocidad = []
    for i, p in enumerate(tray):
        if p is None or i == 0 or tray[i - 1] is None:
            velocidad.append(0.0)
        else:
            velocidad.append(float(np.linalg.norm(p - tray[i - 1])))
    for _ in range(2):                       # suavizado de 3 cuadros, dos pasadas
        copia = velocidad[:]
        for i in range(1, len(velocidad) - 1):
            velocidad[i] = (copia[i - 1] + copia[i] + copia[i + 1]) / 3.0

    # primera pasada: agrupar cuadros contiguos con el mismo estado
    brutos, actual = [], None
    for i, (p, v) in enumerate(zip(tray, velocidad)):
        if p is None:
            if actual:
                brutos.append(actual)
                actual = None
            continue
        tipo = "M" if v > umbral_mov else "D"
        if actual and actual["tipo"] == tipo:
            actual["fin"] = i
        else:
            if actual:
                brutos.append(actual)
            actual = {"tipo": tipo, "ini": i, "fin": i}
    if actual:
        brutos.append(actual)

    # una detencion de un solo cuadro entre dos movimientos es una transicion T
    for s in brutos:
        if s["tipo"] == "D" and (s["fin"] - s["ini"]) <= 1:
            s["tipo"] = "T"

    # dentro de una detencion, un cambio de CM es un movimiento sin contorno,
    # pero solo si cada configuracion se sostiene: un diacritico que parpadea
    # en el limite del umbral no parte la detencion en segmentos
    segmentos = []
    for s in brutos:
        if s["tipo"] != "D":
            segmentos.append(s)
            continue
        cortes = []
        for i in range(s["ini"], s["fin"]):
            if not etiquetas[i] or etiquetas[i] == etiquetas[i + 1]:
                continue
            antes = next((k for k in range(i, s["ini"] - 1, -1)
                          if etiquetas[k] != etiquetas[i]), s["ini"] - 1)
            despues = next((k for k in range(i + 1, s["fin"] + 1)
                            if etiquetas[k] != etiquetas[i + 1]), s["fin"] + 1)
            if i - antes >= MIN_SOSTENIDO and despues - i - 1 >= MIN_SOSTENIDO:
                cortes.append(i)
        if not cortes:
            segmentos.append(s)
            continue
        marcas = [s["ini"]] + cortes + [s["fin"] + 1]
        for k in range(len(marcas) - 1):
            ini, fin = marcas[k], max(marcas[k], marcas[k + 1] - 1)
            tipo = "M" if ini in cortes else "D"
            segmentos.append({"tipo": tipo, "ini": ini, "fin": fin})
    return [s for s in segmentos if s["fin"] >= s["ini"]], etiquetas, tray, velocidad


def recortar_fronteras(segmentos, registros, mano_idx):
    """Quita los segmentos extremos que son postura de entrada o de salida.

    Parte 3 de Oviedo: "los movimientos y posturas que la mano pueda describir o
    adoptar para colocarse en la postura descrita no forman parte de la senal
    misma".  Esas posturas viven en las fronteras del video: la mano relajada
    con los dedos colgando (extension -d+) antes de articular, y la caida del
    brazo al terminar, cuando la muneca pasa bajo el torso y el rastreo la
    pierde.  Devuelve (segmentos conservados, segmentos descartados).
    """
    if len(segmentos) <= 1:
        return segmentos, []

    def extension(s):
        vals = [registros[i]["manos"][mano_idx]["ext"]
                for i in range(s["ini"], s["fin"] + 1)
                if len(registros[i]["manos"]) > mano_idx]
        return Counter(vals).most_common(1)[0][0] if vals else None

    # la extension que domina la sena es su sustancia: no se compara la etiqueta
    # completa porque los diacríticos parpadean cuadro a cuadro
    todas = Counter(reg["manos"][mano_idx]["ext"] for reg in registros
                    if len(reg["manos"]) > mano_idx)
    nucleo = todas.most_common(1)[0][0] if todas else None

    def caida(s):
        for i in range(s["fin"] + 1, min(s["fin"] + 4, len(registros))):
            if len(registros[i]["manos"]) > mano_idx:
                return False
        bajos, vistos = 0, 0
        for i in range(s["ini"], s["fin"] + 1):
            reg = registros[i]
            if len(reg["manos"]) <= mano_idx or reg["ref"] is None:
                continue
            corte = (reg["ref"]["y_hombros"] + reg["ref"]["y_cadera"]) / 2
            vistos += 1
            bajos += reg["manos"][mano_idx]["wrist"][1] > corte
        return vistos > 0 and bajos * 2 > vistos

    def es_frontera(s, al_final):
        ext = extension(s)
        if ext == nucleo:            # la extension que sostiene la sena no se toca
            return False
        return ext == "-d+" or (al_final and caida(s))

    ini, fin = 0, len(segmentos)
    while fin - ini > 1 and es_frontera(segmentos[ini], False):
        ini += 1
    while fin - ini > 1 and es_frontera(segmentos[fin - 1], True):
        fin -= 1
    return segmentos[ini:fin], segmentos[:ini] + segmentos[fin:]


def limpiar_trayectoria(puntos):
    """Quita picos de deteccion (mediana) y temblor de MediaPipe (media).

    La geometria del contorno se mide sobre esta trayectoria limpia: el temblor
    cuadro a cuadro inventa giros e inversiones que la mano no hizo.  La media
    de 5 cuadros (0.18 s a 28 fps) no borra una oscilacion voluntaria de la
    mano, que es mas lenta que eso.
    """
    pts = np.array(puntos)
    if len(pts) < 5:                  # un contorno tan corto no aguanta alisado
        return list(pts)
    suave = [np.median(pts[max(0, i - 1):i + 2], axis=0) for i in range(len(pts))]
    return [np.mean(suave[max(0, i - 2):i + 3], axis=0) for i in range(len(pts))]


def cm_extremo(etiquetas, ini, fin, al_inicio):
    """CM sostenida en un extremo del segmento, no la de un solo cuadro.

    Un cambio de CM es un contraste entre dos configuraciones que se sostienen;
    leido en el cuadro de la frontera, el diacritico que parpadea lo inventa.
    """
    if al_inicio:
        rango = range(ini, min(ini + MIN_SOSTENIDO, fin + 1))
    else:
        rango = range(max(fin - MIN_SOSTENIDO + 1, ini), fin + 1)
    vals = [etiquetas[i] for i in rango if i < len(etiquetas) and etiquetas[i]]
    return Counter(vals).most_common(1)[0][0] if vals else None


def clasificar_movimiento(s, tray, etiquetas, registros):
    """Tipo de contorno (Parte 3), mas oscilacion y roce."""
    ini, fin = s["ini"], s["fin"]
    cm_ini = cm_extremo(etiquetas, ini, fin, True)
    cm_fin = cm_extremo(etiquetas, ini, fin, False)
    puntos = limpiar_trayectoria([p for p in tray[ini:fin + 1] if p is not None])

    if len(puntos) < 2:
        if cm_ini != cm_fin:
            return "sin contorno: cambio de CM (%s -> %s)" % (cm_ini, cm_fin)
        return "detencion (sin desplazamiento)"

    recorrido = sum(float(np.linalg.norm(b - a)) for a, b in zip(puntos, puntos[1:]))
    cuerda = float(np.linalg.norm(puntos[-1] - puntos[0]))
    rectitud = cuerda / recorrido if recorrido > 1e-9 else 1.0

    giros, fuertes = [], 0
    for a, b, c in zip(puntos, puntos[1:], puntos[2:]):
        v1, v2 = b - a, c - b
        giro = angulo_entre(v1, v2)
        if giro > 45.0:
            fuertes += 1
        if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
            giros.append(1.0 if np.cross(v1, v2)[2] >= 0 else -1.0)
    inversiones = sum(1 for x, y in zip(giros, giros[1:]) if x != y)

    # giro de la palma sin desplazamiento => movimiento sin contorno
    giro_palma = 0.0
    reg_ini = registros[ini]["manos"]
    reg_fin = registros[min(fin, len(registros) - 1)]["manos"]
    if reg_ini and reg_fin:
        giro_palma = angulo_entre(reg_ini[0]["n_palma"], reg_fin[0]["n_palma"])
    # sin desplazamiento neto tampoco hay contorno, aunque la rectitud no lo
    # delate; solo vale mirarlo en contornos largos, porque en uno de dos o tres
    # cuadros el cambio de CM real no alcanza a desplazarse
    sin_desplazamiento = rectitud < 0.25 or (len(puntos) >= 5 and cuerda < 0.15)
    if sin_desplazamiento and (cm_ini != cm_fin or giro_palma > 40.0):
        partes = []
        if cm_ini != cm_fin:
            partes.append("cambio de CM (%s -> %s)" % (cm_ini, cm_fin))
        if giro_palma > 40.0:
            partes.append("rotacion de la muneca (%.0f%s)" % (giro_palma, GRADO))
        return "sin contorno: " + " + ".join(partes)

    if rectitud > 0.85:
        tipo = "contorno lineal"
    elif rectitud < 0.35 and inversiones <= 1:
        tipo = "contorno circular"
    elif fuertes >= 3:
        tipo = "contorno en zigzag"
    elif inversiones >= 3:
        tipo = "contorno ondulado"
    else:
        tipo = "contorno curvo"

    extras = []
    if inversiones >= 2 and rectitud < 0.7:
        extras.append("oscilacion (%d inversiones)" % inversiones)
    if any(reg["manos"] and reg["manos"][0]["contacto"]
           for reg in registros[ini:fin + 1]):
        extras.append("roce")
    return tipo + ("; " + ", ".join(extras) if extras else "")


def esquema_bimanual(registros):
    """simetrica / alternativa / asimetrica; devuelve (texto, indice activa, pasiva)."""
    con_manos = [r for r in registros if r["manos"]]
    if not con_manos:
        return "senal monomanual", 0, None
    ambos = [r for r in registros if len(r["manos"]) == 2]
    if len(ambos) < max(3, len(con_manos) // 3):
        return "senal monomanual", 0, None

    manos = [[r["manos"][k] for r in ambos] for k in (0, 1)]
    recorrido = [sum(float(np.linalg.norm(b["wrist"] - a["wrist"]))
                     for a, b in zip(lista, lista[1:])) for lista in manos]
    if max(recorrido) > 1e-6 and min(recorrido) < 0.15 * max(recorrido):
        activa = 0 if recorrido[0] >= recorrido[1] else 1
        return "senal bimanual asimetrica", activa, 1 - activa

    dominante = [Counter(pintar_etiqueta(f) for f in lista).most_common(1)[0][0]
                 for lista in manos]
    if dominante[0] == dominante[1]:
        return "senal bimanual simetrica", 0, None
    return "senal bimanual alternativa", 0, None


# =====================================================================
# impresion
# =====================================================================
def describir_segmento(s, registros, rasgos, etiquetas, tray, catalogo, pasivas,
                       mano_idx):
    centro = min((s["ini"] + s["fin"]) // 2, len(registros) - 1)
    reg = registros[centro]
    f = rasgos[centro]
    if f is None:
        return ["  (sin mano visible en este cuadro)"]

    superficie, locacion, proximidad = ubicacion(f, reg["ref"], reg["pose"])
    lineas = []

    if s["tipo"] == "M":
        lineas.append("  movimiento           : %s" %
                      clasificar_movimiento(s, tray, etiquetas, registros))

    etiqueta = pintar_etiqueta(f)
    lineas.append("  configuracion manual : %s" % etiqueta)
    lineas.append("    candidatos         : %s" % ", ".join(
        "#%d %s (%.0f%%)" % (c["n"], c["glosa"], p * 100)
        for p, c in cotejar(f, catalogo, top=3)))
    if f["puno"]:
        lineas.append("    (los cuatro dedos estan cerrados: la seleccion de "
                      "Oviedo no es observable aqui)")
    lineas.append("  superficie de la mano: %s" % (superficie or "?"))
    lineas.append("  locacion             : %s" % (locacion or "?"))
    lineas.append("  proximidad           : %s" % (proximidad or "?"))
    lineas.append("  plano horizontal     : %s" % plano_horizontal(f))
    lineas.append("  rotacion antebrazo   : %s" %
                  rotacion_antebrazo(f, reg["pose"], reg["ref"]))
    lineas.append("  rasgos no manuales   : %s" %
                  rasgos_no_manuales(reg["pose"], reg["ref"]))
    return lineas


def describir_mano_pasiva(registros, catalogo, pasivas, mano_idx):
    etiquetas = [pintar_etiqueta(f) for f in suavizar_rasgos(registros, mano_idx)
                 if f is not None]
    if not etiquetas:
        return
    dominante, veces = Counter(etiquetas).most_common(1)[0]
    print("\nMANO PASIVA (%s)" % ("izquierda" if mano_idx == 1 else "mano 2"))
    print("  configuracion manual : %s  (%d de %d cuadros)" %
          (dominante, veces, len(etiquetas)))
    if dominante in pasivas:
        print("  regla de Oviedo      : CM permitida en la mano pasiva")
    else:
        print("  regla de Oviedo      : FUERA de las 7 CM permitidas "
              "(solo ocurre en senas con clasificador)")
    reg = next((r for r in reversed(registros) if len(r["manos"]) > mano_idx), None)
    if reg:
        f = reg["manos"][mano_idx]
        superficie, locacion, proximidad = ubicacion(f, reg["ref"], reg["pose"])
        print("  ubicacion            : %s | %s | %s" %
              (superficie or "?", locacion or "?", proximidad or "?"))


def transcribir(ruta, paso=1, max_cuadros=0, umbral_mov=UMBRAL_MOV,
                sin_pose=False, sin_espejo=False, umbral_conf=0.5,
                guardar_frames=None, traza=False):
    catalogo, pasivas = cargar_catalogo()
    registros, fps = leer_cuadros(ruta, paso, max_cuadros, sin_pose, sin_espejo,
                                  umbral_conf, bool(guardar_frames))
    con_mano = [r for r in registros if r["manos"]]
    if not con_mano:
        print("\nNo se detecto ninguna mano en el video.")
        return

    vistos = Counter(f["mano"] for r in con_mano for f in r["manos"])
    print("\nManos detectadas: %s" % ", ".join(
        "%s (%d cuadros)" % (k, v) for k, v in vistos.items()))

    esquema, activa, pasiva = esquema_bimanual(registros)
    rasgos = suavizar_rasgos(registros, activa)
    segmentos, etiquetas, tray, velocidad = segmentar(registros, umbral_mov,
                                                      activa, rasgos)
    segmentos, fronteras = recortar_fronteras(segmentos, registros, activa)

    if traza:
        print("\nTraza por cuadro (etiqueta CM | velocidad de la muneca):")
        for i, reg in enumerate(registros):
            print("  %6d  t=%6.2fs  %s" % (
                reg["n"], reg["t"],
                "%s | %.4f" % (etiquetas[i], velocidad[i]) if etiquetas[i]
                else "- sin mano -"))

    estructura = " - ".join(s["tipo"] for s in segmentos) or "(sin segmentos)"
    print("\n" + "=" * 68)
    print("LECTURA DE LA SEÑA")
    print("=" * 68)
    print("esquema articulatorio  : %s" % esquema)
    print("estructura segmental   : %s" % estructura)
    if fronteras:
        print("fronteras descartadas  : %s" % "; ".join(
            "%s cuadros %d-%d" % (s["tipo"], registros[s["ini"]]["n"],
                                  registros[s["fin"]]["n"]) for s in fronteras))
        print("  (posturas para colocarse en la sena o al soltarla: la Parte 3 "
              "las deja fuera de la senal)")
    print("ventana con mano       : %.2f s a %.2f s" %
          (con_mano[0]["t"], con_mano[-1]["t"]))
    if not sin_espejo:
        print("(se asume video espejado, como hace MediaPipe; si la palma y el "
              "dorso salen invertidos, agrega --sin-espejo)")

    if guardar_frames:
        os.makedirs(guardar_frames, exist_ok=True)

    conteo = Counter()
    for s in segmentos:
        conteo[s["tipo"]] += 1
        nombre = "%s%d" % (s["tipo"], conteo[s["tipo"]])
        print("\n%s   cuadros %d-%d   t=%.2f-%.2f s" %
              (nombre, registros[s["ini"]]["n"], registros[s["fin"]]["n"],
               registros[s["ini"]]["t"], registros[s["fin"]]["t"]))
        for linea in describir_segmento(s, registros, rasgos, etiquetas, tray,
                                        catalogo, pasivas, activa):
            print(linea)
        if guardar_frames and registros[(s["ini"] + s["fin"]) // 2]["imagen"] is not None:
            centro = (s["ini"] + s["fin"]) // 2
            imagen = cv2.imdecode(registros[centro]["imagen"], cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(guardar_frames, "%s_cuadro%d.png"
                                     % (nombre, registros[centro]["n"])), imagen)

    if pasiva is not None:
        describir_mano_pasiva(registros, catalogo, pasivas, pasiva)

    dominantes = [etiquetas[i] for s in segmentos
                  for i in range(s["ini"], s["fin"] + 1) if etiquetas[i]]
    if dominantes:
        print("\n" + "-" * 68)
        print("CM mas sostenidas en la sena:")
        cuenta = Counter(dominantes).most_common()
        inestables = [(e, v) for e, v in cuenta if v < MIN_SOSTENIDO]
        for etiqueta, veces in [(e, v) for e, v in cuenta
                                if v >= MIN_SOSTENIDO][:5]:
            top = cotejar(parsear_etiqueta(etiqueta), catalogo, top=3)
            print("  %-30s %4d cuadros   ->  %s" % (
                etiqueta, veces,
                ", ".join("#%d %s" % (c["n"], c["glosa"]) for _, c in top)))
        if inestables:
            print("  otras %d lecturas de menos de %d cuadros (%d cuadros en "
                  "total): la mano esta en transicion y no llegan a sostenerse"
                  % (len(inestables), MIN_SOSTENIDO,
                     sum(v for _, v in inestables)))
    if guardar_frames:
        print("\nCuadros guardados en: %s" % os.path.abspath(guardar_frames))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(
        description="Lee un video y lo transcribe con el sistema de Oviedo (LSV).")
    p.add_argument("video", help="ruta del video (.mp4, .avi, ...)")
    p.add_argument("--paso", type=int, default=1,
                   help="procesar 1 de cada N cuadros (2 o 3 acelera videos largos)")
    p.add_argument("--max-cuadros", type=int, default=0,
                   help="detenerse despues de N cuadros procesados (0 = todos)")
    p.add_argument("--umbral-mov", type=float, default=UMBRAL_MOV,
                   help="velocidad de la muneca que separa M de D (default %.3f)"
                        % UMBRAL_MOV)
    p.add_argument("--umbral-conf", type=float, default=0.5,
                   help="confianza minima de deteccion de MediaPipe")
    p.add_argument("--sin-pose", action="store_true",
                   help="no usar MediaPipe Pose (sin locacion corporal)")
    p.add_argument("--sin-espejo", action="store_true",
                   help="el video NO esta espejado (camara trasera)")
    p.add_argument("--guardar-frames", metavar="CARPETA",
                   help="guarda un PNG por segmento para validar a simple vista")
    p.add_argument("--traza", action="store_true",
                   help="imprime la etiqueta CM y la velocidad cuadro por cuadro")
    a = p.parse_args()

    if not os.path.exists(a.video):
        sys.exit("No existe el archivo: %s" % a.video)
    transcribir(a.video, a.paso, a.max_cuadros, a.umbral_mov, a.sin_pose,
                a.sin_espejo, a.umbral_conf, a.guardar_frames, a.traza)


if __name__ == "__main__":
    main()
