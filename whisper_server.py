#!/usr/bin/env python3
"""whisper_server.py — Whisper bajo el contrato de job estándar v1 (ADR-001, LADUM).

POR QUE EXISTE
  whisper.cpp acá vivía solo detrás de una GUI Tkinter (whisper_gui.py): un
  humano elige archivo y aprieta un botón. Un bot asignador no puede despachar
  contra eso. Este servidor expone la MISMA superficie que el Procesador --
  POST /jobs con el envelope versionado, GET /health con la versión del
  contrato -- para que quien reparte trabajo no tenga que saber que del otro
  lado hay un .exe de C++ en vez de un pipeline de Python.

QUE PRODUCE, Y POR QUE ASI
  Markdown con ANCLAS DE TIEMPO, no un muro de texto. La salida de cada
  programa del flujo se juzga por una sola pregunta: ¿le permite a graphify
  construir un nodo que apunte de vuelta al Raw correcto? Un transcript plano
  de 45 minutos da un nodo gigante que no ubica nada; en bloques anclados
  (`<!-- t:720 ref:video.mp4#t720 -->`), graphify puede decir "esto se discute
  en video.mp4 al minuto 12" y la IA salta ahí en vez de leer todo. Es el
  patrón C del benchmark -- grafo para localizar, lectura selectiva después --
  aplicado a medios, y el equivalente exacto de `archivo#p<pág>#r<n>` que el
  Procesador usa para las regiones de un PDF.

EL IDIOMA ES OBLIGATORIO, A PROPOSITO
  La autodetección de Whisper falló en silencio en una prueba real de este
  proyecto: tradujo un video entero al inglés sin un solo error visible
  (brecha 4). Por eso `params.language` no tiene default -- hay que pedirlo, y
  "auto" es una elección explícita que queda registrada en meta. Un fallo que
  no se ve es peor que uno que rompe.

USO
  python whisper_server.py           # escucha en :8091 (WHISPER_PORT)
  curl 127.0.0.1:8091/health
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

import base64

DIR = Path(__file__).resolve().parent
MODELS_DIR = DIR / "models"

try:
    import auth
except Exception as _e:   # sin auth.py el servidor arranca igual, sin autenticación
    auth = None
    print(f"  AVISO: auth.py no disponible ({type(_e).__name__}); la autenticación queda apagada")

PORT = int(os.environ.get("WHISPER_PORT", "8091"))
# 127.0.0.1 por defecto igual que el Procesador; la imagen Docker lo pone en
# 0.0.0.0 (dentro de un contenedor, loopback es inalcanzable desde el host).
HOST = os.environ.get("WHISPER_BIND_HOST", "127.0.0.1")

CONTRACT_VERSION = "1.0"
CONTRACT_MAJOR = CONTRACT_VERSION.split(".")[0]
TOOL_NAME = "whisper"

WORK_DIR = os.environ.get("WHISPER_WORK_DIR", "/work")
MAX_BODY = 2 * 1024 * 1024 * 1024   # 2 GB: los videos pesan más que los PDF
MAX_CONCURRENT = max(1, int(os.environ.get("WHISPER_MAX_CONCURRENT", "1")))
_semaforo = threading.Semaphore(MAX_CONCURRENT)
_in_flight = 0
_in_flight_lock = threading.Lock()

# Idiomas que acepta whisper.cpp (-l). "auto" es una elección explícita.
IDIOMAS = frozenset("""auto es en pt fr de it nl ru zh ja ko ar hi tr pl uk sv
    da fi no cs el he hu ro th vi id ms ca eu gl""".split())

CAMPOS_ENVELOPE = {"contract", "job_id", "session_id", "tool", "op", "input", "output", "params",
                   "on_behalf_of", "async"}
CAMPOS_INPUT = {"kind", "filename", "content_base64", "ref"}
CAMPOS_OUTPUT = {"kind", "dir"}

# ══════════════════════════════════════════════════════════════════════════════
# AUDITORIA DE FLAGS -- las 65 que expone whisper-cli, cada una con una decisión
#
# Regla del proyecto: "el contrato no se hace a ciegas, revisa todas las
# funciones que tiene el programa y NUNCA las deja caer en default, siempre se
# deciden las flags".
#
# Decidir no es exponer: un contrato con 65 parámetros es inservible para un bot
# asignador. Cada flag recibe una de tres decisiones, y ninguna queda sin ella:
#
#   EXPUESTA -- el contexto de quien llama cambia legítimamente la respuesta,
#               así que viaja como params.<nombre>
#   FIJA     -- la decide este servicio, con su motivo, y se pasa SIEMPRE de
#               forma explícita en el argv -- también cuando el valor coincide
#               con el default de whisper-cli. Es la parte que importa: si una
#               versión futura cambia un default, nuestro comportamiento no se
#               mueve, y si lo hiciera sería porque alguien editó esta tabla.
#   NO_APLICA-- no tiene sentido en un servicio de transcripción por contrato.
#
# GET /health publica esta tabla: quien despacha trabajo puede ver qué se
# decidió sin leer el código.
# ══════════════════════════════════════════════════════════════════════════════

EXPUESTA, FIJA, NO_APLICA = "expuesta", "fija", "no_aplica"

FLAGS = {
    # ── Expuestas: el llamador tiene contexto que el servicio no tiene ────────
    "-l":    (EXPUESTA, "language", "idioma hablado; OBLIGATORIO, ver nota abajo"),
    "-m":    (EXPUESTA, "model", "modelo ggml; calidad/tiempo lo decide quien pide"),
    "-tr":   (EXPUESTA, "translate", "traducir al inglés; cambia el contenido, no la forma"),
    "-tp":   (EXPUESTA, "temperature", "muestreo; 0 = determinista"),
    "-t":    (EXPUESTA, "threads", "hilos; es presupuesto de CPU del host que despacha"),
    "-bo":   (EXPUESTA, "best_of", "candidatos; calidad/tiempo puro"),
    "-bs":   (EXPUESTA, "beam_size", "beam search; calidad/tiempo puro"),
    "-ml":   (EXPUESTA, "max_len", "largo máximo de segmento en caracteres; afecta la granularidad de las anclas"),
    "-sow":  (EXPUESTA, "split_on_word", "cortar en palabra y no en token; solo tiene sentido con max_len"),
    "-ot":   (EXPUESTA, "offset_ms", "procesar desde este milisegundo — permite reintentar SOLO el "
                                     "tramo que falló, en vez de todo el archivo. Verificado: con "
                                     "120000 arranca exacto en 00:02:00"),
    "-d":    (EXPUESTA, "duration_ms", "procesar solo esta duración; misma razón que offset_ms. "
                                       "Verificado: con 60000 corta en ~00:01:01 de un audio de "
                                       "8:33. LIMITE MEDIDO: whisper procesa en ventanas de 30 s, "
                                       "así que por debajo de 30000 el corte no se aplica y se "
                                       "devuelve la ventana completa"),
    "--prompt": (EXPUESTA, "prompt", "contexto inicial; mejora nombres propios y jerga del dominio"),
    "--vad": (EXPUESTA, "vad", "detección de voz: recorta silencios, baja el tiempo en grabaciones largas"),
    "-vt":   (EXPUESTA, "vad_threshold", "umbral del VAD; solo aplica con vad=true"),

    # ── Fijas: las decide el servicio, y se pasan explícitas ──────────────────
    "-nt":   (FIJA, "false", "SIN timestamps no hay anclas, y sin anclas la salida no ubica nada "
                             "en el Raw: es exactamente lo que este servicio existe para producir. "
                             "Es la única flag que no puede exponerse."),
    "-p":    (FIJA, "1", "más de un procesador parte el audio internamente y puede alterar los "
                         "timestamps, que son la salida que importa. El paralelismo se hace "
                         "afuera, por fracciones, donde es reanudable."),
    "-np":   (FIJA, "false", "el resultado se parsea de stdout; silenciarlo lo dejaría vacío"),
    "-ps":   (FIJA, "false", "los tokens especiales ensucian el texto que va al grafo"),
    "-pc":   (FIJA, "false", "los colores ANSI romperían el parseo de los timestamps"),
    "--print-confidence": (FIJA, "false", "mismo motivo: contamina el stdout que se parsea"),
    "-pp":   (FIJA, "false", "el progreso va a stdout y se mezclaría con los segmentos"),
    "-debug": (FIJA, "false", "volcados de diagnóstico, no salida de producto"),
    "-ls":   (FIJA, "false", "log de scores del decoder; ruido para este uso"),
    "-dl":   (FIJA, "false", "sale tras detectar el idioma SIN transcribir: incompatible con la op"),
    "-di":   (FIJA, "false", "diarización estéreo; no está validada en este proyecto y requiere "
                             "audio de dos canales. Candidata a exponer cuando se mida."),
    "-tdrz": (FIJA, "false", "requiere un modelo tdrz que no está instalado; pedirla sin él falla"),
    "-nf":   (FIJA, "false", "el fallback de temperatura es lo que rescata un tramo difícil; "
                             "apagarlo cambia calidad por tiempo sin que nadie lo pida"),
    "-tpi":  (FIJA, "0.20", "incremento del fallback; solo tiene sentido junto con -nf"),
    "-mc":   (FIJA, "-1", "contexto de texto sin límite: mantiene la coherencia entre segmentos"),
    "-ac":   (FIJA, "0", "contexto de audio completo; recortarlo degrada la calidad"),
    "-wt":   (FIJA, "0.01", "umbral de timestamp por palabra; el default está calibrado upstream"),
    "-et":   (FIJA, "2.40", "umbral de entropía para fallo del decoder; idem"),
    "-lpt":  (FIJA, "-1.00", "umbral de logprob para fallo del decoder; idem"),
    "-nth":  (FIJA, "0.60", "umbral de no-habla; idem"),
    "-sns":  (FIJA, "false", "suprimir tokens de no-habla limpia la salida, pero no se midió el "
                             "efecto sobre los timestamps. Fija hasta medirlo."),
    "-fa":   (FIJA, "true", "flash attention: es el default upstream y acelera sin costo de calidad"),
    "-ng":   (FIJA, "false", "no deshabilitar GPU: si el host la tiene, se usa"),
    "-dev":  (FIJA, "0", "primera GPU; multi-GPU es decisión de despliegue, no de job"),
    "--suppress-regex": (FIJA, "", "vacío: filtrar tokens por regex es una decisión de contenido "
                                   "que corresponde a la IA organizadora, no al transcriptor"),
    "--grammar": (FIJA, "", "GBNF restringe la salida a una gramática; no aplica a habla libre"),
    "--grammar-rule": (FIJA, "", "sin gramática, no aplica"),
    "--grammar-penalty": (FIJA, "100.0", "sin gramática, no aplica"),
    "--carry-initial-prompt": (FIJA, "false", "repetir el prompt en cada ventana sesga la "
                                              "transcripción hacia sus palabras"),
    "-on":   (FIJA, "0", "offset por índice de segmento; el recorte se hace por tiempo (-ot/-d), "
                         "que es lo que el llamador puede razonar"),
    "-vm":   (FIJA, "<el silero que haya en models/>", "modelo de VAD: lo elige el servicio, no el "
                                                       "job, pero se pasa EXPLICITO. Dejarlo al "
                                                       "default hacía que --vad sin modelo matara "
                                                       "el proceso con exit 10 a mitad del trabajo"),
    "-vspd": (FIJA, "250", "duración mínima de habla del VAD; default upstream"),
    "-vsd":  (FIJA, "100", "silencio mínimo para cortar segmento; default upstream"),
    "-vmsd": (FIJA, "FLT_MAX", "sin tope de duración de habla; no se pasa, es el default real"),
    "-vp":   (FIJA, "30", "padding del VAD; default upstream"),
    "-vo":   (FIJA, "0.10", "solape entre segmentos del VAD; default upstream"),

    # ── No aplican a un servicio de transcripción por contrato ────────────────
    "-otxt": (NO_APLICA, "", "la salida viaja por el contrato, no como archivo suelto en disco"),
    "-ovtt": (NO_APLICA, "", "idem"),
    "-osrt": (NO_APLICA, "", "idem"),
    "-olrc": (NO_APLICA, "", "idem"),
    "-ocsv": (NO_APLICA, "", "idem"),
    "-oj":   (NO_APLICA, "", "idem — el markdown anclado ya lleva la estructura que el grafo usa"),
    "-ojf":  (NO_APLICA, "", "idem"),
    "-of":   (NO_APLICA, "", "la ruta de salida la decide output.kind/dir del contrato"),
    "-owts": (NO_APLICA, "", "script de karaoke; no es un producto de este flujo"),
    "-fp":   (NO_APLICA, "", "fuente para el video de karaoke; ver -owts"),
    "-oved": (NO_APLICA, "", "OpenVINO: decisión de despliegue del host, no de job"),
    "-dtw":  (NO_APLICA, "", "timestamps a nivel token; las anclas del grafo son por bloque, "
                             "no por palabra. Reevaluar si se necesita precisión fina."),
    "-h":    (NO_APLICA, "", "ayuda del CLI"),
    "-f":    (NO_APLICA, "", "el archivo lo resuelve input.kind del contrato"),
}

# params que acepta el contrato = las EXPUESTAS + las propias de este servicio
CAMPOS_PARAMS = ({nombre for tipo, nombre, _ in FLAGS.values() if tipo == EXPUESTA}
                 | {"segment_seconds", "tramo_seconds"})

MEDIOS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
          ".mp4", ".mkv", ".avi", ".mov", ".webm", ".mpeg", ".mpg", ".m4v"}


# ══════════════════════════════════════════════════════════════════════════════
# Descubrimiento del entorno
# ══════════════════════════════════════════════════════════════════════════════

def _buscar_whisper_cli():
    candidatos = [
        DIR / "build" / "bin" / "Release" / "whisper-cli.exe",
        DIR / "build" / "bin" / "whisper-cli.exe",
        DIR / "build" / "bin" / "Debug" / "whisper-cli.exe",
        DIR / "build" / "bin" / "whisper-cli",
    ]
    for c in candidatos:
        if c.is_file():
            return str(c)
    import shutil
    return shutil.which("whisper-cli")


def _buscar_ffmpeg():
    import shutil
    return os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")


def modelos_disponibles() -> dict:
    """Nombre corto -> ruta. Se lee del disco, no de una lista hardcodeada: pedir
    un modelo que no está tiene que ser 400, no una sustitución silenciosa."""
    if not MODELS_DIR.is_dir():
        return {}
    out = {}
    for f in sorted(MODELS_DIR.glob("ggml-*.bin")):
        if f.name.startswith("for-tests-"):
            continue  # modelos de test del repo upstream: no sirven para transcribir
        out[f.stem.replace("ggml-", "")] = str(f)
    return out


def modelo_vad():
    """Ruta del modelo VAD (silero), o None si no está descargado.

    `--vad` SIN modelo no da un error de parámetros: whisper-cli arranca, carga
    el modelo de transcripción, convierte el audio, y recién ahí muere con
    exit 10 -- después de haber pagado todo el costo. Medido el 2026-08-08
    despachando un clip real: 1.7 s de trabajo tirados y un `whisper_failed`
    genérico que no decía que faltaba un modelo. Pedir VAD sin tenerlo tiene que
    ser 400 con el motivo, igual que pedir un `model` que no está.

    Los `for-tests-*` del repo upstream se excluyen por la misma razón que en
    modelos_disponibles(): no sirven para trabajo real.
    """
    if not MODELS_DIR.is_dir():
        return None
    for f in sorted(MODELS_DIR.glob("*silero*.bin")):
        if not f.name.startswith("for-tests-"):
            return str(f)
    return None


WHISPER_CLI = _buscar_whisper_cli()
FFMPEG = _buscar_ffmpeg()


# ══════════════════════════════════════════════════════════════════════════════
# Contrato: validación estricta (mismo criterio que el Procesador)
#
# Un parámetro mal escrito no puede degradar en silencio a otro comportamiento.
# Acá el caso grave es `language`: whisper.cpp con un idioma inválido no falla,
# autodetecta -- y la autodetección ya tradujo un video entero al inglés sin
# avisar en una prueba real de este proyecto.
# ══════════════════════════════════════════════════════════════════════════════

def dispositivo() -> dict:
    """En que dispositivo puede correr la transcripcion, leido del BUILD.

    POR QUE SE PUBLICA (bloque transversal del PLAN). El dia que haya una GPU la
    pregunta «¿la esta usando?» tiene que contestarse mirando, no midiendo a
    ciegas — y este proyecto ya supuso tres veces en que corria algo y se
    equivoco las tres (D-024).

    COMO SE AVERIGUA, y por que asi. whisper.cpp NO publica su system_info en
    `--help`: solo lo imprime cuando ya esta transcribiendo, asi que preguntarselo
    costaria una corrida. Pero carga sus backends como bibliotecas con nombre
    fijo (`ggml-cuda`, `ggml-vulkan`, `ggml-metal`, `ggml-blas`), asi que las que
    estan JUNTO al binario son evidencia directa de con que se compilo. Si solo
    esta `ggml-cpu`, una GPU en la maquina no cambia nada hasta recompilar.

    Si el binario no aparece, se dice `desconocido` con el motivo. No se adivina
    a partir de que la maquina tenga o no una GPU: que exista no dice que este
    compilada la ruta para usarla, y confundir las dos cosas seria inventar
    justamente el dato que esto viene a dar (D-003).
    """
    ACELERADOS = ("cuda", "vulkan", "metal", "hip", "sycl", "blas", "openvino", "cann")
    if not WHISPER_CLI:
        return {"device": "desconocido", "podria_usar_gpu": None, "backends": [],
                "nota": "no se encontro whisper-cli; sin binario no hay build que mirar"}
    carpeta = Path(WHISPER_CLI).parent
    try:
        libs = [p.stem for p in carpeta.iterdir()
                if p.suffix.lower() in (".dll", ".so", ".dylib") and p.stem.startswith("ggml")]
    except OSError as e:
        return {"device": "desconocido", "podria_usar_gpu": None, "backends": [],
                "nota": f"no se pudo listar {carpeta}: {e}"}
    backends = sorted(l.replace("ggml-", "") for l in libs if l != "ggml")
    acelerados = [b for b in backends if any(a in b for a in ACELERADOS)]
    return {
        "device": "cuda" if any("cuda" in b for b in acelerados) else "cpu",
        "podria_usar_gpu": bool(acelerados),
        "backends": backends,
        "binario": WHISPER_CLI,
        "nota": ("este build solo trae el backend de CPU: una GPU en la maquina no cambiaria "
                 "nada hasta recompilar whisper.cpp con soporte (ver D-060)")
               if not acelerados else f"backends acelerados presentes: {acelerados}",
    }


class ContractError(Exception):
    def __init__(self, code, message, detail=None):
        super().__init__(message)
        self.code, self.message, self.detail = code, message, detail


def _envelope(job_id, status, outputs=None, meta=None, error=None):
    return {"contract": CONTRACT_VERSION, "job_id": job_id, "status": status,
            "outputs": outputs or [], "meta": meta or {}, "error": error}


def _sin_campos_extra(dic, permitidos, contexto):
    extra = set(dic) - permitidos
    if extra:
        raise ContractError("unknown_field",
                            f"campo(s) no reconocido(s) en {contexto}: {sorted(extra)}. "
                            f"Permitidos: {sorted(permitidos)}")


def _exigir_tipo(valor, tipo, camino, esperado):
    malo = not isinstance(valor, tipo) or (tipo is int and isinstance(valor, bool))
    if malo:
        raise ContractError("wrong_type",
                            f"{camino} debe ser {esperado}, llegó {type(valor).__name__} ({valor!r:.60})")
    return valor


def _validar_params(params: dict) -> dict:
    _exigir_tipo(params, dict, "params", "un objeto")
    _sin_campos_extra(params, CAMPOS_PARAMS, "params (op='transcribe')")

    if "language" not in params:
        raise ContractError(
            "missing_fields",
            "params.language es OBLIGATORIO y no tiene default. La autodetección de "
            "Whisper falló en silencio en una prueba real de este proyecto (tradujo un "
            f"video entero al inglés sin avisar). Elegí un idioma o pedí 'auto' de forma "
            f"explícita. Válidos: {sorted(IDIOMAS)}")
    idioma = _exigir_tipo(params["language"], str, "params.language", "una cadena")
    if idioma not in IDIOMAS:
        raise ContractError("unknown_language",
                            f"language '{idioma}' no está en la lista del contrato. "
                            f"whisper.cpp NO falla con un idioma inválido: autodetecta, así que "
                            f"un typo acá se convierte en una transcripción en otro idioma. "
                            f"Válidos: {sorted(IDIOMAS)}")

    disponibles = modelos_disponibles()
    modelo = params.get("model")
    if modelo is None:
        if not disponibles:
            raise ContractError("no_model", f"No hay modelos ggml-*.bin en {MODELS_DIR}")
        # El más grande de los presentes: más lento y más preciso, y es una
        # elección reportada en meta, no invisible.
        modelo = max(disponibles, key=lambda m: Path(disponibles[m]).stat().st_size)
    else:
        _exigir_tipo(modelo, str, "params.model", "una cadena")
        if modelo not in disponibles:
            raise ContractError("model_not_available",
                                f"model '{modelo}' no está en este servidor. "
                                f"Disponibles: {sorted(disponibles)}")

    traducir = params.get("translate", False)
    _exigir_tipo(traducir, bool, "params.translate", "true o false")
    if traducir and idioma == "auto":
        raise ContractError(
            "ambiguous_request",
            "translate=true con language='auto' es la combinación exacta que produjo el "
            "fallo silencioso conocido: no se puede distinguir un video mal detectado de "
            "uno traducido a propósito. Indicá el idioma de ORIGEN explícitamente.")

    # Tamaño de cada invocación independiente de whisper-cli. 0 = archivo entero
    # (comportamiento de siempre). No se elige un default distinto de 0 hasta
    # medir cuánto cuesta en calidad cortar el contexto entre tramos.
    tramo = params.get("tramo_seconds", 0)
    _exigir_tipo(tramo, int, "params.tramo_seconds", "un entero")
    if tramo not in (0, AUTO) and not (30 <= tramo <= 3600):
        raise ContractError("out_of_range",
                            f"params.tramo_seconds debe ser 0 (sin fraccionar), {AUTO} (que decida "
                            f"el servidor segun la duracion) o estar entre 30 y 3600; llego "
                            f"{tramo}. El piso de 30 s no es arbitrario: whisper procesa en "
                            f"ventanas de 30 s y por debajo el recorte no se aplica.")
    seg = params.get("segment_seconds", 120)
    _exigir_tipo(seg, int, "params.segment_seconds", "un entero")
    if not (10 <= seg <= 1800):
        raise ContractError("out_of_range",
                            f"params.segment_seconds debe estar entre 10 y 1800, llegó {seg}")

    def entero(nombre, default, minimo, maximo):
        v = params.get(nombre, default)
        _exigir_tipo(v, int, f"params.{nombre}", "un entero")
        if not (minimo <= v <= maximo):
            raise ContractError("out_of_range",
                                f"params.{nombre} debe estar entre {minimo} y {maximo}, llegó {v}")
        return v

    def numero(nombre, default, minimo, maximo):
        v = params.get(nombre, default)
        if isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        _exigir_tipo(v, float, f"params.{nombre}", "un número")
        if not (minimo <= v <= maximo):
            raise ContractError("out_of_range",
                                f"params.{nombre} debe estar entre {minimo} y {maximo}, llegó {v}")
        return v

    def booleano(nombre, default):
        v = params.get(nombre, default)
        _exigir_tipo(v, bool, f"params.{nombre}", "true o false")
        return v

    resuelto = {
        "language": idioma, "model": modelo, "model_path": disponibles[modelo],
        "translate": traducir, "segment_seconds": seg, "tramo_seconds": tramo,
        "temperature": numero("temperature", 0.0, 0.0, 1.0),
        "threads": entero("threads", max(1, (os.cpu_count() or 4) // 2), 1, 64),
        "best_of": entero("best_of", 5, 1, 20),
        "beam_size": entero("beam_size", 5, 1, 20),
        "max_len": entero("max_len", 0, 0, 1000),
        "split_on_word": booleano("split_on_word", False),
        "offset_ms": entero("offset_ms", 0, 0, 24 * 3600 * 1000),
        "duration_ms": entero("duration_ms", 0, 0, 24 * 3600 * 1000),
        "prompt": params.get("prompt", ""),
        "vad": booleano("vad", False),
        "vad_threshold": numero("vad_threshold", 0.5, 0.0, 1.0),
    }
    _exigir_tipo(resuelto["prompt"], str, "params.prompt", "una cadena")

    if resuelto["vad"] and not modelo_vad():
        raise ContractError(
            "vad_model_missing",
            "params.vad=true necesita el modelo VAD (silero) y no está descargado en "
            f"{MODELS_DIR}. Sin él whisper-cli no rechaza el pedido: muere a mitad del "
            "trabajo con exit 10, después de convertir el audio y cargar el modelo. "
            "Descargalo con models/download-vad-model.sh (o .cmd), o pedí vad=false.")

    if resuelto["split_on_word"] and resuelto["max_len"] == 0:
        raise ContractError(
            "ambiguous_request",
            "split_on_word=true sin max_len no hace nada: whisper solo parte cuando hay un largo "
            "máximo que respetar. Pedí max_len o sacá split_on_word, para que el resultado no "
            "difiera en silencio de lo que se pidió.")

    return resuelto


def construir_comando(params: dict, wav: Path) -> list[str]:
    """argv completo desde la tabla FLAGS.

    Las FIJAS se pasan EXPLICITAS aunque su valor coincida con el default de
    whisper-cli: así una versión futura que cambie un default no puede mover
    nuestro comportamiento sin que alguien edite la tabla. Es la diferencia
    entre "el default nos sirve" y "decidimos este valor".
    """
    cmd = [WHISPER_CLI, "-m", params["model_path"], "-f", str(wav)]

    # Expuestas
    cmd += ["-l", params["language"], "-tp", f"{params['temperature']:.3f}",
            "-t", str(params["threads"]), "-bo", str(params["best_of"]),
            "-bs", str(params["beam_size"]), "-ml", str(params["max_len"]),
            "-ot", str(params["offset_ms"]), "-d", str(params["duration_ms"])]
    if params["translate"]:
        cmd.append("-tr")
    if params["split_on_word"]:
        cmd.append("-sow")
    if params["prompt"]:
        cmd += ["--prompt", params["prompt"]]
    if params["vad"]:
        # -vm EXPLICITO: _validar_params ya garantizó que existe. Antes se
        # omitía ("se resuelve del entorno si hace falta"), que es justo la clase
        # de default heredado que §Funciones prohíbe -- y acá el default heredado
        # era "no hay modelo", así que el job moría después de hacer el trabajo.
        cmd += ["--vad", "-vm", modelo_vad(), "-vt", f"{params['vad_threshold']:.2f}"]

    # Fijas de valor numérico/textual: siempre explícitas.
    for flag, valor in (("-p", "1"), ("-tpi", "0.20"), ("-mc", "-1"), ("-ac", "0"),
                        ("-wt", "0.01"), ("-et", "2.40"), ("-lpt", "-1.00"),
                        ("-nth", "0.60"), ("-dev", "0"), ("-on", "0")):
        cmd += [flag, valor]
    # Fijas booleanas en false: whisper-cli las activa por presencia, así que
    # "pasarlas explícitamente" es NO ponerlas. Quedan listadas en FLAGS para que
    # la decisión sea auditable, que es lo que la regla pide.
    return cmd


# ══════════════════════════════════════════════════════════════════════════════
# AISLAMIENTO POR USUARIO Y SESION dentro del volumen compartido
#
# Estructura (filosofía §Volumen común):  <WORK_DIR>/<usuario>/<sesión>/
#
# LLEGO TARDE, Y VALE LA PENA DECIR POR QUE. El 2026-08-08 el Procesador y
# Synapse pasaron a resolver dentro de <usuario>/<sesión>; este servidor quedó
# resolviendo contra la RAIZ del volumen, y se registró igual como "aislamiento
# ✅". Con un solo inquilino los dos se comportan igual, así que nada lo delató:
# la guarda impedía salir del volumen, no impedía que un job nombrara el archivo
# de otro usuario dentro de él. Lo destapó el primer cliente que despachaba a los
# tres programas con una identidad de usuario -- el bot asignador -- al recibir
# `unknown_field: on_behalf_of`, un campo que los otros dos aceptaban hacía días.
#
# _segmento_seguro es DELIBERADAMENTE idéntico al de extractor.py,
# routes_contract.py y asignador.py: los cuatro tienen que calcular el MISMO
# nombre de carpeta o el que escribe y el que lee no coinciden.
# ══════════════════════════════════════════════════════════════════════════════

# Los usuarios cuelgan de acá, y la infraestructura del volumen (`_auth/`,
# `_jobs/`, `_contract-jobs/`) se queda en la raíz. Son dos espacios de nombres,
# no uno con una lista negra: el saneo convierte lo inválido en `_`, así que
# cualquier reservado de la raíz es alcanzable escribiéndolo (`@auth` -> `_auth`)
# y ninguna lista puede cerrarlo. Lo que lo cierra es la PROFUNDIDAD — un usuario
# llamado `u` aterriza en `u/u/` y no colisiona con nada.
# Es un solo carácter a propósito: las rutas de este flujo ya rondan los 70.
# Si esto cambia, cambia en los SEIS. La batería de conformidad lo comprueba.
SUBDIR_USUARIOS = 'u'


_SEGMENTO_INVALIDO = re.compile(r"[^A-Za-z0-9._-]")


def _segmento_seguro(valor, defecto: str) -> str:
    limpio = _SEGMENTO_INVALIDO.sub("_", str(valor or "").strip())[:64]
    return limpio if limpio and limpio.strip(".") else defecto


def base_de_sesion(session_id=None, cliente=None, on_behalf_of=None) -> Path:
    """Directorio raíz de ESTE usuario y ESTA sesión. Se crea si no existe.

    El usuario sale de la CREDENCIAL, no del envelope: un cliente no puede
    declararse otro, sólo demostrarlo con su clave. La excepción es un cliente
    con scope 'service' (el bot asignador), que sí declara por quién actúa --y
    los programas lo aceptan porque el SERVICIO está autenticado, no porque
    confíen en una cabecera suelta.
    """
    if auth is not None:
        try:
            crudo = auth.usuario_efectivo(cliente, on_behalf_of)
        except PermissionError as e:
            raise ContractError("forbidden_impersonation", str(e))
    else:
        crudo = (cliente or {}).get("nombre") if cliente else None
    base = (Path(WORK_DIR) / SUBDIR_USUARIOS / _segmento_seguro(crudo, "local")
            / _segmento_seguro(session_id, "sin-sesion")).resolve()
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass   # si no se puede crear, la validación de abajo da el error real
    return base


def resolver_en_work_dir(ref: str, campo: str, base: Path) -> Path:
    """Resuelve un ref DENTRO del directorio de la sesión y rechaza lo de afuera.

    Compara por ANCESTROS (`base in target.parents`), no por prefijo de texto.
    La primera versión de este servidor usaba `str(destino).startswith(str(base))`
    con un comentario que afirmaba replicar la guarda del Procesador -- no la
    replicaba, y dejaba un agujero real: con WORK_DIR=/data/work, el ref
    "../work_evil/secret.txt" resuelve a /data/work_evil/secret.txt, que empieza
    con los mismos caracteres y por lo tanto pasaba. No hacía falta ni un symlink
    ni salir a un directorio inexistente: bastaba un hermano cuyo nombre empiece
    igual. Lo encontró la rutina evaluativa nocturna del 2026-08-07.

    `resolve()` va ANTES de comparar, para que '..' y symlinks queden resueltos.

    EL CODIGO DE ERROR ES `path_outside_workdir`, sin guion bajo entre work y dir.
    Hasta el 2026-08-15 este servidor decia `path_outside_work_dir` mientras el
    Procesador, el asignador y Synapse decian `path_outside_workdir`. La guarda
    funcionaba igual en los cuatro -- lo que divergia era el NOMBRE del fallo, que
    es justamente lo que un cliente usa para clasificar. No habia roto nada porque
    hoy nadie ramifica sobre este codigo (el asignador solo lo emite), pero ese es
    el mismo fallo que ya costo un job sano reportado como error: el contrato fija
    el vocabulario, y una palabra que solo conoce un programa obliga al cliente a
    conocer las dos formas o a leer la desconocida como otra cosa.
    """
    if not ref:
        raise ContractError("missing_fields", f"{campo} es obligatorio")
    destino = Path(ref)
    if not destino.is_absolute():
        destino = base / destino
    destino = destino.resolve()
    if destino != base and base not in destino.parents:
        raise ContractError("path_outside_workdir",
                            f"{campo} debe estar dentro de {base} (su usuario y sesión); "
                            f"recibido: {ref}")
    return destino


def _leer_input(entrada, base: Path):
    if not isinstance(entrada, dict):
        raise ContractError("missing_fields", "'input' es obligatorio y debe ser un objeto")
    _sin_campos_extra(entrada, CAMPOS_INPUT, "input")
    kind = entrada.get("kind", "inline")
    filename = entrada.get("filename") or "media"

    if kind == "inline":
        b64 = entrada.get("content_base64")
        if not b64:
            raise ContractError("missing_fields", "input.content_base64 es obligatorio con kind='inline'")
        try:
            datos = base64.b64decode(b64, validate=True)
        except Exception as e:
            raise ContractError("invalid_base64", f"input.content_base64 no es base64 válido: {e}")
    elif kind == "path":
        destino = resolver_en_work_dir(entrada.get("ref"), "input.ref", base)
        if not destino.is_file():
            raise ContractError("input_not_found", f"input.ref no existe: {destino}")
        # input.filename MANDA sobre el nombre en disco, igual que en el
        # Procesador (extractor.py: `inp.get('filename') or target.name`). El
        # nombre del Raw está saneado para ser seguro en cualquier sistema de
        # archivos —sin tildes ni espacios— pero el que el usuario mandó es el
        # que tiene que aparecer en la salida y en el encabezado del markdown.
        # Pisarlo hacía que una misma carpeta entregada mezclara dos convenciones:
        # '7_Evaluación_financiera.md' de los PDF junto a
        # 'Dise_o_y_evaluacion.md' de la media, sin que nada explicara por qué.
        datos = destino.read_bytes()
        filename = entrada.get("filename") or destino.name
    else:
        raise ContractError("unsupported_input_kind",
                            f"input.kind '{kind}' no soportado (usá 'inline' o 'path')")

    ext = Path(filename).suffix.lower()
    if ext and ext not in MEDIOS:
        raise ContractError("unsupported_media",
                            f"extensión '{ext}' no es audio/video reconocido. "
                            f"Soportadas: {sorted(MEDIOS)}")
    return datos, filename


def parse_job(envelope: dict, cliente=None) -> dict:
    if not isinstance(envelope, dict):
        raise ContractError("invalid_envelope", "El body debe ser un objeto JSON")
    _sin_campos_extra(envelope, CAMPOS_ENVELOPE, "el envelope")

    contract = str(envelope.get("contract") or CONTRACT_VERSION)
    if contract.split(".")[0] != CONTRACT_MAJOR:
        raise ContractError("unsupported_contract",
                            f"contract '{contract}' incompatible; este servidor habla {CONTRACT_VERSION}")

    tool = envelope.get("tool", TOOL_NAME)
    if tool != TOOL_NAME:
        raise ContractError("wrong_tool", f"tool '{tool}' no es este servicio ('{TOOL_NAME}')")

    op = envelope.get("op")
    if op != "transcribe":
        raise ContractError("unsupported_op", f"op '{op}' desconocida (este servicio solo hace 'transcribe')")

    # `async` SE VALIDA POR TIPO. Sin esto, {"async": "false"} -- una cadena, que
    # es el error de serializacion tipico entre lenguajes -- se evaluaba por
    # truthiness de Python: "false" no esta vacia, asi que el job se encolaba
    # como asincrono aunque el cliente hubiera pedido sincrono, y sin avisar. El
    # Procesador ya lo rechazaba con 400; era una divergencia del contrato.
    for campo, tipo, esperado in (("async", bool, 'true o false, no la cadena "true"'),
                                  ("job_id", str, "una cadena"), ("session_id", str, "una cadena"),
                                  ("on_behalf_of", str, "una cadena")):
        if envelope.get(campo) is not None:
            _exigir_tipo(envelope[campo], tipo, campo, esperado)

    out_spec = envelope.get("output") or {}
    _exigir_tipo(out_spec, dict, "output", "un objeto")
    _sin_campos_extra(out_spec, CAMPOS_OUTPUT, "output")

    base = base_de_sesion(envelope.get("session_id"), cliente, envelope.get("on_behalf_of"))
    params = _validar_params(envelope.get("params") or {})
    datos, filename = _leer_input(envelope.get("input"), base)

    return {"job_id": envelope.get("job_id") or uuid.uuid4().hex,
            "session_id": envelope.get("session_id"), "params": params,
            "out_spec": out_spec, "datos": datos, "filename": filename, "base": base}


# ══════════════════════════════════════════════════════════════════════════════
# Transcripción
# ══════════════════════════════════════════════════════════════════════════════

# Se capturan las DOS marcas de tiempo. Con solo la de inicio, la duración de un
# audio de 11 s con un único segmento daba 0 -- el encabezado decía "00:00:00 de
# audio" para un archivo con contenido, que es información falsa entrando al grafo.
_LINEA = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\.\d{3}\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.\d{3}\]\s*(.*)")


def _a_wav(origen: Path) -> tuple[Path, bool]:
    """whisper-cli solo lee WAV 16 kHz mono. Devuelve (ruta, hay_que_borrarla)."""
    if origen.suffix.lower() == ".wav":
        return origen, False
    if not FFMPEG:
        raise ContractError("ffmpeg_missing",
                            f"'{origen.suffix}' necesita ffmpeg para convertirse a WAV 16 kHz y "
                            f"no se encontró ffmpeg (PATH o FFMPEG_BIN)")
    destino = origen.with_suffix(".16k.wav")
    r = subprocess.run([FFMPEG, "-y", "-i", str(origen), "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", str(destino)], capture_output=True, text=True)
    if r.returncode != 0 or not destino.is_file():
        err = r.stderr or ""
        # El archivo INCOMPLETO merece su propio código y su propio consejo. Un
        # MP4/M4A guarda el índice (el átomo `moov`) y sin él ffmpeg no puede
        # abrirlo aunque la cabecera `ftyp` esté perfecta -- que es justo lo que
        # engaña: el archivo se ve válido por el principio y por la extensión.
        # Caso real (2026-08-08): 'Primera sesion.m4a', 8.300.000 bytes exactos,
        # ftyp M4A correcto, sin moov. Es una copia o descarga cortada, no un
        # problema del transcriptor, y decir "ffmpeg falló" manda a revisar el
        # lugar equivocado. `retryable` en False: reintentar no lo va a arreglar.
        if "moov atom not found" in err:
            raise ContractError(
                "media_incompleto",
                f"'{origen.name}' está incompleto: tiene cabecera de audio/video válida pero le "
                f"falta el índice (átomo 'moov'), que en MP4/M4A va al final. Es el resultado "
                f"típico de una copia o descarga cortada, o de una grabación que no cerró bien. "
                f"El archivo hay que volver a obtenerlo; reintentar la transcripción no cambia nada.")
        raise RuntimeError(f"ffmpeg falló ({r.returncode}): {err[-500:]}")
    return destino, True


def _segmentos(salida: str) -> list[tuple[int, int, str]]:
    """(segundo_inicio, segundo_fin, texto) de cada línea con timestamp."""
    out = []
    for linea in salida.splitlines():
        m = _LINEA.match(linea.strip())
        if m:
            h, mi, s, h2, mi2, s2, texto = m.groups()
            if texto.strip():
                out.append((int(h) * 3600 + int(mi) * 60 + int(s),
                            int(h2) * 3600 + int(mi2) * 60 + int(s2),
                            texto.strip()))
    return out


def _hhmmss(seg: int) -> str:
    return f"{seg // 3600:02d}:{(seg % 3600) // 60:02d}:{seg % 60:02d}"


def _a_markdown(segmentos, filename, params, meta_extra) -> str:
    """Bloques anclados al tiempo. El ancla es lo que hace que un nodo del grafo
    pueda apuntar al minuto exacto del Raw en vez de al archivo entero."""
    stem = Path(filename).stem
    dur = segmentos[-1][1] if segmentos else 0   # fin del último segmento, no su inicio
    partes = [
        f"# {filename} — transcripción", "",
        f"> {_hhmmss(dur)} de audio · modelo `{params['model']}` · idioma `{params['language']}`"
        + (" · TRADUCIDO al inglés" if params["translate"] else "")
        + f" · {len(segmentos)} segmentos", "",
        f"> Cada bloque está anclado a su momento: la referencia `{stem}#t<segundos>` "
        f"ubica el pasaje en el archivo original.", "",
    ]
    if not segmentos:
        partes += ["_(whisper no devolvió segmentos con timestamp para este archivo)_", ""]
        return "\n".join(partes)

    paso = params["segment_seconds"]
    bloque_actual = -1
    for seg, _fin, texto in segmentos:
        bloque = seg // paso
        if bloque != bloque_actual:
            bloque_actual = bloque
            inicio = bloque * paso
            partes += ["", f"<!-- t:{inicio} ref:{stem}#t{inicio} -->",
                       f"## {_hhmmss(inicio)}", ""]
        partes.append(texto)
    partes.append("")
    return "\n".join(partes)


# ══════════════════════════════════════════════════════════════════════════════
# FRACCIONAMIENTO REANUDABLE
#
# Hasta ahora este servicio transcribia el archivo entero en UNA invocacion: si
# moria en el minuto 20 de 22, se perdian los 20. Es la unica pieza larga del
# flujo sin fraccionar, y la ironica: las flags que hacen falta —`offset_ms` y
# `duration_ms`— ya estaban expuestas y su propia tabla las documenta como
# "permite reintentar SOLO el tramo que fallo, en vez de todo el archivo".
# Estaban construidas y no las usaba nadie.
#
# El shard se guarda con clave de CONTENIDO (audio + parametros + limites del
# tramo), no de posicion, asi que reanudar y reprocesar un archivo cambiado usan
# el mismo mecanismo. Escritura atomica; un shard ilegible cuenta como pendiente.
#
# NO es gratis y por eso es configurable: whisper mantiene contexto de texto
# entre segmentos (`-mc -1`), y cortar en tramos lo pierde en cada frontera. Es
# el mismo intercambio que `--group-bytes` en el grafo, donde aislar de mas
# midio -22% de nodos. Hay que medirlo antes de elegir el default.
# ══════════════════════════════════════════════════════════════════════════════

BYTES_POR_SEGUNDO = 16000 * 2      # el WAV que produce _a_wav: 16 kHz, mono, 16 bits

# `tramo_seconds = -1` -> que el servidor decida por duracion (ver _tramo_auto).
# -1 y no 0: 0 ya significa "archivo entero" y son decisiones distintas.
AUTO = -1
AUTO_MIN_S = int(os.environ.get("WHISPER_AUTO_TRAMO_MIN_S", "900"))
AUTO_TRAMO_S = int(os.environ.get("WHISPER_AUTO_TRAMO_S", "300"))


def _duracion_wav(wav: Path) -> float:
    """Segundos de audio, sacados del tamaño del WAV.

    Sin ffprobe ni dependencias nuevas: _a_wav garantiza 16 kHz mono 16 bits, así
    que la duración es aritmética sobre el tamaño. Menos una cosa que puede
    faltar en un equipo.
    """
    try:
        return max(0.0, (wav.stat().st_size - 44) / BYTES_POR_SEGUNDO)
    except OSError:
        return 0.0


def _tramo_auto(dur_s: float) -> int:
    """Con `tramo_seconds = -1`, cuánto partir según lo que dura el audio.

    POR QUE ACA Y NO EN EL ASIGNADOR. La decisión de política es del que
    despacha, pero el DATO —cuánto dura— sólo este servidor lo tiene barato:
    `_duracion_wav` es aritmética sobre el WAV ya convertido. Para saberlo antes,
    el asignador tendría que parsear el moov atom de un mp4 o cargar ffprobe en
    su contenedor. Así que el asignador pide `-1` (la política: "fraccioná si
    conviene") y acá se aplica con el hecho a la vista. Mismo reparto que `vad`,
    donde el asignador pregunta la capacidad en vez de suponerla.

    EL UMBRAL SALE DE UNA MEDICION Y DE UN JUICIO, y conviene decir cuál es cuál.
    Medido sobre 8 min de audio: fraccionar en 3 tramos costó +10% de tiempo
    (219,8 s -> 241,1 s) y 93,8% de palabras en común, casi todo formato más tres
    términos de dominio. Eso es lo que se PAGA. Lo que se COMPRA es no repetir
    todo cuando el proceso muere, y eso vale en proporción a lo que dura.

    El corte en 15 min es un juicio anclado en ese único dato, no una medición:
    por debajo, rehacer entero cuesta menos que el 6% de deriva; por encima
    (el video de 22 min son 627 s de proceso) deja de serlo. Se puede mover con
    WHISPER_AUTO_TRAMO_MIN_S sin tocar código, justamente porque es un juicio.
    """
    return AUTO_TRAMO_S if dur_s > AUTO_MIN_S else 0


def _tramos(dur_s: float, tramo_s: int) -> list[tuple[int, int]]:
    """[(offset_ms, duration_ms)] en que se parte el audio.

    Devuelve un solo tramo entero cuando no hay que fraccionar, para que el
    camino de siempre no cambie de forma: mismo comando, mismo resultado.

    El límite inferior de 30 s no es arbitrario: whisper procesa en ventanas de
    30 s, así que pedir menos devuelve la ventana completa igual y el recorte no
    se aplica (medido en este repo el 2026-08-05).
    """
    if tramo_s <= 0 or dur_s <= tramo_s or tramo_s < 30:
        return [(0, 0)]
    tramos, t = [], 0
    while t < dur_s:
        tramos.append((int(t * 1000), int(min(tramo_s, dur_s - t) * 1000) + 1000))
        t += tramo_s
    return tramos


def _sin_solape(previos: list, nuevos: list) -> list:
    """Descarta del tramo nuevo lo que el anterior ya transcribió.

    A cada tramo se le pide un segundo de más para que ninguna palabra caiga
    justo en el corte. Sin deduplicar, ese solape aparece DOS VECES en la salida:
    medido, el tramo 1 terminaba con "Dicha carpeta va a ser el directorio de
    trabajo..." y el tramo 2 empezaba con exactamente lo mismo. Un texto repetido
    no es sólo ruido -- para el grafo son dos afirmaciones donde había una.

    Se conserva la del tramo ANTERIOR y se descarta la del nuevo: la primera se
    transcribió con el contexto de todo lo que venía antes, la segunda arranca en
    frío. Ante lo mismo dicho dos veces, gana la que tuvo más contexto.

    SE DESCARTA POR EL FIN, NO POR EL INICIO, y la diferencia costó 19 palabras.
    La primera versión tiraba todo segmento que EMPEZARA antes del fin anterior,
    y eso se llevó uno que empezaba en 360 s pero llegaba hasta 367: solapaba un
    segundo y aportaba seis, así que se perdió una oración entera ("...con una
    opción preliminar de una estructura de cómo se va a desarrollar"). Ahora sólo
    cae el segmento que ya está ENTERAMENTE cubierto.

    Queda un solape chico posible en el que sí se conserva. Es deliberado: repetir
    un segundo de audio es un costo acotado y visible, perder una oración es un
    hueco que nadie nota.
    """
    if not previos or not nuevos:
        return nuevos
    hasta = previos[-1][1]              # fin del último segmento ya aceptado
    return [s for s in nuevos if s[1] > hasta]


def _clave_tramo(datos: bytes, params: dict, off: int, dur: int) -> str:
    h = hashlib.sha256()
    h.update(datos)
    h.update(json.dumps({k: v for k, v in sorted(params.items())
                         if k not in ("offset_ms", "duration_ms")}).encode())
    h.update(f"{off}:{dur}".encode())
    return h.hexdigest()[:16]


def _shard_leer(base: Path, clave: str):
    p = base / "trabajo" / "whisper" / f"{clave}.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return [tuple(x) for x in d["segmentos"]]
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
        return None       # ilegible o a medio escribir = PENDIENTE, nunca hecho


def _shard_guardar(base: Path, clave: str, segmentos) -> None:
    p = base / "trabajo" / "whisper" / f"{clave}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.parcial")
    tmp.write_text(json.dumps({"segmentos": segmentos}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def execute_job(parsed: dict) -> dict:
    job_id, params = parsed["job_id"], parsed["params"]
    t0 = time.perf_counter()

    if not WHISPER_CLI:
        return _envelope(job_id, "error", meta={"op": "transcribe"}, error={
            "code": "whisper_cli_missing",
            "message": f"No se encontró whisper-cli. Compilar whisper.cpp o poner el binario en PATH.",
            "detail": None})

    tmpdir = Path(tempfile.mkdtemp(prefix="whisper-job-"))
    borrar = []
    try:
        origen = tmpdir / parsed["filename"]
        origen.write_bytes(parsed["datos"])
        wav, temporal = _a_wav(origen)
        if temporal:
            borrar.append(wav)

        # Un tramo entero cuando no hay que fraccionar: mismo comando de siempre.
        dur = _duracion_wav(wav)
        pedido = params.get("tramo_seconds", 0)
        tramo_s = _tramo_auto(dur) if pedido == AUTO else pedido
        tramos = _tramos(dur, tramo_s)
        base = parsed.get("base")
        segmentos, reusados = [], 0

        for off, dur in tramos:
            clave = _clave_tramo(parsed["datos"], params, off, dur) if base else None
            if clave:
                previo = _shard_leer(base, clave)
                if previo is not None:
                    # La MISMA deduplicación que el camino fresco. Sin esto un job
                    # reanudado devolvía una salida distinta —y peor, con el solape
                    # repetido— que uno corrido de una sola vez. Reanudar tiene que
                    # dar el mismo resultado, o la reanudabilidad cambia la respuesta.
                    segmentos += _sin_solape(segmentos, previo)
                    reusados += 1
                    continue

            p_tramo = dict(params)
            if len(tramos) > 1:
                p_tramo["offset_ms"], p_tramo["duration_ms"] = off, dur
            cmd = construir_comando(p_tramo, wav)

            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            if r.returncode != 0:
                # Un fallo con stderr VACIO no es lo mismo que uno con mensaje: significa
                # que el proceso murió sin llegar a quejarse (aborto nativo, falta de
                # memoria, DLL). Visto real el 2026-08-05: exit 0xc0e90002 sin una sola
                # línea, y el MISMO comando funcionó al reintentar. Decirlo es lo que
                # permite a quien llama distinguir "reintentá" de "esto no va a andar".
                detalle = (r.stderr or "").strip()
                if not detalle:
                    detalle = (f"whisper-cli murió sin escribir nada en stderr (exit "
                               f"{r.returncode} = {r.returncode & 0xFFFFFFFF:#x}). Suele ser un aborto "
                               f"nativo por recursos, no un error de argumentos: el mismo comando "
                               f"puede funcionar al reintentar. Comando: {' '.join(cmd)}")
                hechos = len(tramos) - (len(tramos) - tramos.index((off, dur)))
                return _envelope(job_id, "error",
                                 meta={"op": "transcribe",
                                       "elapsed_s": round(time.perf_counter() - t0, 2),
                                       "retryable": not (r.stderr or "").strip(),
                                       # Lo ya hecho quedó guardado: reenviar el mismo
                                       # job retoma desde acá, no desde cero.
                                       "tramos": len(tramos), "tramos_hechos": hechos},
                                 error={"code": "whisper_failed",
                                        "message": (f"whisper-cli salió con {r.returncode} en el tramo "
                                                    f"{tramos.index((off, dur)) + 1}/{len(tramos)}"),
                                        "detail": detalle[-1200:]})

            del_tramo = _segmentos(r.stdout)
            # El shard se escribe DESPUES de que el tramo salió bien: un corte
            # antes o durante deja el shard inexistente o intacto, nunca a medias.
            # Se guarda SIN deduplicar, con lo que whisper devolvió: el shard es
            # el resultado del tramo, y el solape es un problema del ensamblado.
            if clave:
                _shard_guardar(base, clave, del_tramo)
            segmentos += _sin_solape(segmentos, del_tramo)
        md = _a_markdown(segmentos, parsed["filename"], params, {})
        meta = {"op": "transcribe", "elapsed_s": round(time.perf_counter() - t0, 2),
                "language": params["language"], "model": params["model"],
                "translated": params["translate"], "segments": len(segmentos),
                # [1] = fin del último segmento. Con [0] (su inicio) un audio de 11 s
                # con un solo segmento reportaba duración 0.
                "duration_s": segmentos[-1][1] if segmentos else 0,
                # Sin esto no hay forma de saber si fraccionó: la primera medición
                # comparó "entero contra entero" sin que nada lo delatara.
                "tramos": len(tramos), "tramos_reusados": reusados,
                "chars": len(md)}
        avisos = []
        if not segmentos:
            avisos.append("whisper-cli no emitió líneas con timestamp; el markdown queda sin "
                          "anclas y por lo tanto no ubica nada en el archivo original")
        if 0 < params["duration_ms"] < 30000:
            # Medido: whisper procesa en ventanas de 30 s. Pedir menos NO recorta,
            # devuelve la ventana entera. Sin este aviso, quien pidió 6 s y recibió
            # 30 no tiene forma de saber que no fue un error suyo.
            avisos.append(f"duration_ms={params['duration_ms']} es menor que la ventana de 30 s de "
                          f"whisper: el recorte no se aplica y la salida cubre la ventana completa")
        if avisos:
            meta["warning"] = " | ".join(avisos)
        return _envelope(job_id, "ok", outputs=[_escribir_salida(md, parsed)], meta=meta)
    except ContractError:
        raise
    except Exception as e:
        return _envelope(job_id, "error",
                         meta={"op": "transcribe", "elapsed_s": round(time.perf_counter() - t0, 2)},
                         error={"code": "transcription_error", "message": str(e),
                                "detail": traceback.format_exc()})
    finally:
        for f in borrar:
            try:
                f.unlink()
            except OSError:
                pass
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def _escribir_salida(md: str, parsed: dict) -> dict:
    spec = parsed["out_spec"]
    if spec.get("kind") == "path":
        # La salida se escribe DENTRO del directorio de la sesión, igual que la
        # entrada: si output.dir se resolviera contra la raíz del volumen, un job
        # podría dejarle archivos a otro usuario en su carpeta.
        destino_dir = resolver_en_work_dir(spec.get("dir") or str(parsed["base"]),
                                           "output.dir", parsed["base"])
        destino_dir.mkdir(parents=True, exist_ok=True)
        destino = destino_dir / (Path(parsed["filename"]).stem + ".md")
        destino.write_text(md, encoding="utf-8")
        return {"kind": "markdown", "ref": str(destino)}
    return {"kind": "markdown", "content": md}


# ══════════════════════════════════════════════════════════════════════════════
# Servidor
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Mitad asíncrona del contrato
#
# POR QUE LLEGA ULTIMA, Y POR QUE IGUAL HABIA QUE HACERLA. El Procesador y
# Synapse aceptan `{"async": true}` desde el 2026-08-04; este servidor no, y el
# asignador lo trataba como excepción con un `SOPORTA_ASYNC = {...}` que lo
# dejaba afuera. Funcionaba -- el LOTE es asíncrono, así que quien sostenía la
# conexión larga era un hilo del asignador y no el usuario. Pero la asimetría se
# paga en cada consumidor nuevo, que tiene que saber que uno de los cuatro
# programas es distinto. Lo señaló la rutina evaluativa del 2026-08-07.
#
# Y es el que MAS lo necesita: es el job más largo del flujo (627 s un video de
# 22 min), o sea el que más tiempo pasa con una conexión HTTP abierta esperando.
#
# ENCOLAR NO AUMENTA EL PARALELISMO. El worker corre bajo el MISMO semáforo que
# el camino síncrono: con MAX_CONCURRENT=1 los jobs se procesan de a uno, en
# orden. Lo único que cambia es que el cliente no tiene que sostener la conexión.
# ══════════════════════════════════════════════════════════════════════════════

ASYNC_TTL_S = max(60, int(os.environ.get("WHISPER_ASYNC_TTL_S", "3600")))
ASYNC_MAX_JOBS = max(16, int(os.environ.get("WHISPER_ASYNC_MAX_JOBS", "512")))
_async_jobs = {}
_async_lock = threading.Lock()


def _estado_dir() -> Path:
    d = Path(WORK_DIR) / "_jobs" / TOOL_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _estado_guardar(job_id: str, reg: dict) -> None:
    """El registro en DISCO existe para una sola pregunta: si el servidor se
    reinicia con un job en vuelo, ¿qué le contesta al que venga a consultarlo?

    Sin esto la respuesta sería `job_not_found`, que manda a revisar el lugar
    equivocado -- el job existió y su trabajo se perdió, que es algo muy
    distinto de un id inventado. Whisper es justamente el job más largo, así que
    es el más probable de estar corriendo cuando algo se cae."""
    try:
        p = _estado_dir() / f"{_segmento_seguro(job_id, 'sin-id')}.json"
        tmp = p.with_suffix(".json.parcial")
        tmp.write_text(json.dumps(reg, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
    except OSError:
        pass          # el disco no puede tumbar un job que por lo demás va bien


def _estado_reconciliar() -> int:
    """Al arrancar: todo lo que quedó `queued`/`running` en disco murió con el
    proceso anterior. Se marca como interrumpido para que el poll diga la verdad
    en vez de inventar un desconocido."""
    tocados = barridos = 0
    try:
        archivos = list(_estado_dir().glob("*.json"))
    except OSError:
        return 0
    for p in archivos:
        try:
            reg = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if reg.get("estado") in ("queued", "running"):
            reg["estado"] = "error"
            reg["interrumpido"] = True
            reg["finished_at"] = time.time()
            _estado_guardar(reg.get("job_id"), reg)
            tocados += 1
            continue
        # BARRIDO. Sin esto cada job asincrono dejaba un .json permanente: una
        # fuga de archivos de estado sin cota en un servidor de larga vida. Se
        # borra lo TERMINADO y vencido, nunca lo que quedo en curso -- eso
        # acaba de marcarse `interrumpido` y su respuesta le sirve a alguien.
        fin = reg.get("finished_at")
        if fin and (time.time() - fin) > ASYNC_TTL_S:
            try:
                p.unlink()
                barridos += 1
            except OSError:
                pass
    if barridos:
        print(f"  {barridos} registro(s) de job vencidos borrados del disco", file=sys.stderr)
    return tocados


def _async_purgar(ahora=None) -> None:
    """Descarta resultados TERMINADOS que nadie vino a buscar. Nunca toca los
    encolados ni los corriendo. Se llama al registrar y al consultar, así que no
    hace falta un hilo de limpieza."""
    ahora = ahora or time.time()
    with _async_lock:
        for jid in [j for j, v in _async_jobs.items()
                    if v["finished_at"] and (ahora - v["finished_at"]) > ASYNC_TTL_S]:
            _async_jobs.pop(jid, None)
        if len(_async_jobs) > ASYNC_MAX_JOBS:
            terminados = sorted(((j, v) for j, v in _async_jobs.items() if v["finished_at"]),
                                key=lambda kv: kv[1]["finished_at"])
            for jid, _ in terminados[: len(_async_jobs) - ASYNC_MAX_JOBS]:
                _async_jobs.pop(jid, None)


def encolar(parsed: dict) -> dict:
    """Registra el job, lanza su hilo y devuelve el envelope de aceptación."""
    job_id = parsed["job_id"]
    _async_purgar()
    with _async_lock:
        vivo = _async_jobs.get(job_id)
        if vivo and vivo["finished_at"] is None:
            raise ContractError(
                "job_already_running",
                f"job_id '{job_id}' ya está encolado o corriendo; usá otro o consultá "
                f"GET /jobs/{job_id}")
        _async_jobs[job_id] = {"status": "queued", "submitted_at": time.time(),
                               "started_at": None, "finished_at": None, "result": None}
    registro = {"job_id": job_id, "estado": "queued", "session_id": parsed.get("session_id"),
                "submitted_at": _async_jobs[job_id]["submitted_at"],
                "started_at": None, "finished_at": None}
    _estado_guardar(job_id, registro)

    def _worker():
        global _in_flight
        try:
            # El semáforo se toma ACA dentro, no antes: hasta que le toque el
            # turno el job sigue en `queued`, que es la verdad que ve quien
            # consulta. Marcarlo `running` al encolar sería mentir sobre la cola.
            with _semaforo:
                with _async_lock:
                    j = _async_jobs.get(job_id)
                    if j:
                        j["status"], j["started_at"] = "running", time.time()
                registro.update(estado="running", started_at=time.time())
                _estado_guardar(job_id, registro)
                with _in_flight_lock:
                    _in_flight += 1
                try:
                    resultado = execute_job(parsed)
                finally:
                    with _in_flight_lock:
                        _in_flight -= 1
        except Exception as e:
            resultado = _envelope(job_id, "error", error={
                "code": "internal_error", "message": str(e), "detail": traceback.format_exc()})
        with _async_lock:
            j = _async_jobs.get(job_id)
            if j:
                j.update(result=resultado, status=resultado.get("status", "error"),
                         finished_at=time.time())
        # EL ENVELOPE COMPLETO VA A DISCO, no solo el estado. Sin esto, un job
        # que TERMINO BIEN antes de un reinicio se perdia entero y el poll
        # respondia `result_expired` -- un mensaje falso, porque no habia vencido
        # ningun TTL: nunca se habia escrito. Whisper es el job mas largo del
        # flujo (627 s un video de 22 min), o sea el que mas probablemente
        # termine mientras el cliente todavia no lo consulto. El Procesador ya
        # lo hacia asi; esto era una divergencia, no una decision.
        registro.update(estado=resultado.get("status", "error"),
                        finished_at=time.time(), envelope=resultado)
        _estado_guardar(job_id, registro)

    threading.Thread(target=_worker, daemon=True).start()
    return _envelope(job_id, "queued", meta={"poll": f"GET /jobs/{job_id}"})


def consultar(job_id: str) -> tuple[dict, int]:
    """(envelope, código). `queued`/`running` devuelven 200 con el estado."""
    _async_purgar()
    with _async_lock:
        j = _async_jobs.get(job_id)
        if j and j["finished_at"] and j["result"]:
            return j["result"], 200
        if j:
            espera = round(time.time() - j["submitted_at"], 1)
            return _envelope(job_id, j["status"],
                             meta={"esperando_s": espera, "en_cola": _en_cola()}), 200

    # No está en memoria: puede ser un id inventado, un resultado vencido, o un
    # job que el reinicio se llevó puesto. Las tres respuestas son distintas.
    try:
        p = _estado_dir() / f"{_segmento_seguro(job_id, 'sin-id')}.json"
        reg = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except (OSError, json.JSONDecodeError):
        reg = None
    if reg and reg.get("envelope"):
        # Terminado antes de un reinicio: se devuelve el trabajo ya hecho en vez
        # de mentir con un TTL que no vencio.
        return reg["envelope"], 200
    if reg and reg.get("interrumpido"):
        return _envelope(job_id, "error", error={
            "code": "job_interrupted",
            "message": "El servidor se reinició mientras este job estaba en curso; su trabajo se "
                       "perdió. Reenvialo — no es un id desconocido ni un resultado vencido. Los "
                       "tramos ya transcritos se reusan, así que no arranca de cero.",
            "detail": f"estado previo: {reg.get('estado')}"}), 200
    if reg:
        return _envelope(job_id, "error", error={
            "code": "result_expired",
            "message": f"El job terminó pero su resultado ya se descartó (TTL {ASYNC_TTL_S}s)."}), 410
    return _envelope(job_id, "error", error={
        "code": "job_not_found", "message": f"No hay ningún job con id '{job_id}'"}), 404


def _en_cola() -> int:
    return sum(1 for v in _async_jobs.values() if v["status"] == "queued")


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """El servidor, con el `SO_REUSEADDR` de Windows apagado (D-148).

    EN WINDOWS `SO_REUSEADDR` NO SIGNIFICA LO QUE PARECE, y `HTTPServer` lo
    prende siempre. En Linux quiere decir «reinicia sin esperar el TIME_WAIT».
    En Windows quiere decir que un SEGUNDO proceso puede atarse al MISMO puerto
    y quedarse con las conexiones nuevas.

    Costo una lectura equivocada el 2026-08-27 en el borde: un proceso viejo
    seguia escuchando, el nuevo imprimio su banner entero y despues fallo al
    atar EN SILENCIO. Las respuestas venian del viejo, y el sintoma apuntaba a
    un codigo que estaba bien.

    El borde se arreglo ese dia; estos tres quedaron afuera y el hueco vivio
    anotado en `TASKS.md` desde el 28-ago hasta que la rutina nocturna del
    2026-09-01 lo volvio a nombrar. Que el arreglo de un programa no se aplique
    a los que comparten la forma es como se acumulan estas cosas.
    """
    daemon_threads = True
    allow_reuse_address = os.name != 'nt'


class Handler(BaseHTTPRequestHandler):

    def log_message(self, *a):
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        ruta = urlparse(self.path).path
        if ruta.startswith("/jobs/"):
            env, code = consultar(ruta[len("/jobs/"):])
            self.send_json(env, code)
            return
        if ruta == "/health":
            with _in_flight_lock:
                vuelo = _in_flight
            self.send_json({
                "status": "ok", "contract": CONTRACT_VERSION, "tool": TOOL_NAME,
                "ops": ["transcribe"], "work_dir": WORK_DIR,
                "dispositivo": dispositivo(),
                # Publicado para que quien despacha decida `vad` CHEQUEANDO en
                # vez de suponer: sin modelo, --vad muere a mitad del trabajo.
                "vad": {"disponible": bool(modelo_vad()), "modelo": modelo_vad()},
                "whisper_cli": WHISPER_CLI, "ffmpeg": FFMPEG,
                "models": sorted(modelos_disponibles()),
                "languages": sorted(IDIOMAS),
                "async": {"supported": True,
                          "submit": 'POST /jobs con {"async": true} -> 202',
                          "poll": "GET /jobs/<job_id>",
                          "result_ttl_s": ASYNC_TTL_S,
                          "en_cola": _en_cola()},
                "params": sorted(CAMPOS_PARAMS),
                # Se publica para que quien despacha pueda PEDIR el modo auto
                # sabiendo que existe, en vez de mandar -1 a ciegas contra un
                # servidor viejo que lo rechazaria por out_of_range.
                "tramo_auto": {"valor": AUTO, "desde_s": AUTO_MIN_S,
                               "tramo_s": AUTO_TRAMO_S},
                # /health ABIERTO a propósito: un orquestador tiene que poder
                # preguntar si hace falta credencial ANTES de mandar una.
                "auth": {
                    "requerida": bool(auth and auth.requerida()),
                    "transporte": "cabecera Authorization: Bearer <clave>",
                    "clientes": len(auth.listar()) if auth else 0,
                },
                "max_concurrent": MAX_CONCURRENT, "in_flight": vuelo,
                # La auditoría completa se publica acá: quien despacha trabajo ve
                # qué se decidió con cada flag del programa sin leer el código.
                "flags_auditadas": {
                    "total": len(FLAGS),
                    "expuestas": sorted(n for t, n, _ in FLAGS.values() if t == EXPUESTA),
                    "fijas": {f: {"valor": v, "motivo": m}
                              for f, (t, v, m) in FLAGS.items() if t == FIJA},
                    "no_aplican": {f: m for f, (t, _v, m) in FLAGS.items() if t == NO_APLICA},
                },
                "note": "params.language es obligatorio: la autodetección puede traducir en silencio",
            })
        else:
            self.send_json({"error": "not_found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/jobs":
            self.send_json(_envelope(None, "error", error={
                "code": "not_found", "message": "El único endpoint de trabajo es POST /jobs"}), 404)
            return
        # Autenticación por CABECERA, no en el envelope: el envelope viaja por
        # logs y reintentos, y ahí la credencial quedaría expuesta. Apagada
        # mientras no haya clientes registrados (ver auth.py).
        cliente = None
        if auth is not None:
            ok, motivo = auth.verificar(self.headers.get("Authorization"))
            if ok:
                # El cliente autenticado define el USUARIO del directorio de
                # trabajo. Antes se descartaba, y por eso este servidor resolvía
                # todo contra la raíz del volumen (ver §AISLAMIENTO).
                cliente = motivo if isinstance(motivo, dict) else None
            if not ok:
                # DRENAR EL BODY ANTES DE RESPONDER. Sin esto el cliente, que
                # todavía está escribiendo (un video son cientos de MB), recibe un
                # error de red en vez del 401 -- medido en el Procesador con un PDF
                # de 320 KB: WinError 10053. Un rechazo que llega como conexión
                # abortada manda a revisar el lugar equivocado.
                try:
                    pendiente = int(self.headers.get("Content-Length", 0) or 0)
                    while pendiente > 0:
                        trozo = self.rfile.read(min(pendiente, 65536))
                        if not trozo:
                            break
                        pendiente -= len(trozo)
                except Exception:
                    pass
                self.send_json(_envelope(None, "error", error={
                    "code": "unauthorized", "message": motivo,
                    "detail": "Formato: Authorization: Bearer <clave>. "
                              "Registrar un cliente: python auth.py add \"<nombre>\"",
                }), 401)
                return

        job_id = None
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                raise ContractError("empty_body", "El body está vacío")
            if n > MAX_BODY:
                raise ContractError("body_too_large", f"El body supera el máximo de {MAX_BODY // (1024**3)} GB")
            try:
                envelope = json.loads(self.rfile.read(n))
            except json.JSONDecodeError as e:
                raise ContractError("invalid_json", f"El body no es JSON válido: {e}")
            job_id = envelope.get("job_id") if isinstance(envelope, dict) else None

            parsed = parse_job(envelope, cliente)
            if envelope.get("async"):
                # 202 recién DESPUES de validar: si validara adentro del hilo,
                # un envelope malo devolvería 202 y el error aparecería en el
                # poll, que es el peor lugar para enterarse de un typo.
                self.send_json(encolar(parsed), 202)
                return
            global _in_flight
            with _semaforo:
                with _in_flight_lock:
                    _in_flight += 1
                try:
                    self.send_json(execute_job(parsed), 200)
                finally:
                    with _in_flight_lock:
                        _in_flight -= 1
        except ContractError as e:
            self.send_json(_envelope(job_id, "error", error={
                "code": e.code, "message": e.message, "detail": e.detail}), 400)
        except Exception as e:
            self.send_json(_envelope(job_id, "error", error={
                "code": "internal_error", "message": str(e),
                "detail": traceback.format_exc()}), 500)


if __name__ == "__main__":
    modelos = modelos_disponibles()
    print(f"\n  Whisper Server — http://{HOST}:{PORT}")
    print(f"  Contrato: v{CONTRACT_VERSION} (tool=\"{TOOL_NAME}\", op=transcribe, work_dir={WORK_DIR})")
    print(f"  whisper-cli: {WHISPER_CLI or 'NO ENCONTRADO — compilar whisper.cpp'}")
    print(f"  ffmpeg:      {FFMPEG or 'NO ENCONTRADO — solo se podrán procesar .wav 16kHz'}")
    print(f"  Modelos:     {', '.join(sorted(modelos)) or 'NINGUNO en ' + str(MODELS_DIR)}")
    _rotos = _estado_reconciliar()
    if _rotos:
        print(f"  AVISO: {_rotos} job(s) quedaron en curso en el arranque anterior; "
              f"marcados como interrumpidos")
    print(f"  Endpoints:   GET /health   POST /jobs   GET /jobs/<id>")
    print(f"  Concurrencia: {MAX_CONCURRENT}")
    print(f"  params.language es OBLIGATORIO (la autodetección puede traducir en silencio)\n")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
