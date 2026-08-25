#!/usr/bin/env bash
# Arranque de whisper (8091). ESTE ARCHIVO SÍ SE VERSIONA.
#
# POR QUÉ EXISTE. Hasta el 2026-08-25 el arranque era `arrancar.local.sh`, que no
# se versiona porque llevaba la ruta absoluta del volumen (D-037). Consecuencia:
# en una máquina nueva el archivo NO EXISTE, y prender el flujo empezaba por
# reconstruirlo a mano leyendo INSTALL.md. El eslabón más frágil de ARRANCAR.md
# era el paso que ningún archivo del repo podía hacer por vos.
#
# La salida ya estaba inventada en el propio proyecto: el `docker-compose.yml`
# nunca tuvo ese problema porque escribe `${LADUM_WORK_DIR:-../work}` — variable
# con DEFAULT RELATIVO. Eso se versiona sin romper D-037: no hay ninguna ruta de
# ninguna máquina adentro. Esto hace lo mismo para los dos programas nativos.
#
# `arrancar.local.sh` sigue existiendo y sigue sin versionarse, pero pasa a ser
# OPCIONAL y sólo para lo que de verdad es de esta máquina. Si está, se lee.
#
#   bash arrancar.sh              # desde Git Bash, en la carpeta del repo
#
# Variables (todas opcionales):
#   LADUM_WORK_DIR   el volumen compartido. Si no está, se busca hacia arriba.
#   LADUM_PYTHON     intérprete a usar. Si no está, se prueba cuál funciona.
#   WHISPER_PORT     8091 por defecto.
#   WHISPER_BIND_HOST  127.0.0.1. Poner 0.0.0.0 sólo si algo externo lo llama.

set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROGRAMA="whisper"

# ── 1. WSL NO ────────────────────────────────────────────────────────────────
# Esta es LA causa del `exec: python: not found` del 25-ago, y no es obvia:
# desde PowerShell, `bash` NO es Git Bash, es `C:\WINDOWS\system32\bash.exe`, que
# es WSL. Y dentro de WSL no hay `python` (sólo `python3`), ni whisper-cli, ni
# ffmpeg, ni graphify, ni ollama — comprobado: cero de los cuatro. Así que el
# error de python era el primer síntoma de un arranque que iba a fallar entero.
if [ -n "${WSL_DISTRO_NAME:-}" ] || grep -qi microsoft /proc/version 2>/dev/null; then
  cat >&2 <<'FIN'
ERROR: esto está corriendo dentro de WSL, y acá el flujo no puede funcionar.

Ninguno de los binarios que necesita vive en WSL: whisper-cli, ffmpeg, graphify
y ollama están instalados en Windows. Además el binario de whisper es un .exe.

Pasa casi siempre por esto: desde PowerShell, `bash` significa WSL, no Git Bash.

  Abrí Git Bash y corré:   bash arrancar.sh
  O desde PowerShell:      & "C:\Program Files\Git\bin\bash.exe" arrancar.sh
FIN
  exit 1
fi

# ── 2. lo que sea de ESTA máquina, si existe ────────────────────────────────
# Se lee ANTES de resolver nada, para que pueda fijar LADUM_WORK_DIR o
# LADUM_PYTHON. No se versiona y no hace falta que exista.
if [ -f "$AQUI/arrancar.local.sh" ]; then
  # UN `exec` ACÁ ADENTRO SE RECHAZA, no se obedece. El archivo viejo TERMINABA
  # en `exec python ...` porque era el arranque entero; leerlo con `source`
  # arranca el servidor ahí mismo y se saltea todo lo que viene después —
  # incluida la guarda de WSL de arriba. Comprobado el 2026-08-25: pasó en la
  # primera prueba contra una copia sin migrar, y el síntoma no fue un error sino
  # un servidor corriendo que PARECÍA haber pasado los controles.
  if grep -qE '^[[:space:]]*exec[[:space:]]' "$AQUI/arrancar.local.sh"; then
    cat >&2 <<'FIN'
ERROR: arrancar.local.sh todavía tiene un `exec` adentro (es el formato viejo).

Ahora ese archivo es SÓLO overrides: lo lee arrancar.sh con `source`, así que un
`exec` arrancaría el servidor ahí mismo salteándose todas las comprobaciones.

Dejá adentro nada más que las variables de esta máquina, y borrá la línea `exec`:

  export LADUM_WORK_DIR="<ruta al dir que los contenedores montan en /work>"
FIN
    exit 1
  fi
  # shellcheck disable=SC1091
  . "$AQUI/arrancar.local.sh"
  echo "  local:       arrancar.local.sh leído"
fi

# ── 3. un python que de verdad corra ─────────────────────────────────────────
# No alcanza con `command -v`: en Windows `python3` suele ser el stub de la
# Microsoft Store, que existe en el PATH y abre la tienda en vez de ejecutar.
# Por eso cada candidato se PRUEBA, no se asume.
elegir_python() {
  local c
  for c in "${LADUM_PYTHON:-}" python python3 py; do
    [ -n "$c" ] || continue
    if [ "$c" = "py" ]; then
      py -3 -c "import sys" >/dev/null 2>&1 && { echo "py -3"; return 0; }
    elif command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys" >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  return 1
}
if ! PY="$(elegir_python)"; then
  cat >&2 <<'FIN'
ERROR: no encontré ningún Python 3 que funcione.

Se probaron: python, python3, py -3. Los dos servidores nativos usan SÓLO la
biblioteca estándar, así que cualquier Python 3 sirve — no hace falta venv ni
instalar nada.

  Instalalo, o decí cuál usar:  export LADUM_PYTHON=/ruta/al/python
FIN
  exit 1
fi

# ── 4. el volumen compartido, por ruta relativa ──────────────────────────────
# Tiene que ser EL MISMO directorio físico que los contenedores montan en /work:
# si no, los nativos y los contenedores escriben en lugares distintos y el lote
# falla tres capas más abajo con un archivo que "no está". Por eso acá NO se
# inventa un directorio: o se encuentra, o se para.
#
# Se busca hacia arriba porque las dos disposiciones que existen hoy no tienen la
# misma profundidad: `<REPOS>/whisper.cpp` y `<REPOS>/Whispers/whisper.cpp`.
if [ -z "${LADUM_WORK_DIR:-}" ]; then
  for rel in .. ../.. ../../..; do
    cand="$AQUI/$rel/work"
    if [ -d "$cand/_auth" ]; then LADUM_WORK_DIR="$(cd "$cand" && pwd)"; break; fi
    if [ -d "$cand" ] && [ -z "${LADUM_WORK_DIR:-}" ]; then
      LADUM_WORK_DIR="$(cd "$cand" && pwd)"   # sirve, pero sin la marca de _auth
    fi
  done
fi
if [ -z "${LADUM_WORK_DIR:-}" ] || [ ! -d "$LADUM_WORK_DIR" ]; then
  cat >&2 <<FIN
ERROR: no encontré el volumen compartido y no voy a inventar uno.

Busqué un 'work/' subiendo desde $AQUI y no apareció. Si acá eligiera un
directorio por mi cuenta, los programas nativos escribirían en un lado y los
contenedores en otro, y el lote fallaría después con un archivo que 'no está'.

Las dos salidas:
  1. Levantá los contenedores primero: 'docker compose up -d' crea el work/.
  2. O decí cuál es:  export LADUM_WORK_DIR=<ruta al mismo dir que monta /work>
     (en este equipo hoy es el que ya venías usando; ponelo en arrancar.local.sh)
FIN
  exit 1
fi

export LADUM_WORK_DIR
export WHISPER_WORK_DIR="$LADUM_WORK_DIR"
export LADUM_AUTH_FILE="${LADUM_AUTH_FILE:-$LADUM_WORK_DIR/_auth/clients.json}"

# ── 5. lo que va a hacer falta, comprobado ACÁ y no a mitad del primer job ────
falta=""
bin_whisper=""
for c in "$AQUI/build/bin/Release/whisper-cli.exe" "$AQUI/build/bin/whisper-cli.exe" \
         "$AQUI/build/bin/Debug/whisper-cli.exe" "$AQUI/build/bin/whisper-cli"; do
  [ -f "$c" ] && { bin_whisper="$c"; break; }
done
[ -n "$bin_whisper" ] || command -v whisper-cli >/dev/null 2>&1 \
  || falta="$falta\n  · whisper-cli — hay que compilar el repo (ver README de whisper.cpp)"
command -v ffmpeg >/dev/null 2>&1 \
  || falta="$falta\n  · ffmpeg — no está en el PATH"
ls "$AQUI"/models/ggml-*.bin >/dev/null 2>&1 \
  || falta="$falta\n  · modelos — no hay ggml-*.bin en models/ (models/download-ggml-model.sh)"

echo "  programa:    $PROGRAMA"
echo "  python:      $PY"
echo "  work:        $LADUM_WORK_DIR"
echo "  auth:        $LADUM_AUTH_FILE"
echo "  puerto:      ${WHISPER_PORT:-8091}  (bind ${WHISPER_BIND_HOST:-127.0.0.1})"
if [ -n "$falta" ]; then
  # No se para: el servidor arranca igual y su /health lo dice. Pero se avisa
  # ahora, que es cuando todavía se puede arreglar sin perder un lote.
  printf "\n  AVISO: falta algo para que pueda transcribir:%b\n\n" "$falta" >&2
fi

exec $PY "$AQUI/whisper_server.py"
