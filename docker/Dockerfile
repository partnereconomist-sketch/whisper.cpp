# Imagen de whisper_server.py — el transcriptor del flujo LADUM, en contenedor.
#
# POR QUE AHORA (plan de migracion a servidor, F0). Whisper corria NATIVO en el
# host porque necesitaba su `whisper-cli.exe` compilado. En el servidor todo va
# en un compose y `restart` es el unico supervisor, asi que el binario se
# compila ACA, al construir la imagen (P8): el build queda optimizado para la
# CPU de la maquina que lo construye, que es la misma que lo corre.
#
# DOS ETAPAS. La de build tiene el compilador y las fuentes; la final solo el
# binario, sus bibliotecas, ffmpeg y Python. Las GUI de escritorio
# (`whisper_gui.py`, `transcribir_menu.py`) NO entran: dentro del flujo, cero
# GUI (D-021). Por eso aca se copia una lista y no `*.py`, y la lista se
# COMPRUEBA al final (ver el `RUN python` de abajo).
#
# LOS MODELOS NO VAN EN LA IMAGEN. Pesan GB y cambian por otro camino: se montan
# como volumen sobre /app/models, que es la ruta FIJA que lee el servidor
# (`whisper_server.py`, MODELS_DIR = <carpeta del .py>/models). El compose monta
# ahi la carpeta `models/` de este repo, donde ya viven (ignorados por git).
#
# EL BACKEND ES UNA VARIABLE, Y DICE HASTA DONDE LLEGA (D-100):
#   GGML_FLAGS   flags de CMake del backend. Vacio = solo CPU (lo de hoy).
#                BLAS:   --build-arg GGML_FLAGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
#                        --build-arg BUILD_APT="libopenblas-dev" --build-arg RUNTIME_APT="libopenblas0"
#                Vulkan: -DGGML_VULKAN=ON, con libvulkan-dev/glslc y libvulkan1.
#   BUILD_APT / RUNTIME_APT   paquetes extra que ese backend necesita.
#   BUILD_IMAGE / RUNTIME_IMAGE
#                CUDA no entra solo con flags: necesita imagenes base de NVIDIA
#                (nvidia/cuda:*-devel para compilar, *-runtime para correr) y, en
#                el compose, `gpus: all`. ESO NO LO CAMBIA NINGUNA VARIABLE DE
#                ESTE ARCHIVO: es otro despliegue, y se dice en vez de fingirlo.
#   GGML_NATIVE  ON = -march=native. Rapido aca y SIGILL en una CPU mas vieja:
#                la imagen no se lleva a otra maquina, se reconstruye alla.
#
# Lo que el build dejo es lo que /health publica en `dispositivo` (D-099): el
# servidor se lo pregunta al cargador dinamico, no lo supone.
ARG BUILD_IMAGE=debian:bookworm-slim
ARG RUNTIME_IMAGE=python:3.12-slim-bookworm

FROM ${BUILD_IMAGE} AS build
ARG BUILD_APT=""
ARG GGML_FLAGS=""
ARG GGML_NATIVE=ON
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ${BUILD_APT} \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
# Solo lo que el build necesita: lo de afuera lo filtra Dockerfile.dockerignore,
# que va al lado de este archivo para no tocar el `.dockerignore` de upstream.
COPY . .
# Bibliotecas COMPARTIDAS a proposito: son la evidencia de con que backend se
# compilo, y `dispositivo()` las lee. Un estatico no dejaria nada que mirar.
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
        -DGGML_NATIVE=${GGML_NATIVE} \
        -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON -DWHISPER_BUILD_SERVER=OFF \
        -DWHISPER_SDL2=OFF -DWHISPER_CURL=OFF -DWHISPER_COMMON_FFMPEG=OFF ${GGML_FLAGS} \
    && cmake --build build --target whisper-cli -j"$(nproc)" \
    && mkdir -p /opt/whisper/bin /opt/whisper/lib \
    && cp build/bin/whisper-cli /opt/whisper/bin/ \
    && find build -name 'lib*.so*' -exec cp -a {} /opt/whisper/lib/ \;

FROM ${RUNTIME_IMAGE}
ARG RUNTIME_APT=""
# ffmpeg: todo lo que no es WAV de 16 kHz pasa por ahi antes de whisper-cli.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ${RUNTIME_APT} \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/whisper /opt/whisper
# Las bibliotecas en ../lib, al lado de bin/, como en una instalacion Unix: el
# cargador las encuentra por ldconfig, y `dispositivo()` por `ldd`.
RUN echo /opt/whisper/lib > /etc/ld.so.conf.d/whisper.conf && ldconfig
ENV PATH="/opt/whisper/bin:${PATH}"

WORKDIR /app
COPY whisper_server.py auth.py ./
# LA LISTA SE COMPRUEBA, no se confia. El guard de `import auth` es blando a
# proposito (D-022): sin el modulo el servidor arranca SIN autenticacion. Una
# imagen que no lo trajera correria abierta y /health diria `requerida: false`
# -- paso en el Procesador el 2026-08-08. Y el dispositivo tiene que salir del
# build: si `dispositivo()` no ve el backend de CPU, que siempre se compila,
# esta mirando en el lugar equivocado y mentiria igual con una GPU.
RUN python -c "import whisper_server as w; \
assert w.auth is not None, 'auth.py no entro a la imagen'; \
assert w.WHISPER_CLI, 'whisper-cli no esta en el PATH'; \
d = w.dispositivo(); print('dispositivo:', d['device'], d['backends']); \
assert 'cpu' in d['backends'], d"

# Sin esto el bind es 127.0.0.1: inalcanzable desde fuera del contenedor.
ENV WHISPER_BIND_HOST=0.0.0.0
ENV WHISPER_WORK_DIR=/work
# stdout sin buffer: si no, `docker logs` queda vacio hasta que el proceso muera.
ENV PYTHONUNBUFFERED=1

EXPOSE 8091
CMD ["python", "whisper_server.py"]
