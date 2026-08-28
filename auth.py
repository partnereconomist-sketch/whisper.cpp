#!/usr/bin/env python3
"""auth.py — Autenticación por API key para el servidor de contrato de Whisper.

POR QUE AHORA, SI TODAVIA NO HAY NADIE AFUERA
  Regla del proyecto (filosofía §Seguridad en desarrollo vs producción): la
  seguridad es prioritaria en PRODUCCION, no en desarrollo, y un arreglo LOCAL
  puede postergarse. Pero este no es local: la credencial cambia la FORMA de
  cómo se llama al servicio, así que cada consumidor que se acople sin ella
  habría que rehacerlo. Por eso el hueco se abre ahora y los clientes empiezan a
  mandar la cabecera desde ya, aunque el servidor todavía no la exija.

APAGADA POR DEFECTO, A PROPOSITO
  Sin clientes registrados, `requerida()` es False y todo pasa igual que antes:
  no frena el desarrollo. Registrar el primer cliente la enciende. GET /health
  publica en qué estado está, para que nadie tenga que adivinar por qué recibe
  401 -- o por qué NO lo recibe.

VA EN CABECERA, NO EN EL ENVELOPE
  `Authorization: Bearer <key>`. Decisión del usuario, y el motivo es bueno: el
  envelope viaja por logs, shards y reintentos; si un tercero accede al contrato
  de alguien más, no obtiene además su credencial. Como efecto colateral, el
  contrato de job no cambia de versión por esto.

MODELO
  Mismo que cerebro-mcp/auth.mjs, ya construido y validado: se guarda el SHA-256
  de la clave, nunca la clave; cada cliente tiene nombre y scope; la comparación
  es de tiempo constante. Se duplica en vez de compartirse porque el ADR-001
  prohíbe que un programa importe código de otro -- son contenedores
  independientes.

USO
  python auth.py add "bot-asignador" [--scope full|read|service]
  python auth.py list
  python auth.py revoke "bot-asignador"
"""
import hashlib
import hmac
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

DIR = Path(__file__).resolve().parent

# REGISTRO COMPARTIDO por los tres programas del flujo. Antes cada uno tenía el
# suyo: dar de alta un cliente eran tres operaciones y revocarlo también, así que
# una revocación olvidada dejaba una puerta abierta sin que nadie lo notara.
# LADUM_AUTH_FILE apunta a un único archivo alcanzable por todos; si no está, se
# cae al archivo propio del programa (desarrollo aislado).
AUTH_FILE = Path(os.environ.get('LADUM_AUTH_FILE')
                 or os.environ.get('WHISPER_AUTH_FILE')
                 or (DIR / '.auth' / 'clients.json'))
REGISTRO_PROPIO = not (os.environ.get('LADUM_AUTH_FILE')
                       or os.environ.get('WHISPER_AUTH_FILE'))

# SCOPES
#   read    -- solo consultar (hoy ninguna op del contrato lo es; queda declarado)
#   full    -- puede pedir trabajo, actuando SIEMPRE como sí mismo
#   service -- además puede declarar EN NOMBRE DE QUIEN trabaja (`on_behalf_of`).
#              Es lo que va a necesitar el bot asignador: recibe el pedido de un
#              usuario y despacha a los programas por él. Los programas lo aceptan
#              porque el SERVICIO está autenticado, no porque confíen en una
#              cabecera suelta. Un cliente 'full' que mande on_behalf_of recibe un
#              error, no un silencio.
SCOPES = ('read', 'full', 'service')
# El tope por scope es por OPERACION del contrato, no por endpoint: /jobs es uno solo.
OPS_SOLO_LECTURA = frozenset()
# Prefijo común: la clave sirve para los tres programas, así que no lleva el
# nombre de ninguno.
PREFIJO_CLAVE = 'ladum_'


def _hash(clave: str) -> str:
    return hashlib.sha256(clave.encode('utf-8')).hexdigest()


def _cargar() -> dict:
    try:
        return json.loads(AUTH_FILE.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _guardar(clientes: dict) -> None:
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = AUTH_FILE.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(clientes, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, AUTH_FILE)
    try:  # el archivo de claves no debería ser legible por otros usuarios
        os.chmod(AUTH_FILE, 0o600)
    except OSError:
        pass


def requerida() -> bool:
    """True solo si hay al menos un cliente registrado. Sin clientes, el servidor
    funciona como siempre -- la seguridad no frena el desarrollo."""
    return bool(_cargar())


def agregar(nombre: str, scope: str = 'full', reemplazar: bool = False) -> str:
    if scope not in SCOPES:
        raise ValueError(f"scope inválido '{scope}' — usar uno de {SCOPES}")
    clientes = _cargar()
    # DOS CLIENTES CON EL MISMO NOMBRE SON DOS PUERTAS, NO UNA RENOVADA. El
    # registro se indexa por HASH, así que `add joshua` sobre un `joshua` que ya
    # existe no reemplaza nada: deja las dos credenciales vivas. Quien creía
    # haber rotado la suya sigue teniendo la anterior abierta, y nada se lo dice.
    #
    # Pasó el 2026-08-28: reemitir por las dudas dejó dos `borde` y dos `joshua`
    # en el registro compartido, las cuatro funcionando. Se descubrió porque una
    # clave que ya se creía sustituida seguía contestando 200.
    previos = [h for h, c in clientes.items() if c.get('nombre') == nombre]
    if previos and not reemplazar:
        raise ValueError(
            f"ya existe un cliente '{nombre}' ({len(previos)} credencial(es) viva(s)). "
            f"Emitir otra NO revoca la anterior: quedarían todas abiertas, y sólo se ve "
            f"corriendo `auth.py list`. Usá --reemplazar para revocar las viejas y emitir una "
            f"nueva, o elegí otro nombre.")
    for h in previos:
        clientes.pop(h)
    clave = PREFIJO_CLAVE + secrets.token_hex(24)
    clientes[_hash(clave)] = {'nombre': nombre, 'scope': scope,
                              'creado': datetime.now(timezone.utc).isoformat()}
    _guardar(clientes)
    return clave  # se devuelve UNA sola vez: después solo queda el hash


def revocar(nombre: str) -> int:
    clientes = _cargar()
    fuera = [h for h, c in clientes.items() if c.get('nombre') == nombre]
    for h in fuera:
        clientes.pop(h)
    if fuera:
        _guardar(clientes)
    return len(fuera)


def listar() -> list:
    return [{'nombre': c.get('nombre'), 'scope': c.get('scope'), 'creado': c.get('creado'),
             'hash': h[:12]} for h, c in _cargar().items()]


def verificar(cabecera: str | None, op: str | None = None):
    """(ok, cliente_o_motivo). Distingue 'falta la cabecera' de 'la clave no
    sirve': son dos errores distintos para quien llama, y confundirlos manda a
    revisar el lugar equivocado."""
    clientes = _cargar()
    if not clientes:
        return True, None                      # auth apagada
    if not cabecera:
        return False, 'falta la cabecera Authorization'
    partes = cabecera.split(None, 1)
    if len(partes) != 2 or partes[0].lower() != 'bearer':
        return False, "el formato debe ser 'Authorization: Bearer <clave>'"
    entrante = _hash(partes[1].strip())
    for h, c in clientes.items():
        # compare_digest y no ==: una comparación normal filtra información por
        # el tiempo que tarda en fallar.
        if hmac.compare_digest(h, entrante):
            if c.get('scope') == 'read' and op is not None and op not in OPS_SOLO_LECTURA:
                return False, f"el cliente '{c.get('nombre')}' tiene scope 'read' y op='{op}' escribe"
            return True, c
    return False, 'clave desconocida o revocada'


def usuario_efectivo(cliente, on_behalf_of=None) -> str:
    """De quién es el directorio de trabajo de este job.

    Sin autenticación es 'local'. Con ella, el usuario sale de la CREDENCIAL y no
    del envelope -- salvo que quien llama sea un `service`, que sí puede declarar
    por quién actúa. Un cliente que no es service y manda `on_behalf_of` recibe un
    error: si se ignorara en silencio, su trabajo terminaría en un directorio
    distinto del que pidió y nadie sabría por qué.
    """
    if cliente is None:
        if on_behalf_of:
            raise PermissionError(
                'on_behalf_of requiere un cliente autenticado con scope="service"; '
                'con la autenticación apagada no hay forma de verificar quién lo pide')
        return 'local'
    if on_behalf_of:
        if cliente.get('scope') != 'service':
            raise PermissionError(
                f"el cliente '{cliente.get('nombre')}' tiene scope '{cliente.get('scope')}' y "
                f"no puede actuar en nombre de otro; hace falta scope='service'")
        return str(on_behalf_of)
    return str(cliente.get('nombre') or 'local')


def _cli(argv) -> int:
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0
    # El fallback al registro propio era SILENCIOSO, y esa es justo la forma que
    # tiene de fallar: se crea un cliente en un archivo que el servidor en marcha
    # no lee, y el cliente no existe para nadie -- sin un solo mensaje. Medido por
    # uso el 2026-08-25 corriendo la prueba E.3 del reporte de la rutina, que dejo
    # un registro paralelo al compartido sin avisar.
    if REGISTRO_PROPIO:
        print('AVISO: sin LADUM_AUTH_FILE. Se usa el registro PROPIO de este programa:\n'
              f'  {AUTH_FILE}\n'
              'Eso NO es el registro compartido del flujo: lo que des de alta aca no lo\n'
              've ningun servidor arrancado con LADUM_AUTH_FILE.\n', file=sys.stderr)
    cmd = argv[0]
    if cmd == 'add':
        if len(argv) < 2:
            print(f'uso: auth.py add "<nombre>" [--scope {"|".join(SCOPES)}]', file=sys.stderr)
            return 1
        uso = f'uso: auth.py add "<nombre>" [--scope {"|".join(SCOPES)}] [--reemplazar]'
        scope, reemplazar, resto = 'full', False, argv[2:]
        while resto:
            if resto[0] == '--scope':
                if len(resto) < 2:
                    print('--scope necesita un valor.', file=sys.stderr)
                    print(uso, file=sys.stderr)
                    return 1
                scope, resto = resto[1], resto[2:]
            elif resto[0] == '--reemplazar':
                reemplazar, resto = True, resto[1:]
            else:
                # Un argumento que no se entiende NO puede terminar en mas privilegio
                # del que se pidio. `add nombre service` daba scope=full en silencio:
                # el usuario pedia el scope mas estrecho y se llevaba el mas ancho, y
                # nada lo decia. Es D-006 en la herramienta que reparte credenciales.
                print(f'argumento no reconocido: {resto[0]!r}', file=sys.stderr)
                print(uso, file=sys.stderr)
                return 1
        previos = sum(1 for c in listar() if c['nombre'] == argv[1])
        try:
            clave = agregar(argv[1], scope, reemplazar)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1
        revocadas = (f"Se revocaron {previos} credencial(es) anterior(es) de '{argv[1]}'.\n"
                     if reemplazar and previos else '')
        print(f"{revocadas}Cliente '{argv[1]}' creado (scope={scope}).\n\n  {clave}\n\n"
              f"Guardala ahora: solo se muestra una vez (se almacena el hash).\n"
              f"Usar como:  Authorization: Bearer {clave}\n"
              f"O en los clientes propios:  LADUM_SERVICE_KEY={clave}")
        return 0
    if cmd == 'list':
        cs = listar()
        if not cs:
            print('(sin clientes registrados — la autenticación está APAGADA)')
            return 0
        for c in cs:
            print(f"{c['nombre']}\t{c['scope']}\t{c['creado']}\t(hash {c['hash']}...)")
        return 0
    if cmd == 'revoke':
        if len(argv) < 2:
            print('uso: auth.py revoke "<nombre>"', file=sys.stderr)
            return 1
        n = revocar(argv[1])
        print(f"{n} cliente(s) revocado(s)." + ('' if requerida() else
              ' No queda ninguno: la autenticación quedó APAGADA.'))
        return 0
    print(f"comando '{cmd}' desconocido; usar add | list | revoke", file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(_cli(sys.argv[1:]))
