from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pymssql
import pandas as pd
import re
from groq import Groq
import os

app = FastAPI(title="Chatbot OSCE API")

# ── CORS (permite que Power BI llame a esta API) ───────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Config (usa variables de entorno en Railway, sin defaults) ──
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_SERVER = os.getenv("DB_SERVER")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

SCHEMA_CONTEXT = """
Base de datos SQL Server. Schema: dbo.
Contiene datos de contrataciones publicas del Peru (OSCE/SEACE), con analisis
de riesgo/anomalia. Esquema verificado via INFORMATION_SCHEMA.COLUMNS.

⚠️ IMPORTANTE: usa EXCLUSIVAMENTE las tablas y columnas listadas abajo, con el
schema "dbo." y estos nombres EXACTOS (respeta mayusculas/minusculas). NUNCA
inventes ni asumas nombres de tabla o columna que no esten en esta lista.

dbo.HECHOS_PROCESO - tabla de hechos principal (un registro por proceso de contratacion)
  ocid                        (varchar) identificador unico del proceso, PK logica
  entidad_key                 (bigint)  FK -> dbo.DIM_ENTIDAD.entidad_key
  proveedor_key               (bigint)  FK -> dbo.DIM_PROVEEDOR.proveedor_key
  proc_key                    (bigint)  FK -> dbo.DIM_PROCEDIMIENTO.proc_key
  geo_key                     (bigint)  FK -> dbo.DIM_GEOGRAFIA.geo_key
  fecha_convocatoria_key       (bigint)  FK -> dbo.DIM_TIEMPO.fecha_key (fecha de convocatoria)
  fecha_adjudicacion_key       (bigint)  FK -> dbo.DIM_TIEMPO.fecha_key (fecha de adjudicacion)
  n_oferentes                 (bigint)  cantidad de oferentes/postores
  monto_adjudicado            (float)   monto final adjudicado
  valor_referencial           (float)   valor referencial del proceso
  brecha_adj_ref               (float)   diferencia entre monto adjudicado y valor referencial
  n_awards                    (bigint)
  n_proveedores_distintos      (bigint)
  n_miembros_consorcio         (bigint)
  tiempo_decision              (bigint)  dias entre convocatoria y adjudicacion
  es_postor_unico              (bit)
  es_postor_unico_competitivo   (bit)
  es_directo                   (bit)     contratacion directa (sin concurso)
  ganador_sancionado_historial  (bit)
  ganador_reincidente           (bit)
  sancionado_post_adjudicacion  (bit)
  concentracion_economica_ent_prov (float)
  dependencia_prov_ent          (float)
  tasa_exito_ganador             (float)
  n_postulaciones_par           (bigint)
  n_adj_par_anio                (bigint)
  ganador_recurrente_entidad     (bigint)
  ganador_recurrente_global      (bigint)

dbo.DIM_ENTIDAD - entidades contratantes (compradoras)
  entidad_key    (bigint)  PK
  buyer_id       (varchar) codigo de la entidad
  buyer_name     (varchar) nombre de la entidad compradora -> usar SIEMPRE con LIKE + UPPER

dbo.DIM_PROVEEDOR - proveedores
  proveedor_key       (bigint)  PK
  proveedor_id         (varchar)
  proveedor_nombre     (varchar) nombre del proveedor -> usar SIEMPRE con LIKE + UPPER
  es_consorcio         (bit)
  prov_origen          (varchar)
  prov_departamento    (varchar) departamento de origen del PROVEEDOR (no del proceso)
  prov_provincia       (varchar)
  prov_distrito        (varchar)

dbo.DIM_GEOGRAFIA - ubicacion geografica del PROCESO/contrato
  geo_key             (bigint)  PK
  prov_departamento    (varchar) departamento donde ocurre el proceso
  prov_provincia       (varchar)
  prov_distrito        (varchar)

dbo.DIM_PROCEDIMIENTO - metodo/tipo de procedimiento de contratacion
  proc_key         (bigint)  PK
  metodo           (varchar) metodo de contratacion
  metodo_detalle    (varchar)
  categoria        (varchar) categoria del proceso/contratacion

dbo.DIM_TIEMPO - dimension tiempo (se usa DOS VECES desde HECHOS_PROCESO:
  una para fecha_convocatoria_key y otra para fecha_adjudicacion_key.
  Si la pregunta necesita ambas fechas, usa DOS alias distintos, ej:
  dt_conv y dt_adj, cada uno con su propio JOIN a dbo.DIM_TIEMPO)
  fecha_key    (bigint)  PK
  fecha        (datetime)
  anio         (int)
  mes          (int)
  trimestre    (int)
  nombre_mes   (varchar)

dbo.SCORES_RIESGO - scores de anomalia/riesgo por proceso (uno a uno con HECHOS_PROCESO)
  ocid                    (varchar) FK -> dbo.HECHOS_PROCESO.ocid (NO es bigint, es varchar)
  b_postor_unico          (float)   sub-score: postor unico
  b_directo               (float)   sub-score: contratacion directa
  b_sancionado            (float)   sub-score: proveedor sancionado
  b_reincidente           (float)   sub-score: ganador reincidente
  b_concentracion         (float)   sub-score: concentracion economica
  b_dependencia           (float)   sub-score: dependencia proveedor-entidad
  b_brecha                (float)   sub-score: brecha monto adjudicado vs referencial
  b_tiempo                (float)   sub-score: tiempo de decision
  b_tasa_exito            (float)   sub-score: tasa de exito del ganador
  b_fraccionamiento       (float)   sub-score: fraccionamiento de compras
  t_proveedor, t_valor_ref, t_monto, t_metodo, t_categoria, t_tenderers,
  t_periodo_ofertas, t_contrato_firmado, t_periodo_consultas  (bigint) - umbrales/flags de cada sub-score
  score_integridad        (float)   score compuesto de integridad
  score_transparencia     (float)   score compuesto de transparencia
  score_anomalia          (float)   score compuesto de anomalia/riesgo (el mas usado para "riesgo mas alto")

RELACIONES CLAVE:
- dbo.HECHOS_PROCESO.entidad_key -> dbo.DIM_ENTIDAD.entidad_key
- dbo.HECHOS_PROCESO.proveedor_key -> dbo.DIM_PROVEEDOR.proveedor_key
- dbo.HECHOS_PROCESO.proc_key -> dbo.DIM_PROCEDIMIENTO.proc_key
- dbo.HECHOS_PROCESO.geo_key -> dbo.DIM_GEOGRAFIA.geo_key
- dbo.HECHOS_PROCESO.fecha_convocatoria_key -> dbo.DIM_TIEMPO.fecha_key
- dbo.HECHOS_PROCESO.fecha_adjudicacion_key -> dbo.DIM_TIEMPO.fecha_key
- dbo.HECHOS_PROCESO.ocid -> dbo.SCORES_RIESGO.ocid (join por texto, no por key numerica)

NOTA: no asumas ninguna otra tabla o columna fuera de esta lista (ej. no existe
FactContrato, DimFecha, NombreEstandar, MontoContrato, etc. de versiones previas).
"""

# ── Reglas para filtros de texto (evita 0 falsos por nombre incompleto) ──
REGLAS_FILTROS_TEXTO = """
REGLAS OBLIGATORIAS PARA FILTROS DE TEXTO (entidad, proveedor, buyer_name, proveedor_nombre):

1. NUNCA uses el operador "=" para comparar nombres de entidad o proveedor.
   Los nombres oficiales en la base de datos suelen incluir sufijos, siglas o
   variantes (ej: "AGENCIA DE PROMOCION DE LA INVERSION PRIVADA - PROINVERSION")
   que el usuario no escribe en su pregunta. Un "=" exacto casi siempre
   devuelve 0 filas aunque la entidad exista.

2. USA SIEMPRE "LIKE" con comodines en ambos extremos, sobre las palabras
   clave mas distintivas de la pregunta, nunca la frase completa:
   Correcto:   WHERE de.buyer_name LIKE '%PROMOCION%INVERSION%PRIVADA%'
   Incorrecto: WHERE de.buyer_name = 'Agencia de Promocion de la Inversion Privada'
   Incorrecto: WHERE de.buyer_name LIKE '%Agencia de Promocion de la Inversion Privada%'
   (la frase completa como comodin es tan fragil como el "=" si falta o sobra
   una palabra; usa solo 2-4 palabras clave separadas por %)
   Lo mismo aplica a dp.proveedor_nombre para filtros de proveedor.

3. Todos los filtros de texto deben ser insensibles a mayusculas/minusculas
   y usar UPPER() en ambos lados de la comparacion:
   WHERE UPPER(de.buyer_name) LIKE UPPER('%PROMOCION%INVERSION%PRIVADA%')

4. Si la pregunta del usuario nombra una entidad o proveedor, NO copies el
   texto del usuario tal cual dentro del LIKE. Extrae unicamente las 2-4
   palabras mas especificas y descarta palabras genericas como "agencia",
   "de", "la", "empresa", "publica"/"privada" si generan ambiguedad
   (usa solo las palabras que identifican de forma unica a la entidad,
   ej: "PROMOCION", "INVERSION", "PRIVADA" o "PROINVERSION").

5. Esta misma regla aplica a TODOS los filtros de texto sin excepcion:
   nombres de entidad, nombres de proveedor, categorias, departamentos,
   o cualquier campo tipo string. Nunca coincidencia exacta salvo que el
   usuario pida explicitamente un codigo o ID exacto (ej: RUC, codigo SEACE).
"""

# ── Conexion a Azure SQL (via pymssql / FreeTDS) ────────────────
_conn = None

def get_connection():
    global _conn
    try:
        if _conn:
            cur = _conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchall()
            return _conn
    except Exception:
        _conn = None

    _conn = pymssql.connect(
        server=DB_SERVER,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        as_dict=False,
        login_timeout=30,
        timeout=30,
    )
    return _conn

def run_query(sql: str) -> pd.DataFrame:
    return pd.read_sql(sql, get_connection())

# ── Limpieza SQL ───────────────────────────────────────────────
def limpiar_sql(sql: str) -> str:
    sql = re.sub(r"```sql|```", "", sql, flags=re.IGNORECASE).strip()

    top_al_final = re.search(
        r'(SELECT)\s+(.*?)\s+FROM(.*?)(ORDER BY.*?)\s+TOP\s+(\d+)\s*$',
        sql, re.IGNORECASE | re.DOTALL
    )
    if top_al_final:
        cols = top_al_final.group(2)
        resto = top_al_final.group(3)
        order = top_al_final.group(4)
        n = top_al_final.group(5)
        sql = f"SELECT TOP {n} {cols} FROM{resto}{order}"

    if not re.search(r'SELECT\s+TOP\s+\d+', sql, re.IGNORECASE):
        sql = re.sub(r'^SELECT\s+', 'SELECT TOP 100 ', sql, flags=re.IGNORECASE)

    return sql.strip()

# ── Groq: pregunta → SQL ───────────────────────────────────────
def pregunta_a_sql(pregunta: str, sql_con_error: str = None) -> str:
    client = Groq(api_key=GROQ_API_KEY)

    system_prompt = f"""Eres un experto en SQL Server. Convierte preguntas en lenguaje natural a SQL valido para SQL Server.

{SCHEMA_CONTEXT}

REGLAS CRITICAS:
1. Responde SOLO con SQL puro, sin markdown, sin explicaciones
2. TOP va SIEMPRE despues de SELECT: "SELECT TOP 100 col FROM tabla ORDER BY col DESC"
3. Alias fijos: hp=dbo.HECHOS_PROCESO, de=dbo.DIM_ENTIDAD, dp=dbo.DIM_PROVEEDOR, dt=dbo.DIM_TIEMPO
4. Schema dbo. siempre antes del nombre de tabla
5. LIMIT no existe en SQL Server, usa TOP
6. Si no puedes responder, o si necesitas una columna que no aparece en el
   SCHEMA_CONTEXT: SELECT 'No tengo datos para esa consulta' AS mensaje
   — NUNCA inventes un nombre de tabla o columna que no este en el schema

{REGLAS_FILTROS_TEXTO}"""

    messages = [{"role": "system", "content": system_prompt}]

    if sql_con_error:
        messages += [
            {"role": "user", "content": pregunta},
            {"role": "assistant", "content": sql_con_error},
            {"role": "user", "content": "Ese SQL dio error. Corrígelo y genera uno nuevo."},
        ]
    else:
        messages.append({"role": "user", "content": pregunta})

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.1,
        max_tokens=1000,
    )
    return r.choices[0].message.content.strip()

# ── Groq: datos → narrativa ────────────────────────────────────
def interpretar_resultado(pregunta: str, df: pd.DataFrame) -> str:
    client = Groq(api_key=GROQ_API_KEY)

    datos_str = df.head(20).to_string(index=False) if not df.empty else "Sin resultados"

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """Eres un analista senior de inteligencia de negocios especializado en contrataciones publicas del Peru (OSCE/SEACE 2022-2025).
Redacta respuestas en forma de analisis narrativo ejecutivo, fluido y profesional en espanol.

ESTILO:
- Escribe en prosa continua, sin etiquetas como "Analisis:", "Contexto:", "Hallazgos:"
- Responde directamente con datos concretos: nombres, montos exactos, fechas, porcentajes
- Si hay multiples elementos destacados, usa viñetas breves SIN encabezados previos
- Cuando detectes algo inusual (concentracion, montos atipicos, contratacion directa), menciónalo
- Usa S/ para soles y US$ para dolares
- Al final, en cursiva, sugiere una pregunta de profundizacion relevante
- Maximo 200 palabras

REGLA SOBRE RESULTADOS EN CERO:
- Si el resultado tiene 0 filas y la pregunta involucraba un nombre de entidad
  o proveedor, NO concluyas que "no hay actividad", que "es inusual" o que
  "hay una posible omision en la informacion". Esa conclusion suele ser falsa:
  lo mas probable es que el nombre de la entidad/proveedor no coincidio de
  forma exacta con el registrado en la base de datos.
- En ese caso, responde indicando que no se encontro una entidad o proveedor
  que coincida con esas palabras clave, y sugiere al usuario verificar el
  nombre exacto o intentar con una version mas corta (ej. solo el nombre
  principal, sin sufijos ni siglas).

PROHIBIDO: mencionar SQL, tablas, columnas, bases de datos."""
            },
            {
                "role": "user",
                "content": f"Pregunta: {pregunta}\n\nDatos ({len(df)} registros):\n{datos_str}"
            }
        ],
        temperature=0.4,
        max_tokens=600,
    )
    return r.choices[0].message.content.strip()

# ── Modelos de request/response ────────────────────────────────
class ChatRequest(BaseModel):
    pregunta: str

class ChatResponse(BaseModel):
    respuesta: str
    sql: str
    n_registros: int
    datos: list

# ── Endpoints ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "servicio": "Chatbot OSCE - Contrataciones Publicas Peru"}

@app.get("/health")
def health():
    try:
        get_connection().cursor().execute("SELECT 1")
        return {"status": "ok", "db": "conectada"}
    except Exception as e:
        return {"status": "error", "db": str(e)}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    pregunta = req.pregunta.strip()
    if not pregunta:
        raise HTTPException(400, "Pregunta vacía")

    sql = None
    df = None
    ultimo_error = None

    for intento in range(3):
        try:
            sql_raw = pregunta_a_sql(pregunta, sql if intento > 0 else None)
            sql = limpiar_sql(sql_raw)
            df = run_query(sql)
            break
        except Exception as e:
            ultimo_error = str(e)

    if df is None:
        raise HTTPException(500, f"No pude ejecutar la consulta: {ultimo_error}")

    respuesta = interpretar_resultado(pregunta, df)

    datos_json = []
    if not df.empty:
        datos_json = df.head(100).to_dict(orient="records")

    return ChatResponse(
        respuesta=respuesta,
        sql=sql,
        n_registros=len(df),
        datos=datos_json,
    )
