from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pymssql
import pandas as pd
import re
import threading
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

# Schema verificado el 08/07/2026 contra OECE_DW via INFORMATION_SCHEMA.COLUMNS.
# Esta es la fuente de verdad real (reemplaza la version anterior con dw.FactContrato
# que correspondia a OECE_DW_1 y ya no aplica).
SCHEMA_CONTEXT = """
Base de datos SQL Server: OECE_DW
Contiene datos de contrataciones publicas del Peru (OSCE/SEACE) con analisis de riesgo/integridad.
Todas las tablas usan el schema "dbo".

dbo.HECHOS_PROCESO - tabla de hechos principal (grano: un proceso de contratacion = ocid)
    ocid (varchar, PK/clave del proceso)
    entidad_key (bigint, FK -> DIM_ENTIDAD.entidad_key)
    proveedor_key (bigint, FK -> DIM_PROVEEDOR.proveedor_key)
    proc_key (bigint, FK -> DIM_PROCEDIMIENTO.proc_key)
    geo_key (bigint)
    fecha_convocatoria_key (bigint, FK -> DIM_TIEMPO.fecha_key)
    fecha_adjudicacion_key (bigint, FK -> DIM_TIEMPO.fecha_key)
    n_oferentes (bigint) - numero de postores en el proceso
    monto_adjudicado (float), valor_referencial (float), brecha_adj_ref (float)
    n_awards (bigint), n_proveedores_distintos (bigint), n_miembros_consorcio (bigint)
    tiempo_decision (bigint) - dias entre convocatoria y adjudicacion
    es_postor_unico (bit), es_postor_unico_competitivo (bit), es_directo (bit)
    ganador_sancionado_historial (bit), ganador_reincidente (bit), sancionado_post_adjudicacion (bit)
    concentracion_economica_ent_prov (float), dependencia_prov_ent (float)
    tasa_exito_ganador (float)
    n_postulaciones_par (bigint), n_adj_par_anio (bigint)
    ganador_recurrente_entidad (bigint), ganador_recurrente_global (bigint)

dbo.SCORES_RIESGO - scores de riesgo/integridad por proceso (1 fila por ocid)
    ocid (varchar, FK -> HECHOS_PROCESO.ocid)
    b_postor_unico, b_directo, b_sancionado, b_reincidente, b_concentracion,
    b_dependencia, b_brecha, b_tiempo, b_tasa_exito, b_fraccionamiento (float)
        -> banderas/subindicadores normalizados de cada factor de riesgo individual
    t_proveedor, t_valor_ref, t_monto, t_metodo, t_categoria, t_tenderers,
    t_periodo_ofertas, t_contrato_firmado, t_periodo_consultas (bigint)
        -> variables de tiempo/conteo usadas para calcular los scores agregados
    score_integridad (float) - score compuesto de integridad del proceso
    score_transparencia (float) - score compuesto de transparencia del proceso
    score_anomalia (float) - score compuesto de anomalia estadistica del proceso

dbo.DIM_ENTIDAD
    entidad_key (bigint, PK), buyer_id (varchar), buyer_name (varchar)

dbo.DIM_PROVEEDOR
    proveedor_key (bigint, PK), proveedor_id (varchar), proveedor_nombre (varchar)
    es_consorcio (bit), prov_origen (varchar)
    prov_departamento (varchar), prov_provincia (varchar), prov_distrito (varchar)

dbo.DIM_PROCEDIMIENTO
    proc_key (bigint, PK), metodo (varchar), metodo_detalle (varchar), categoria (varchar)

dbo.DIM_TIEMPO
    fecha_key (bigint, PK), fecha (datetime), anio (int), mes (int)
    trimestre (int), nombre_mes (varchar)

RELACIONES CLAVE:
- HECHOS_PROCESO.ocid = SCORES_RIESGO.ocid
- HECHOS_PROCESO.entidad_key = DIM_ENTIDAD.entidad_key
- HECHOS_PROCESO.proveedor_key = DIM_PROVEEDOR.proveedor_key
- HECHOS_PROCESO.proc_key = DIM_PROCEDIMIENTO.proc_key
- HECHOS_PROCESO.fecha_convocatoria_key / fecha_adjudicacion_key = DIM_TIEMPO.fecha_key

REGLA DE NEGOCIO — RIESGO (MUY IMPORTANTE):
No existe una columna llamada "score de riesgo". Cuando te pidan procesos/proveedores/entidades
"con mas riesgo", "riesgosos", "mas peligrosos" o similar, el criterio correcto es:
    SCORES_RIESGO.score_integridad MAS BAJO (ORDER BY score_integridad ASC)
NUNCA ordenes por score_anomalia DESC pensando que eso es "riesgo" - score_anomalia mide
anomalia estadistica, no riesgo de integridad, y son conceptos distintos.
Si la pregunta menciona explicitamente "anomalia" usa score_anomalia DESC.
Si menciona explicitamente "transparencia" usa score_transparencia ASC (mas baja = menos transparente).
"""

# ── Conexion a Azure SQL (via pymssql / FreeTDS) ────────────────
# FIX: la conexión ya NO se comparte como variable global entre requests.
# FastAPI ejecuta endpoints sync en threads del threadpool, así que compartir
# un solo objeto pymssql.Connection entre threads concurrentes corrompe el
# socket y produce exactamente los errores 20003 / 20047 que viste.
# Ahora cada request abre su propia conexión y la cierra al terminar.
# Si más adelante quieres pooling real, usa algo como sqlalchemy con un
# QueuePool en vez de reabrir conexión por request.

_conn_lock = threading.Lock()


def get_connection():
    return pymssql.connect(
        server=DB_SERVER,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        as_dict=False,
        login_timeout=30,
        timeout=60,  # subido de 30 a 60: joins de 5 tablas sin WHERE pueden tardar
    )


def run_query(sql: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


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
3. Alias fijos: hp=HECHOS_PROCESO, sr=SCORES_RIESGO, de=DIM_ENTIDAD, dp=DIM_PROVEEDOR,
   dpr=DIM_PROCEDIMIENTO, dt=DIM_TIEMPO
4. Schema dbo. siempre antes del nombre de tabla (ej: dbo.HECHOS_PROCESO)
5. LIMIT no existe en SQL Server, usa TOP
6. Si no puedes responder: SELECT 'No tengo datos para esa consulta' AS mensaje
7. No existe una columna llamada "score de riesgo". Si te piden proveedores/entidades/
   procesos con mayor riesgo, el criterio de negocio es: SCORES_RIESGO.score_integridad
   MAS BAJO (ORDER BY score_integridad ASC), NO score_anomalia mas alto. Nunca ordenes
   por score_anomalia DESC pensando que eso es "riesgo" - son conceptos distintos.
8. Si la consulta cruza HECHOS_PROCESO con SCORES_RIESGO, siempre filtra o pon un TOP
   razonable antes del ORDER BY cuando no haya WHERE, para evitar ordenar el dataset
   completo en cada consulta."""

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
        conn = get_connection()
        conn.cursor().execute("SELECT 1")
        conn.close()
        return {"status": "ok", "db": "conectada"}
    except Exception as e:
        return {"status": "error", "db": str(e)}


@app.get("/schema-check")
def schema_check():
    """
    Endpoint de diagnostico temporal: devuelve el esquema REAL de la BD
    conectada ahora mismo, para comparar contra SCHEMA_CONTEXT.
    Bórralo cuando ya no lo necesites (no debería quedar expuesto en prod).
    """
    try:
        conn = get_connection()
        df = pd.read_sql(
            """
            SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
            """,
            conn,
        )
        conn.close()
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(500, f"No pude leer el schema: {e}")


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
