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
Base de datos SQL Server: OECE_DW
Contiene datos de contrataciones publicas del Peru (OSCE/SEACE), modelo estrella.
Todas las tablas estan en el schema dbo (NO usar schema dw, ese ya no existe).

TABLAS PRINCIPALES:

dbo.HECHOS_PROCESO - tabla de hechos principal (un registro = un proceso de contratacion)
    ocid (varchar, PK) - identificador OCDS del proceso, formato tipo 'ocds-dgv273-seacev3-2022-47-5'
    entidad_key, proveedor_key, proc_key, geo_key (bigint) - llaves foraneas a dimensiones
    fecha_convocatoria_key, fecha_adjudicacion_key (bigint) - llaves foraneas a DIM_TIEMPO
    n_oferentes (bigint)
    monto_adjudicado, valor_referencial, brecha_adj_ref (float)
    n_awards, n_proveedores_distintos, n_miembros_consorcio (bigint)
    tiempo_decision (bigint) - dias entre convocatoria y adjudicacion
    es_postor_unico, es_postor_unico_competitivo, es_directo (bit)
    ganador_sancionado_historial, ganador_reincidente, sancionado_post_adjudicacion (bit)
    concentracion_economica_ent_prov, dependencia_prov_ent, tasa_exito_ganador (float)
    n_postulaciones_par, n_adj_par_anio (bigint)
    ganador_recurrente_entidad, ganador_recurrente_global (bigint)

dbo.SCORES_RIESGO - scores de riesgo/anomalia por proceso (creada por el equipo)
    ocid (varchar) - MISMO FORMATO que HECHOS_PROCESO.ocid, se une DIRECTO sin transformar
    b_postor_unico, b_directo, b_sancionado, b_reincidente, b_concentracion,
    b_dependencia, b_brecha, b_tiempo, b_tasa_exito, b_fraccionamiento (float) - banderas/subscores individuales
    t_proveedor, t_valor_ref, t_monto, t_metodo, t_categoria, t_tenderers,
    t_periodo_ofertas, t_contrato_firmado, t_periodo_consultas (bigint) - variables de contexto/umbral
    score_integridad, score_transparencia, score_anomalia (float) - scores finales (mientras mas alto, mas riesgo/anomalia)

IMPORTANTE SOBRE EL JOIN HECHOS_PROCESO <-> SCORES_RIESGO:
    Se unen DIRECTAMENTE por ocid: HECHOS_PROCESO.ocid = SCORES_RIESGO.ocid
    NO hay que separar ni recortar el ocid con LEFT/CHARINDEX, ambas columnas ya tienen el mismo formato.
    No todos los procesos tienen score (es un LEFT JOIN si se quiere incluir procesos sin score calculado).

dbo.DIM_ENTIDAD - entidades contratantes (compradoras)
    entidad_key (bigint, PK)
    buyer_id, buyer_name (varchar)

dbo.DIM_PROVEEDOR - proveedores/postores
    proveedor_key (bigint, PK)
    proveedor_id, proveedor_nombre (varchar)
    es_consorcio (bit)
    prov_origen, prov_departamento, prov_provincia, prov_distrito (varchar) - ubicacion del proveedor

dbo.DIM_GEOGRAFIA - ubicacion geografica del proceso/entidad
    geo_key (bigint, PK)
    prov_departamento, prov_provincia, prov_distrito (varchar)

dbo.DIM_PROCEDIMIENTO - tipo/metodo de contratacion
    proc_key (bigint, PK)
    metodo, metodo_detalle, categoria (varchar)

dbo.DIM_TIEMPO - dimension de tiempo
    fecha_key (bigint, PK)
    fecha (datetime), anio, mes, trimestre (int)
    nombre_mes (varchar)

RELACIONES CLAVE:
    HECHOS_PROCESO.entidad_key -> DIM_ENTIDAD.entidad_key
    HECHOS_PROCESO.proveedor_key -> DIM_PROVEEDOR.proveedor_key
    HECHOS_PROCESO.proc_key -> DIM_PROCEDIMIENTO.proc_key
    HECHOS_PROCESO.geo_key -> DIM_GEOGRAFIA.geo_key
    HECHOS_PROCESO.fecha_convocatoria_key -> DIM_TIEMPO.fecha_key
    HECHOS_PROCESO.fecha_adjudicacion_key -> DIM_TIEMPO.fecha_key
    HECHOS_PROCESO.ocid -> SCORES_RIESGO.ocid (join directo, mismo formato)

NOTA MONEDA: No existe columna de moneda en HECHOS_PROCESO. Los montos (monto_adjudicado,
valor_referencial) se asumen en Soles (S/) salvo que el usuario indique lo contrario.
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
3. Alias fijos: hp=HECHOS_PROCESO, sr=SCORES_RIESGO, de=DIM_ENTIDAD, dp=DIM_PROVEEDOR,
   dg=DIM_GEOGRAFIA, dpr=DIM_PROCEDIMIENTO, dt=DIM_TIEMPO
4. Schema dbo. siempre antes del nombre de tabla (dbo.HECHOS_PROCESO, dbo.SCORES_RIESGO, etc.)
5. LIMIT no existe en SQL Server, usa TOP
6. Para unir HECHOS_PROCESO con SCORES_RIESGO usa: hp.ocid = sr.ocid (join directo, NUNCA recortar
   ni transformar el ocid, ambas columnas ya tienen el mismo formato)
7. Si no puedes responder: SELECT 'No tengo datos para esa consulta' AS mensaje"""

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
                "content": """Eres un analista senior de inteligencia de negocios especializado en contrataciones publicas del Peru (OSCE/SEACE).
Redacta respuestas en forma de analisis narrativo ejecutivo, fluido y profesional en espanol.

ESTILO:
- Escribe en prosa continua, sin etiquetas como "Analisis:", "Contexto:", "Hallazgos:"
- Responde directamente con datos concretos: nombres, montos exactos, fechas, porcentajes
- Si hay multiples elementos destacados, usa viñetas breves SIN encabezados previos
- Cuando detectes algo inusual (concentracion, montos atipicos, contratacion directa, score de anomalia alto), menciónalo
- Usa S/ para soles
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
