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
DB_SERVER    = os.getenv("DB_SERVER")
DB_NAME      = os.getenv("DB_NAME")
DB_USER      = os.getenv("DB_USER")
DB_PASS      = os.getenv("DB_PASS")

SCHEMA_CONTEXT = """
Base de datos SQL Server: OECE_DW
Datamart de analisis de riesgo de corrupcion/fraude en contrataciones publicas del Peru (OSCE/SEACE).
Contiene procesos de contratacion con indicadores de riesgo pre-calculados (postor unico, sancionados, concentracion economica, scores de integridad/transparencia/anomalia).

TABLAS (schema: dbo):

dbo.HECHOS_PROCESO - tabla de hechos principal, un registro por proceso de contratacion
  ocid (varchar) - identificador unico del proceso, clave para unir con SCORES_RIESGO
  entidad_key, proveedor_key, proc_key, geo_key (bigint) - llaves a las dimensiones
  fecha_convocatoria_key, fecha_adjudicacion_key (bigint) - llaves a DIM_TIEMPO
  n_oferentes (bigint) - cantidad de postores que participaron
  monto_adjudicado, valor_referencial (float) - montos en soles
  brecha_adj_ref (float) - diferencia entre monto adjudicado y valor referencial
  n_awards, n_proveedores_distintos, n_miembros_consorcio (bigint)
  tiempo_decision (bigint) - dias entre convocatoria y adjudicacion
  es_postor_unico, es_postor_unico_competitivo (bit) - solo se presento un postor
  es_directo (bit) - contratacion directa (sin concurso abierto)
  ganador_sancionado_historial, ganador_reincidente, sancionado_post_adjudicacion (bit)
  concentracion_economica_ent_prov (float) - que tan concentrados estan los contratos de esa entidad en ese proveedor
  dependencia_prov_ent (float) - dependencia del proveedor respecto a esa entidad
  tasa_exito_ganador (float) - tasa historica de exito del proveedor ganador
  n_postulaciones_par, n_adj_par_anio, ganador_recurrente_entidad, ganador_recurrente_global (bigint)

dbo.SCORES_RIESGO - scores de riesgo calculados por proceso (1 a 1 con HECHOS_PROCESO via ocid)
  ocid (varchar) - unir con HECHOS_PROCESO.ocid
  b_postor_unico, b_directo, b_sancionado, b_reincidente, b_concentracion,
  b_dependencia, b_brecha, b_tiempo, b_tasa_exito, b_fraccionamiento (float) - banderas/subscores individuales de riesgo (0-1)
  t_proveedor, t_valor_ref, t_monto, t_metodo, t_categoria, t_tenderers,
  t_periodo_ofertas, t_contrato_firmado, t_periodo_consultas (bigint) - flags de alertas puntuales
  score_integridad (float) - score general de integridad del proceso
  score_transparencia (float) - score de transparencia
  score_anomalia (float) - score de anomalia/riesgo de fraude (mientras mas alto, mas riesgo)

dbo.DIM_ENTIDAD - entidades contratantes (compradores)
  entidad_key (bigint), buyer_id (varchar), buyer_name (varchar)

dbo.DIM_PROVEEDOR - proveedores
  proveedor_key (bigint), proveedor_id (varchar), proveedor_nombre (varchar)
  es_consorcio (bit), prov_origen (varchar)
  prov_departamento, prov_provincia, prov_distrito (varchar) - ubicacion del proveedor

dbo.DIM_GEOGRAFIA - geografia de la entidad/proceso
  geo_key (bigint), prov_departamento, prov_provincia, prov_distrito (varchar)

dbo.DIM_PROCEDIMIENTO - metodo y categoria de contratacion
  proc_key (bigint), metodo (varchar), metodo_detalle (varchar), categoria (varchar)

dbo.DIM_TIEMPO - dimension de tiempo
  fecha_key (bigint), fecha (datetime), anio (int), mes (int), trimestre (int), nombre_mes (varchar)

RELACIONES CLAVE (no hay foreign keys formales en la BD, unir por estas columnas):
- HECHOS_PROCESO.entidad_key -> DIM_ENTIDAD.entidad_key
- HECHOS_PROCESO.proveedor_key -> DIM_PROVEEDOR.proveedor_key
- HECHOS_PROCESO.proc_key -> DIM_PROCEDIMIENTO.proc_key
- HECHOS_PROCESO.geo_key -> DIM_GEOGRAFIA.geo_key
- HECHOS_PROCESO.fecha_convocatoria_key -> DIM_TIEMPO.fecha_key
- HECHOS_PROCESO.fecha_adjudicacion_key -> DIM_TIEMPO.fecha_key
- HECHOS_PROCESO.ocid -> SCORES_RIESGO.ocid

NOTA IMPORTANTE: si la pregunta es sobre riesgo, fraude, corrupcion, anomalias, postor unico,
contratacion directa, proveedores sancionados o concentracion economica, usa SCORES_RIESGO
(especialmente score_anomalia, score_integridad, score_transparencia) unida a HECHOS_PROCESO via ocid.
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
        cols  = top_al_final.group(2)
        resto = top_al_final.group(3)
        order = top_al_final.group(4)
        n     = top_al_final.group(5)
        sql   = f"SELECT TOP {n} {cols} FROM{resto}{order}"
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
   dpr=DIM_PROCEDIMIENTO, dg=DIM_GEOGRAFIA, dt=DIM_TIEMPO
4. Schema dbo. siempre antes del nombre de tabla
5. LIMIT no existe en SQL Server, usa TOP
6. Para preguntas de riesgo/fraude/anomalia, une hp con sr via ocid
7. Si no puedes responder: SELECT 'No tengo datos para esa consulta' AS mensaje"""

    messages = [{"role": "system", "content": system_prompt}]
    if sql_con_error:
        messages += [
            {"role": "user",      "content": pregunta},
            {"role": "assistant", "content": sql_con_error},
            {"role": "user",      "content": "Ese SQL dio error. Corrígelo y genera uno nuevo."},
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
                "content": """Eres un analista senior de inteligencia de negocios especializado en riesgo de corrupcion en contrataciones publicas del Peru (OSCE/SEACE).

Redacta respuestas en forma de analisis narrativo ejecutivo, fluido y profesional en espanol.

ESTILO:
- Escribe en prosa continua, sin etiquetas como "Analisis:", "Contexto:", "Hallazgos:"
- Responde directamente con datos concretos: nombres, montos exactos, fechas, porcentajes
- Si hay multiples elementos destacados, usa viñetas breves SIN encabezados previos
- Cuando detectes algo inusual (concentracion, montos atipicos, contratacion directa), menciónalo
- Si los datos incluyen score_anomalia, score_integridad o score_transparencia, explica que significan en terminos de riesgo (score de anomalia alto = mayor riesgo de irregularidad, no necesariamente fraude comprobado)
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
    df  = None
    ultimo_error = None

    for intento in range(3):
        try:
            sql_raw = pregunta_a_sql(pregunta, sql if intento > 0 else None)
            sql     = limpiar_sql(sql_raw)
            df      = run_query(sql)
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
