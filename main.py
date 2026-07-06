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
Base de datos SQL Server: OECE_DW_1
Contiene datos de contrataciones publicas del Peru (OSCE/SEACE) 2022-2025.

TABLAS PRINCIPALES (schema: dw):

dw.FactContrato - tabla de hechos principal
  ContratoKey, ClaveContrato, Ocid, IdContrato, IdAdjudicacion, IdLicitacion
  EntidadKey, UbicacionEntidadKey, MonedaKey, CategoriaKey, MetodoKey
  FechaFirmaKey, FechaInicioKey, FechaFinKey
  TituloContrato, DescripcionContrato
  FechaFirma, FechaInicio, FechaFin, DuracionDias
  MontoContrato (decimal), MontoFinal (decimal), MontoLicitacion (decimal)
  MonedaContrato, NombreMonedaContrato
  NombreEntidadOriginal, NombreEntidadEstandar, RucEntidad
  MetodoContratacion, DetalleMetodoContratacion
  CategoriaPrincipal
  AnioArchivo, MesArchivo

dw.DimEntidad - entidades contratantes
  EntidadKey, ClaveOrganizacion, Ruc, NombreOriginal, NombreEstandar
  TipoOrganizacion, Departamento, Region, Localidad, Pais
  EsEntidad (bit), EsProveedor (bit)

dw.DimProveedor - proveedores
  ProveedorKey, ClaveOrganizacion, Ruc, NombreOriginal, NombreEstandar
  TipoOrganizacion, Departamento, Region, Localidad

dw.DimFecha - dimension tiempo
  FechaKey, Fecha, Anio, Semestre, Trimestre, MesNumero, MesNombre
  AnioMes, Dia, EsFinSemana

dw.DimUbicacion - geografia
  UbicacionKey, ClaveUbicacion, Pais, Departamento, Region, Localidad

dw.DimCategoria - categorias de contratacion
  CategoriaKey, ClaveCategoria, CategoriaPrincipal

dw.DimMetodoContratacion - metodos de contratacion
  MetodoKey, ClaveMetodo, MetodoContratacion, DetalleMetodoContratacion

dw.DimMoneda - monedas
  MonedaKey, ClaveMoneda, CodigoMoneda, NombreMoneda

dw.BridgeContratoProveedor - relacion contrato-proveedor
  ContratoKey, ProveedorKey, RucProveedor, NombreProveedorOriginal, NombreProveedorEstandar

dw.vwContratoProveedorBI - vista BI con datos de proveedor por contrato
  ContratoKey, ProveedorKey, RucProveedor, NombreProveedorEstandar
  CantidadProveedores, PesoProveedor, MontoContrato, MontoProrrateado

dw.ValorMoneda - tipo de cambio a Soles (PEN) por moneda
  MonedaKey, NombreMoneda, TipodeCambio

NOTA FRAUDE: Si existe la tabla dw.FactFraude, tiene columnas:
  ContratoKey, ScoreFraude (0.0-1.0), EsFraude (bit), MotivosRiesgo (nvarchar)
  Unirla con FactContrato via ContratoKey para consultas de riesgo.

RELACIONES CLAVE:
- FactContrato.EntidadKey -> DimEntidad.EntidadKey
- FactContrato.FechaFirmaKey -> DimFecha.FechaKey
- FactContrato.CategoriaKey -> DimCategoria.CategoriaKey
- FactContrato.MetodoKey -> DimMetodoContratacion.MetodoKey
- FactContrato.MonedaKey -> DimMoneda.MonedaKey
- FactContrato.MonedaKey -> ValorMoneda.MonedaKey (para tipo de cambio)
- FactContrato.UbicacionEntidadKey -> DimUbicacion.UbicacionKey
- BridgeContratoProveedor.ContratoKey -> FactContrato.ContratoKey
- BridgeContratoProveedor.ProveedorKey -> DimProveedor.ProveedorKey
  (FactContrato NO tiene ProveedorKey directo; el vinculo a proveedor
  SIEMPRE pasa por BridgeContratoProveedor)

═══════════════════════════════════════════════════════════════
REGLAS DE NEGOCIO OBLIGATORIAS (replican EXACTO el reporte Power BI
oficial del proyecto - aplican SIEMPRE, el usuario no necesita pedirlas):
═══════════════════════════════════════════════════════════════

1. MONTO EN SOLES (conversion de moneda obligatoria):
   El "monto final" que se reporta SIEMPRE es MontoContrato convertido a
   soles, NUNCA la columna MontoFinal (esa columna no se usa en el reporte
   oficial). Formula exacta:
       fc.MontoContrato * vm.TipodeCambio
   Requiere JOIN: INNER JOIN dw.ValorMoneda vm ON fc.MonedaKey = vm.MonedaKey
   Si una pregunta pide "monto", "monto total", "monto final" -> usa SIEMPRE
   esta formula convertida, nunca sumes MontoContrato o MontoFinal sin convertir.

2. SOLO CONTRATOS VALIDOS:
   Un contrato es "valido" solo si su fecha de fin ya paso respecto a hoy.
   Agrega SIEMPRE en el WHERE:
       AND fc.FechaFin < CAST(GETDATE() AS DATE)
   (Este resultado cambia dia a dia porque depende de la fecha actual,
   es el comportamiento correcto y esperado, igual que en Power BI.)

3. SOLO CONTRATOS CON PROVEEDOR VINCULADO ("Aplica"):
   Un contrato solo cuenta si tiene un proveedor vinculado en
   BridgeContratoProveedor. Agrega SIEMPRE:
       INNER JOIN dw.BridgeContratoProveedor bcp ON fc.ContratoKey = bcp.ContratoKey
       INNER JOIN dw.DimProveedor dp ON bcp.ProveedorKey = dp.ProveedorKey
   (el INNER JOIN ya excluye automaticamente los contratos sin proveedor)

EJEMPLO DE QUERY CORRECTO COMPLETO:
SELECT
    SUM(fc.MontoContrato * vm.TipodeCambio) AS MontoTotal,
    COUNT(fc.ContratoKey) AS CantidadContratos
FROM dw.FactContrato fc
INNER JOIN dw.DimFecha dfe ON fc.FechaFirmaKey = dfe.FechaKey
INNER JOIN dw.ValorMoneda vm ON fc.MonedaKey = vm.MonedaKey
INNER JOIN dw.BridgeContratoProveedor bcp ON fc.ContratoKey = bcp.ContratoKey
INNER JOIN dw.DimProveedor dp ON bcp.ProveedorKey = dp.ProveedorKey
WHERE fc.NombreEntidadOriginal = 'NOMBRE EXACTO DE LA ENTIDAD'
    AND dfe.Anio = 2022
    AND fc.FechaFin < CAST(GETDATE() AS DATE)
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
3. Alias fijos: fc=FactContrato, de=DimEntidad, dp=DimProveedor, dfe=DimFecha,
   du=DimUbicacion, dca=DimCategoria, dm=DimMetodoContratacion, dmo=DimMoneda,
   bcp=BridgeContratoProveedor, vm=ValorMoneda
4. Schema dw. siempre antes del nombre de tabla
5. LIMIT no existe en SQL Server, usa TOP
6. Si no puedes responder: SELECT 'No tengo datos para esa consulta' AS mensaje
7. OBLIGATORIO en TODA consulta que involucre montos, conteo de contratos,
   o cualquier metrica agregada de FactContrato: aplica las 3 reglas de negocio
   de la seccion "REGLAS DE NEGOCIO OBLIGATORIAS" de arriba (conversion a soles
   via ValorMoneda, filtro de contrato valido por fecha, join a proveedor).
   Esto aplica incluso si el usuario no las menciona explicitamente en su pregunta.
   NUNCA sumes MontoContrato o MontoFinal sin multiplicar por TipodeCambio."""

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
