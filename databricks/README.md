# Databricks en Dat-IA Lakehouse

Esta carpeta contendrá la capa analítica del fork y se implementará de forma incremental sobre Databricks Free Edition.

## Estructura

- `bronze/`: ingestión y tablas raw en Delta.
- `silver/`: limpieza, tipado y reglas de calidad.
- `gold/`: marts y tablas semánticas consumidas por el agente.
- `sql/`: consultas SQL reutilizables y validaciones analíticas.
- `jobs/`: definiciones y documentación de orquestación.

## Principios

1. PostgreSQL se mantiene inicialmente como fuente operacional.
2. Databricks se usa como plataforma analítica; no reemplaza PostgreSQL por defecto.
3. Las credenciales nunca se versionan en Git.
4. No se agrega ninguna dependencia de Databricks hasta que exista un caso ejecutable y testeable.
5. Cada etapa debe conservar el funcionamiento actual de Dat-IA.

## Primer objetivo funcional

Construir el flujo:

```text
PostgreSQL / dataset Olist
        ↓
      Bronze
        ↓
      Silver
        ↓
       Gold
```

La integración del agente con la capa Gold se hará después de validar este pipeline de forma independiente.
