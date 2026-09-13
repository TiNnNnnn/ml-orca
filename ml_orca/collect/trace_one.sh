#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 RULES_FILE SCHEMA_SQL QUERY_SQL OUTPUT_LOG" >&2
    exit 2
fi

RULES_FILE=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
SCHEMA_FILE=$2
QUERY_FILE=$3
OUTPUT_LOG=$4
PG_CONFIG=${PG_CONFIG:-$(command -v pg_config || true)}
PORT=${DSL_TRACE_PORT:-55440}
TRACE_DETAILS=off
if [[ ${DSL_TRACE_VERBOSE:-0} = 1 ]]; then
    TRACE_DETAILS=on
fi
case ${DSL_TRACE_XFORMS:-0} in
    1|on) TRACE_XFORMS=on ;;
    0|off) TRACE_XFORMS=off ;;
    *)
        echo "DSL trace failed: DSL_TRACE_XFORMS must be 0, 1, off, or on" >&2
        exit 1
        ;;
esac
TRACE_RESULTS=$TRACE_DETAILS
if [[ "$TRACE_XFORMS" = on ]]; then
    TRACE_RESULTS=on
fi
MAX_ALTERNATIVES=${DSL_TRACE_MAX_ALTERNATIVES:-0}
MAX_ALTERNATIVES_PER_RULE=${DSL_TRACE_MAX_ALTERNATIVES_PER_RULE:-0}

fail()
{
    echo "DSL trace failed: $*" >&2
    exit 1
}

[[ -n "$PG_CONFIG" && -x "$PG_CONFIG" ]] || fail "PG_CONFIG is not executable"
[[ -r "$RULES_FILE" ]] || fail "rules file is not readable: $RULES_FILE"
[[ -r "$SCHEMA_FILE" ]] || fail "schema file is not readable: $SCHEMA_FILE"
[[ -r "$QUERY_FILE" ]] || fail "query file is not readable: $QUERY_FILE"
[[ "$PORT" =~ ^[0-9]+$ ]] || fail "DSL_TRACE_PORT must be numeric"
[[ "$MAX_ALTERNATIVES" =~ ^[0-9]+$ ]] || fail "DSL_TRACE_MAX_ALTERNATIVES must be numeric"
[[ "$MAX_ALTERNATIVES_PER_RULE" =~ ^[0-9]+$ ]] || fail "DSL_TRACE_MAX_ALTERNATIVES_PER_RULE must be numeric"

PG_BINDIR=$($PG_CONFIG --bindir)
TRACE_ROOT=$(mktemp -d /tmp/pgorca-dsl-trace.XXXXXX)
DATA_DIR=$TRACE_ROOT/data
SOCKET_DIR=$TRACE_ROOT/socket
SERVER_LOG=$TRACE_ROOT/postgresql.log
RUN_SQL=$TRACE_ROOT/run.sql
SERVER_STARTED=0
mkdir -p "$SOCKET_DIR" "$(dirname "$OUTPUT_LOG")"

cleanup()
{
    if [[ $SERVER_STARTED -eq 1 ]]; then
        "$PG_BINDIR/pg_ctl" -D "$DATA_DIR" stop -m fast >/dev/null 2>&1 || true
    fi
    if [[ ${DSL_TRACE_KEEP_TMP:-0} = 1 ]]; then
        echo "trace workspace preserved at $TRACE_ROOT" >&2
    else
        rm -rf -- "$TRACE_ROOT"
    fi
}
trap cleanup EXIT

"$PG_BINDIR/initdb" -D "$DATA_DIR" --no-locale --encoding=UTF8 --auth=trust >/dev/null
MONSOON_DSL_RULES="$RULES_FILE" \
    "$PG_BINDIR/pg_ctl" -D "$DATA_DIR" -l "$SERVER_LOG" \
    -o "-c listen_addresses='' -c logging_collector=off -k $SOCKET_DIR -p $PORT" \
    start >/dev/null || {
        echo "PostgreSQL startup log:" >&2
        tail -n 40 "$SERVER_LOG" >&2 || true
        fail "temporary PostgreSQL server did not start"
    }
SERVER_STARTED=1

PSQL=("$PG_BINDIR/psql" -X -v ON_ERROR_STOP=1 -h "$SOCKET_DIR" -p "$PORT" -d postgres)
"${PSQL[@]}" -q -c "CREATE EXTENSION pg_orca;"
"${PSQL[@]}" -q -f "$SCHEMA_FILE"

{
    echo "LOAD 'pg_orca';"
    echo "SET pg_orca.enable_orca=on;"
    echo "SET pg_orca.enable_dsl_rule=on;"
    echo "SET pg_orca.dsl_rule_max_alternatives=$MAX_ALTERNATIVES;"
    echo "SET pg_orca.dsl_rule_max_alternatives_per_rule=$MAX_ALTERNATIVES_PER_RULE;"
    echo "SET pg_orca.trace_dsl_rule=on;"
    echo "SET optimizer_print_xform=$TRACE_XFORMS;"
    echo "SET optimizer_print_xform_results=$TRACE_RESULTS;"
    echo "SET client_min_messages=log;"
    echo "EXPLAIN (COSTS OFF)"
    cat "$QUERY_FILE"
} >"$RUN_SQL"

"${PSQL[@]}" -q -f "$RUN_SQL" >"$OUTPUT_LOG" 2>&1
if ! grep -Fq "DSL_TRACE " "$OUTPUT_LOG"; then
    if [[ ${DSL_TRACE_ALLOW_EMPTY:-0} = 1 ]]; then
        echo "pgORCA trace contains no DSL application records" >&2
    else
        fail "no DSL_TRACE records were produced"
    fi
fi

echo "pgORCA trace written to $OUTPUT_LOG"
