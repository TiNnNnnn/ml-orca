#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 RULES_FILE SCHEMA_SQL QUERY_DIR OUTPUT_DIR" >&2
    exit 2
fi

RULES_FILE=$1
SCHEMA_FILE=$2
QUERY_DIR=$3
OUTPUT_DIR=$4
PG_CONFIG=${PG_CONFIG:-$(command -v pg_config || true)}
PORT=${DSL_TRACE_PORT:-55440}
STATEMENT_TIMEOUT=${DSL_TRACE_STATEMENT_TIMEOUT:-60000}
MAX_ALTERNATIVES=${DSL_TRACE_MAX_ALTERNATIVES:-0}
MAX_ALTERNATIVES_PER_RULE=${DSL_TRACE_MAX_ALTERNATIVES_PER_RULE:-0}
DISABLE_XFORMS=${DSL_TRACE_DISABLE_XFORMS:-}
DSL_ENABLED=${DSL_TRACE_DSL_ENABLED:-on}
POLICY_PATH=${DSL_TRACE_POLICY_PATH:-}
DPHYPER_ENABLED=${DSL_TRACE_DPHYPER_ENABLED:-off}
DPHYPER_SHADOW=${DSL_TRACE_DPHYPER_SHADOW:-on}
DPHYPER_PAIR_BUDGET=${DSL_TRACE_DPHYPER_PAIR_BUDGET:-100}
DPHYPER_EDGE_BUDGET=${DSL_TRACE_DPHYPER_EDGE_BUDGET:-100000}
TRACE_DETAILS=off
COMPRESS_LOGS=${DSL_TRACE_COMPRESS_LOGS:-0}
CAPTURE_CONTEXT=${DSL_TRACE_CAPTURE_CONTEXT:-0}
MIN_FREE_KB=${DSL_TRACE_MIN_FREE_KB:-0}
if [[ ${DSL_TRACE_VERBOSE:-0} = 1 ]]; then
    TRACE_DETAILS=on
fi
case ${DSL_TRACE_XFORMS:-0} in
    1|on) TRACE_XFORMS=on ;;
    0|off) TRACE_XFORMS=off ;;
    *)
        echo "DSL corpus trace failed: DSL_TRACE_XFORMS must be 0, 1, off, or on" >&2
        exit 1
        ;;
esac
case ${DSL_TRACE_OPT_STATS:-0} in
    1|on) TRACE_OPT_STATS=on ;;
    0|off) TRACE_OPT_STATS=off ;;
    *)
        echo "DSL corpus trace failed: DSL_TRACE_OPT_STATS must be 0, 1, off, or on" >&2
        exit 1
        ;;
esac
case ${DSL_TRACE_PRINT_QUERY:-0} in
    1|on) TRACE_QUERY=on ;;
    0|off) TRACE_QUERY=off ;;
    *)
        echo "DSL corpus trace failed: DSL_TRACE_PRINT_QUERY must be 0, 1, off, or on" >&2
        exit 1
        ;;
esac
TRACE_RESULTS=$TRACE_DETAILS
if [[ "$TRACE_XFORMS" = on ]]; then
    TRACE_RESULTS=on
fi

fail()
{
    echo "DSL corpus trace failed: $*" >&2
    exit 1
}

[[ -n "$PG_CONFIG" && -x "$PG_CONFIG" ]] || fail "PG_CONFIG is not executable"
[[ -r "$RULES_FILE" ]] || fail "rules file is not readable: $RULES_FILE"
[[ -r "$SCHEMA_FILE" ]] || fail "schema file is not readable: $SCHEMA_FILE"
[[ -d "$QUERY_DIR" ]] || fail "query directory is not readable: $QUERY_DIR"
[[ "$PORT" =~ ^[0-9]+$ ]] || fail "DSL_TRACE_PORT must be numeric"
[[ "$COMPRESS_LOGS" = 0 || "$COMPRESS_LOGS" = 1 ]] || fail "invalid DSL_TRACE_COMPRESS_LOGS"
[[ "$CAPTURE_CONTEXT" = 0 || "$CAPTURE_CONTEXT" = 1 ]] || fail "invalid DSL_TRACE_CAPTURE_CONTEXT"
[[ "$MIN_FREE_KB" =~ ^[0-9]+$ ]] || fail "invalid DSL_TRACE_MIN_FREE_KB"
[[ "$STATEMENT_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || \
    fail "DSL_TRACE_STATEMENT_TIMEOUT must be a positive integer"
[[ "$MAX_ALTERNATIVES" =~ ^[0-9]+$ ]] || \
    fail "DSL_TRACE_MAX_ALTERNATIVES must be a non-negative integer"
[[ "$MAX_ALTERNATIVES_PER_RULE" =~ ^[0-9]+$ ]] || \
    fail "DSL_TRACE_MAX_ALTERNATIVES_PER_RULE must be a non-negative integer"
IFS=',' read -r -a DISABLED_XFORM_NAMES <<< "$DISABLE_XFORMS"
for XFORM_NAME in "${DISABLED_XFORM_NAMES[@]}"; do
    [[ -z "$XFORM_NAME" || "$XFORM_NAME" =~ ^CXform[A-Za-z0-9_]+$ ]] || \
        fail "DSL_TRACE_DISABLE_XFORMS contains an invalid xform name"
done
[[ "$DSL_ENABLED" = on || "$DSL_ENABLED" = off ]] || \
    fail "DSL_TRACE_DSL_ENABLED must be on or off"
if [[ -n "$POLICY_PATH" ]]; then
    [[ -r "$POLICY_PATH" ]] || fail "DSL_TRACE_POLICY_PATH is not readable"
    [[ "$POLICY_PATH" != *"'"* ]] || \
        fail "DSL_TRACE_POLICY_PATH must not contain a single quote"
fi
[[ "$DPHYPER_ENABLED" = on || "$DPHYPER_ENABLED" = off ]] || \
    fail "DSL_TRACE_DPHYPER_ENABLED must be on or off"
[[ "$DPHYPER_SHADOW" = on || "$DPHYPER_SHADOW" = off ]] || \
    fail "DSL_TRACE_DPHYPER_SHADOW must be on or off"
[[ "$DPHYPER_PAIR_BUDGET" =~ ^[1-9][0-9]*$ ]] || \
    fail "DSL_TRACE_DPHYPER_PAIR_BUDGET must be a positive integer"
[[ "$DPHYPER_EDGE_BUDGET" =~ ^[1-9][0-9]*$ ]] || \
    fail "DSL_TRACE_DPHYPER_EDGE_BUDGET must be a positive integer"

PG_BINDIR=$($PG_CONFIG --bindir)
TRACE_ROOT=$(mktemp -d /tmp/pgorca-dsl-corpus.XXXXXX)
DATA_DIR=$TRACE_ROOT/data
SOCKET_DIR=$TRACE_ROOT/socket
SERVER_LOG=$TRACE_ROOT/postgresql.log
RUN_SQL=$TRACE_ROOT/run.sql
STATUS_FILE=$OUTPUT_DIR/status.tsv
SERVER_STARTED=0
mkdir -p "$SOCKET_DIR" "$OUTPUT_DIR/logs"

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
SERVER_LOG_OPTIONS=""
if [[ "$COMPRESS_LOGS" = 1 ]]; then
    # The client archive retains all diagnostics. Do not duplicate unbounded
    # LOG-level rule traces in the temporary server's uncompressed stderr file.
    SERVER_LOG_OPTIONS="-c log_min_messages=fatal -c log_min_error_statement=panic"
fi
MONSOON_DSL_RULES="$RULES_FILE" \
    "$PG_BINDIR/pg_ctl" -D "$DATA_DIR" -l "$SERVER_LOG" \
    -o "-c listen_addresses='' -c logging_collector=off $SERVER_LOG_OPTIONS -k $SOCKET_DIR -p $PORT" \
    start >/dev/null || {
        tail -n 40 "$SERVER_LOG" >&2 || true
        fail "temporary PostgreSQL server did not start"
    }
SERVER_STARTED=1

PSQL=("$PG_BINDIR/psql" -X -v ON_ERROR_STOP=1 -h "$SOCKET_DIR" -p "$PORT" -d postgres)
"${PSQL[@]}" -q -c "CREATE EXTENSION pg_orca;"
"${PSQL[@]}" -q -f "$SCHEMA_FILE"
if [[ "$CAPTURE_CONTEXT" = 1 ]]; then
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    CONTEXT_ARGS=(--psql "$PG_BINDIR/psql" --socket "$SOCKET_DIR" --port "$PORT"
        --audit-bin "${DSL_TRACE_AUDIT_BIN:?context capture requires audit binary}"
        --rules "$RULES_FILE" --schema "$SCHEMA_FILE" --output "$OUTPUT_DIR/pre-workload-context.json")
    if [[ -n "$POLICY_PATH" ]]; then
        CONTEXT_ARGS+=(--policy "$POLICY_PATH")
    fi
    LEARNABLE_REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
    PYTHONPATH="$LEARNABLE_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        python3 -m ml_orca.collect.capture_rule_history_context "${CONTEXT_ARGS[@]}"
fi

: >"$STATUS_FILE"
shopt -s nullglob
QUERY_FILES=("$QUERY_DIR"/*.sql)
[[ ${#QUERY_FILES[@]} -gt 0 ]] || fail "query directory contains no .sql files"

for QUERY_FILE in "${QUERY_FILES[@]}"; do
    if [[ "$MIN_FREE_KB" -gt 0 ]]; then
        FREE_KB=$(df -Pk "$OUTPUT_DIR" | awk 'NR==2 {print $4}')
        [[ "$FREE_KB" =~ ^[0-9]+$ && "$FREE_KB" -ge "$MIN_FREE_KB" ]] || fail "insufficient free disk; remaining queries not run"
    fi
    STEM=$(basename "$QUERY_FILE" .sql)
    OUTPUT_LOG=$OUTPUT_DIR/logs/$STEM.log
    {
        echo "LOAD 'pg_orca';"
        echo "SET pg_orca.enable_orca=on;"
        echo "SET pg_orca.enable_dsl_rule=$DSL_ENABLED;"
        if [[ -n "$POLICY_PATH" ]]; then
            echo "SET pg_orca.dsl_rule_policy_path='$POLICY_PATH';"
        else
            echo "RESET pg_orca.dsl_rule_policy_path;"
        fi
        echo "SET pg_orca.enable_dphyper=$DPHYPER_ENABLED;"
        echo "SET pg_orca.dphyper_shadow=$DPHYPER_SHADOW;"
        echo "SET pg_orca.dphyper_pair_budget=$DPHYPER_PAIR_BUDGET;"
        echo "SET pg_orca.dphyper_edge_budget=$DPHYPER_EDGE_BUDGET;"
        echo "SET pg_orca.trace_dsl_rule=on;"
        echo "SET pg_orca.dsl_rule_max_alternatives=$MAX_ALTERNATIVES;"
        echo "SET pg_orca.dsl_rule_max_alternatives_per_rule=$MAX_ALTERNATIVES_PER_RULE;"
        echo "SET optimizer_print_xform=$TRACE_XFORMS;"
        echo "SET optimizer_print_xform_results=$TRACE_RESULTS;"
        echo "SET optimizer_print_optimization_stats=$TRACE_OPT_STATS;"
        echo "SET optimizer_print_query=$TRACE_QUERY;"
        echo "SET client_min_messages=log;"
        echo "SET statement_timeout='${STATEMENT_TIMEOUT}ms';"
        echo "SET optimizer_enable_query_parameter=on;"
        echo "SET plan_cache_mode=force_generic_plan;"
        for XFORM_NAME in "${DISABLED_XFORM_NAMES[@]}"; do
            if [[ -n "$XFORM_NAME" ]]; then
                echo "DO \$dsl\$ BEGIN PERFORM disable_xform('$XFORM_NAME'); END \$dsl\$;"
            fi
        done
        cat "$QUERY_FILE"
    } >"$RUN_SQL"

    if [[ "$COMPRESS_LOGS" = 1 ]]; then
        set +e
        "${PSQL[@]}" -q -f "$RUN_SQL" 2>&1 | gzip -c >"$OUTPUT_LOG.gz"
        PIPE_RESULTS=("${PIPESTATUS[@]}")
        set -e
        [[ "${PIPE_RESULTS[1]}" = 0 ]] || fail "query trace compression failed"
        RETURN_CODE=${PIPE_RESULTS[0]}
    elif "${PSQL[@]}" -q -f "$RUN_SQL" >"$OUTPUT_LOG" 2>&1; then
        RETURN_CODE=0
    else
        RETURN_CODE=$?
    fi
    printf '%s\t%s\n' "$STEM" "$RETURN_CODE" >>"$STATUS_FILE"
done

echo "DSL corpus traces written to $OUTPUT_DIR"
