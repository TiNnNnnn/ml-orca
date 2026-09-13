"""SQLGlot scope bindings to pre-run catalog rows; never executes or rewrites SQL in PG."""

from enum import Enum
import math

import sqlglot
from sqlglot import exp
from sqlglot.dialects import Dialect
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import MappingSchema
from sqlglot.tokens import TokenType

from ml_orca.encoding.rule_policy_encoding import category, flag, number


def search_path_names(setting):
    if not isinstance(setting, str):
        raise ValueError('missing pre-run search_path')
    tokens = Dialect.get_or_raise('postgres').tokenize(setting)
    if not tokens or len(tokens) % 2 == 0:
        raise ValueError('invalid search_path')
    names = []
    for i, token in enumerate(tokens):
        if i % 2:
            if token.token_type != TokenType.COMMA:
                raise ValueError('invalid search_path separator')
        else:
            names.append(token.text if token.token_type == TokenType.IDENTIFIER else token.text.lower())
    return names


def query_context(sql, catalog):
    """Return explicit unavailability, not a guessed table/column or zero stats."""
    try:
        return bind_query(sql, catalog)
    except (sqlglot.errors.SqlglotError, ValueError, KeyError, TypeError) as error:
        return {'sequence': [('sql_binding_unavailable', None)], 'sources': [], 'columns': [], 'source_references': [],
                'complete': False, 'errors': [str(error)], 'sqlglot_version': sqlglot.__version__}


def bind_query(sql, catalog):
    statements = sqlglot.parse(sql, read='postgres') if isinstance(sql, str) else []
    if (len(statements) != 1 or not isinstance(statements[0], exp.Query)
            or statements[0].find(exp.Into, exp.DML)):
        raise ValueError('require one read-only query')
    tree = normalize_identifiers(statements[0], dialect='postgres')
    if any(cte.args.get('recursive') for cte in tree.find_all(exp.With)):
        raise ValueError('recursive CTE bindings need explicit recursion semantics')
    relations = catalog['relations']
    by_name = {(r['schema'], r['name']): i for i, r in enumerate(relations)}
    if len(by_name) != len(relations):
        raise ValueError('duplicate catalog relation name')
    columns = {}
    for i, relation in enumerate(relations):
        columns[i] = {c['name']: c['position'] for c in catalog['columns'] if c['relation_oid'] == relation['oid']}

    def resolve(table):
        if not isinstance(table.this, exp.Identifier):
            raise ValueError('table function is not a catalog relation')
        if table.catalog and table.catalog != catalog.get('database'):
            raise ValueError('cross-database reference not in catalog snapshot')
        if table.db:
            if (table.db, table.name) not in by_name:
                raise ValueError('qualified relation missing from catalog')
            return by_name[table.db, table.name]
        matches = {schema: index for (schema, name), index in by_name.items() if name == table.name}
        dynamic = False
        for schema in search_path_names(catalog.get('settings', {}).get('search_path')):
            if schema in ('$user', 'pg_temp'):
                dynamic = True
            elif schema in matches:
                if dynamic and any(candidate != schema for candidate in matches):
                    raise ValueError('dynamic search_path prefix is ambiguous without session identity')
                return matches[schema]
        raise ValueError('relation not resolved by captured search_path')

    # CTE/derived sources are Scope objects, not catalog tables with the same name.
    for scope in traverse_scope(tree):
        for _, source in scope.selected_sources.values():
            if isinstance(source, exp.Table):
                relation = relations[resolve(source)]
                source.set('db', exp.to_identifier(relation['schema'], quoted=True))
                source.set('catalog', None)
    schema = {}
    for (namespace, name), index in by_name.items():
        schema.setdefault(namespace, {})[name] = {column: 'UNKNOWN' for column in columns[index]}
    tree = qualify(tree, dialect='postgres', schema=MappingSchema(schema, dialect='postgres', normalize=False),
                   infer_schema=False, validate_qualify_columns=True)
    scopes = traverse_scope(tree)
    if not scopes:
        raise ValueError('query has no supported scope')
    scope_ids = {id(scope): i for i, scope in enumerate(scopes)}
    owners = {id(scope.expression): scope for scope in scopes}
    source_ids, sources = {}, []
    for scope in scopes:
        for name, (_, source) in scope.selected_sources.items():
            record = {'scope': scope_ids[id(scope)], 'source': len(sources)}
            if isinstance(source, exp.Table):
                record.update(kind='relation', relation=resolve(source))
            elif isinstance(source, Scope) and id(source) in scope_ids:
                record.update(kind='derived', output_scope=scope_ids[id(source)])
            else:
                raise ValueError('unsupported source binding')
            source_ids[id(scope), name] = len(sources)
            sources.append(record)
    out, bindings, references = [], [], []

    def source_reference(record):
        references.append({'token': len(out), 'source': record['source']})
        number(out, 'source_ref', record['source'])
        category(out, 'source_kind', record['kind'])

    def source_for(scope, name):
        while scope is not None:
            if (id(scope), name) in source_ids:
                return sources[source_ids[id(scope), name]], scope.selected_sources[name][1]
            if not (scope.is_subquery or scope.is_union or scope.is_correlated_subquery):
                break
            scope = scope.parent
        raise ValueError('column source is outside its legal scope')

    def text_bytes(field, text):
        raw = text.encode('utf-8')
        number(out, field + ':length', len(raw))
        for byte in raw:
            number(out, field + ':byte', byte)

    def walk(node, scope):
        if not isinstance(node, exp.Expression):
            raise ValueError('unsupported non-expression AST child')
        scope = owners.get(id(node), scope)
        if id(node) in owners:
            number(out, 'sql_scope', scope_ids[id(scope)])
        category(out, 'sql_node', node.key)
        if isinstance(node, exp.Column):
            binding = {'token': len(out) - 1, 'scope': scope_ids[id(scope)]}
            if not node.table:
                names = scope.expression.named_selects
                if node.find_ancestor(exp.Order) is None or names.count(node.name) != 1:
                    raise ValueError('unresolved or ambiguous output column')
                binding.update(kind='output', output_scope=scope_ids[id(scope)], output_position=names.index(node.name))
                number(out, 'output_position', binding['output_position'])
            else:
                record, source = source_for(scope, node.table)
                binding.update(source=record['source'], kind=record['kind'])
                source_reference(record)
                if record['kind'] == 'relation':
                    relation = record['relation']
                    if node.name not in columns[relation]:
                        raise ValueError('column missing from catalog relation')
                    binding.update(relation=relation, column_position=columns[relation][node.name])
                    number(out, 'column_position', binding['column_position'])
                else:
                    names = source.expression.named_selects
                    if names.count(node.name) != 1:
                        raise ValueError('ambiguous derived output column')
                    binding.update(output_scope=record['output_scope'], output_position=names.index(node.name))
                    number(out, 'output_position', binding['output_position'])
            bindings.append(binding)
            return
        if isinstance(node, exp.Table):
            record, _ = source_for(scope, node.alias_or_name)
            source_reference(record)
        if isinstance(node, exp.Literal):
            flag(out, 'string_literal', node.is_string)
            text_bytes('literal', node.this)  # Exact value; no per-value vocabulary entry.
            if not node.is_string:
                value = float(node.this)
                number(out, 'numeric_literal', value if math.isfinite(value) else None, log=True)
            return
        if isinstance(node, (exp.ByteString, exp.HexString, exp.BitString, exp.RawString, exp.National)):
            text_bytes('literal', node.this)
            return
        if isinstance(node, exp.Identifier):
            raise ValueError('identifier role is not resolved')
        for key, value in sorted(node.args.items()):
            if key == 'alias' or isinstance(node, exp.Table) and key in ('this', 'db', 'catalog'):
                continue
            category(out, 'sql_arg', key)
            if isinstance(value, list):
                number(out, 'sql_list_length', len(value))
                for child in value:
                    walk(child, scope)
            elif isinstance(value, exp.Expression):
                walk(value, scope)
            elif type(value) is bool:
                flag(out, 'sql_bool', value)
            elif isinstance(value, Enum):
                category(out, 'sql_enum', value.name)
            elif value is None:
                out.append(('sql_none', None))
            elif isinstance(value, str):
                category(out, 'sql_attribute', value)
            elif type(value) in (int, float):
                number(out, 'sql_number', value)
            else:
                raise ValueError('unsupported SQL AST attribute')
        out.append(('sql_end_node', None))

    walk(tree, scopes[-1])
    return {'sequence': out, 'sources': sources, 'columns': bindings, 'source_references': references,
            'complete': True, 'errors': [],
            'scope_count': len(scopes), 'sqlglot_version': sqlglot.__version__,
            'scope': 'catalog_and_derived_output_binding_not_derived_cardinality_or_pg_semantic_proof'}
