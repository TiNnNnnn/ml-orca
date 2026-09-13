"""Scope/binding invariants for prospective query features."""

from copy import deepcopy
import unittest

from ml_orca.encoding.query_policy_encoding import query_context, search_path_names


def catalog_fixture():
    return {'database': 'test', 'settings': {'search_path': '"$user", public'},
            'relations': [{'oid': '1', 'schema': 'public', 'name': 'p'},
                          {'oid': '2', 'schema': 'public', 'name': 'c'}],
            'columns': [{'relation_oid': oid, 'name': name, 'position': position}
                        for oid, names in [('1', ('id', 'value')), ('2', ('id', 'pid'))]
                        for position, name in enumerate(names, 1)]}


class QueryEncodingTest(unittest.TestCase):
    def encode(self, sql, catalog=None):
        result = query_context(sql, catalog or catalog_fixture())
        self.assertTrue(result['complete'], result['errors'])
        return result

    def test_aliases_and_literal_changes_have_distinct_roles(self):
        original = self.encode('SELECT a.id FROM p a WHERE a.value = 2 ORDER BY a.id')
        renamed = self.encode('SELECT z.id FROM p z WHERE z.value = 2 ORDER BY z.id')
        self.assertEqual(original, renamed)
        changed = self.encode('SELECT a.id FROM p a WHERE a.value = 4 ORDER BY a.id')
        self.assertNotEqual(original['sequence'], changed['sequence'])
        self.assertEqual(original['columns'], changed['columns'])
        self.assertEqual({(c['relation'], c['column_position']) for c in original['columns']}, {(0,1),(0,2)})
        for prefix in ('', 'E', 'B', 'X'):
            one = self.encode(f"SELECT {prefix}'01'")['sequence']
            two = self.encode(f"SELECT {prefix}'10'")['sequence']
            self.assertNotEqual(one, two)
            self.assertEqual([t for t, _ in one], [t for t, _ in two])

    def test_correlated_subquery_and_self_join_occurrences(self):
        result = self.encode('SELECT x.id FROM p x WHERE EXISTS (SELECT 1 FROM c y WHERE y.pid=x.id)')
        self.assertEqual(result['scope_count'], 2)
        self.assertTrue(any(c['scope'] == 0 and c['relation'] == 0 for c in result['columns']))
        self.assertTrue(any(c['scope'] == 0 and c['relation'] == 1 for c in result['columns']))
        joined = self.encode('SELECT a.id FROM p a JOIN p b ON a.id=b.id')
        self.assertEqual(len(joined['sources']), 2)
        self.assertEqual({s['relation'] for s in joined['sources']}, {0})
        self.assertEqual({c['source'] for c in joined['columns']}, {0,1})
        for reference in joined['source_references']:
            self.assertEqual(joined['sequence'][reference['token']], ('source_ref', reference['source']))
        self.assertEqual({r['source'] for r in joined['source_references']}, {0,1})

    def test_cte_and_derived_columns_are_not_fake_base_statistics(self):
        result = self.encode('WITH p AS (SELECT pid AS k FROM c) SELECT x.k FROM p x ORDER BY x.k')
        self.assertEqual({s['relation'] for s in result['sources'] if s['kind'] == 'relation'}, {1})
        derived = [c for c in result['columns'] if c['kind'] == 'derived']
        self.assertTrue(derived)
        self.assertTrue(all('relation' not in c and 'output_scope' in c for c in derived))
        nested = self.encode('SELECT s.k FROM (SELECT max(value) AS k FROM p) s')
        self.assertTrue(any(c['kind'] == 'derived' for c in nested['columns']))

    def test_union_window_star_and_output_alias(self):
        for sql in ('SELECT id AS k FROM p UNION ALL SELECT pid AS k FROM c ORDER BY k',
                    'SELECT row_number() OVER (ORDER BY id) AS n FROM p ORDER BY n',
                    'SELECT * FROM p', 'SELECT count(*) AS n FROM p',
                    'SELECT 1'):
            self.encode(sql)
        result = self.encode('SELECT * FROM p')
        self.assertEqual({c['column_position'] for c in result['columns']}, {1,2})

    def test_quoted_names_search_path_and_catalog_permutation(self):
        catalog = catalog_fixture()
        catalog['relations'][0].update(schema='Mixed', name='Parent')
        catalog['columns'][0]['name'] = 'ID'
        self.encode('SELECT "ID" FROM "Mixed"."Parent"', catalog)
        self.assertEqual(search_path_names('"Mixed", public'), ['Mixed', 'public'])
        self.assertFalse(query_context('SELECT id FROM "Mixed"."Parent"', catalog)['complete'])
        before = self.encode('SELECT p.id FROM p')
        reordered = catalog_fixture()
        reordered['relations'].reverse()
        after = self.encode('SELECT p.id FROM p', reordered)
        self.assertEqual(before['sequence'], after['sequence'])
        self.assertEqual(before['sources'][0]['relation'], 0)
        self.assertEqual(after['sources'][0]['relation'], 1)

    def test_unknown_ambiguous_or_illegal_sources_fail_closed(self):
        for sql in ('SELECT id FROM p JOIN c ON p.id=c.pid', 'SELECT z FROM p',
                    'SELECT id FROM missing', 'SELECT x.id FROM p', 'SELECT 1; SELECT 2',
                    'DELETE FROM p', 'SELECT id INTO x FROM p',
                    'WITH x AS (DELETE FROM p RETURNING id) SELECT 1',
                    'SELECT p.id FROM p JOIN (SELECT p.id) x ON true',
                    'WITH RECURSIVE x AS (SELECT 1 UNION ALL SELECT * FROM x) SELECT * FROM x'):
            result = query_context(sql, catalog_fixture())
            self.assertFalse(result['complete'], sql)
            self.assertEqual(result['columns'], [])
            self.assertTrue(result['errors'])
        ambiguous = catalog_fixture()
        ambiguous['relations'].append({'oid': '3', 'schema': 'possible_user', 'name': 'p'})
        ambiguous['columns'].append({'relation_oid': '3', 'name': 'id', 'position': 1})
        self.assertFalse(query_context('SELECT id FROM p', ambiguous)['complete'])
        ambiguous['settings']['search_path'] = 'public, possible_user'
        self.encode('SELECT id FROM p', ambiguous)

    def test_renamed_catalog_identifiers_do_not_become_embeddings(self):
        before = self.encode('SELECT p.id FROM p WHERE p.value=2')
        catalog = deepcopy(catalog_fixture())
        catalog['relations'][0]['name'] = 'renamed'
        catalog['columns'][0]['name'] = 'key'
        catalog['columns'][1]['name'] = 'data'
        after = self.encode('SELECT renamed.key FROM renamed WHERE renamed.data=2', catalog)
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
