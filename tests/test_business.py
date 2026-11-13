import copy
import json
import tempfile
import unittest
from pathlib import Path


def load(name):
    return json.loads((Path(__file__).resolve().parents[1] / 'fixtures' / name).read_text(encoding='utf-8'))


from policy_replay_book.decisions import validate_decisions, factual_summary


class BusinessTests(unittest.TestCase):
    def setUp(self):
        self.document = load('study.json')

    def test_factual_not_policy(self):
        result = factual_summary(self.document)
        self.assertEqual(sum(group['decisions'] for group in result['actions'].values()), 480)
        self.assertGreater(result['pending'], 0)

    def test_zero_probability_action(self):
        rows = copy.deepcopy(self.document['decisions'][:1])
        rows[0]['propensity'] = {'direct': 0, 'clarify': 1, 'human': 0}
        rows[0]['action'] = 'direct'
        with self.assertRaises(ValueError):
            validate_decisions(rows)

    def test_duplicate_decision(self):
        row = self.document['decisions'][0]
        with self.assertRaises(ValueError):
            validate_decisions([row, row])

    def test_bad_sequence(self):
        rows = copy.deepcopy(self.document['decisions'][:1])
        rows[0]['step'] = 2
        with self.assertRaises(ValueError):
            validate_decisions(rows)

    def test_pending_not_zero(self):
        document = copy.deepcopy(self.document)
        document['cutoff'] = -1
        result = factual_summary(document)
        self.assertEqual(result['pending'], len(document['decisions']))
        self.assertTrue(all(group['observed_mean'] is None for group in result['actions'].values()))
