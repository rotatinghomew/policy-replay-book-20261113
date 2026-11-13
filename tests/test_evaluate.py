import copy
import json
import unittest
from pathlib import Path

from policy_replay_book.evaluate import evaluate
from policy_replay_book.io import read_json
from policy_replay_book.offpolicy import (
    candidate_version,
    prepare_units,
    validate_candidate,
)


def load(name):
    return json.loads((Path(__file__).resolve().parents[1] / 'fixtures' / name).read_text(encoding='utf-8'))


class SingleStepEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.document = load('study.json')
        self.result = evaluate(self.document)

    def test_output_contract(self):
        for key in ('estimand', 'candidates', 'support', 'uncertainty', 'recommendation'):
            self.assertIn(key, self.result)
        self.assertEqual(self.result['estimand']['mode'], 'single_step')
        self.assertEqual(self.result['estimand']['reward_version'], 'net-v1')
        for name, candidate in self.result['candidates'].items():
            self.assertIn('candidate_version', candidate)
            self.assertIn('support', candidate)
            self.assertIn('censoring', candidate)
            self.assertIn('difference_vs_baseline', candidate)

    def test_baseline_matches_factual_mean(self):
        # 可核验基准：与 decisions.factual_summary 的已结算均值口径一致。
        from policy_replay_book.decisions import factual_summary
        summary = factual_summary(self.document)
        settled_rows = [r for r in self.document['decisions']
                        if r['reward_known_at'] <= self.document['cutoff']]
        expected = sum(r['reward'] for r in settled_rows) / len(settled_rows)
        self.assertAlmostEqual(self.result['baseline']['logging_policy_factual_mean'], expected)
        self.assertEqual(summary['pending'],
                         len(self.document['decisions']) - len(settled_rows))

    def test_censored_never_treated_as_zero(self):
        # 未结算单元被计数，但主估计的均值只来自已结算单元。
        candidate = self.result['candidates']['cautious']
        self.assertEqual(candidate['support']['n_settled'] + candidate['support']['n_censored'],
                         len(self.document['decisions']))
        self.assertGreater(candidate['censoring']['n_censored'], 0)
        # Manski 界存在且基准均值落在界内。
        low, high = candidate['censoring']['manski_bounds_baseline']
        self.assertLessEqual(low, self.result['baseline']['logging_policy_factual_mean'])
        self.assertLessEqual(self.result['baseline']['logging_policy_factual_mean'], high)

    def test_user_clusters_kept_together(self):
        # 同一用户的所有单元必须落在同一交叉拟合折。
        mode, units = prepare_units(self.document)
        from policy_replay_book.offpolicy import _assign_folds
        clusters = sorted({u['cluster'] for u in units})
        fold_of = _assign_folds(clusters, self.document['seed'])
        folds_by_user = {}
        for unit in units:
            folds_by_user.setdefault(unit['cluster'], set()).add(fold_of[unit['cluster']])
        self.assertTrue(all(len(folds) == 1 for folds in folds_by_user.values()))

    def test_difference_uncertainty_is_paired(self):
        for candidate in self.result['candidates'].values():
            diff = candidate['difference_vs_baseline']
            low, high = diff['ci']
            self.assertIsNotNone(low)
            self.assertLess(low, high)
            self.assertIn('paired_cluster_bootstrap', diff['method'])

    def test_recommendation_is_conservative(self):
        # study.json 上没有候选能给出严格为正的净收益区间。
        for decision in self.result['recommendation'].values():
            self.assertIn(decision['decision'],
                          ('hold', 'do_not_proceed', 'reject_offline', 'proceed_small_traffic'))
        self.assertNotEqual(self.result['recommendation']['direct']['decision'],
                            'proceed_small_traffic')


class UnsupportedActionTests(unittest.TestCase):
    def setUp(self):
        self.document = load('unsupported.json')
        self.result = evaluate(self.document)

    def test_zero_logging_action_rejected(self):
        heavy = self.result['candidates']['unsupported']
        self.assertFalse(heavy['identifiable'])
        self.assertGreater(len(heavy['support']['hard_positivity_violations']), 0)
        self.assertGreater(heavy['support']['candidate_mass_on_zero_logging_action'], 0.5)
        self.assertEqual(self.result['recommendation']['unsupported']['decision'], 'reject_offline')

    def test_covered_candidate_remains_evaluable(self):
        covered = self.result['candidates']['direct']
        self.assertTrue(covered['identifiable'])
        self.assertEqual(covered['support']['hard_positivity_violations'], [])


class TrajectoryEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.document = load('sequential.json')
        self.result = evaluate(self.document)

    def test_trajectory_mode_contract(self):
        self.assertEqual(self.result['estimand']['mode'], 'trajectory')
        self.assertEqual(self.result['estimand']['unit'], 'trajectory')
        candidate = self.result['candidates']['direct']
        # 多步模式不提供单步双重稳健估计。
        self.assertIsNone(candidate['estimators']['doubly_robust'])
        self.assertIn('联合似然比', candidate['estimators']['note'])

    def test_no_support_trajectory_rejected(self):
        candidate = self.result['candidates']['direct']
        self.assertFalse(candidate['identifiable'])
        self.assertEqual(candidate['support']['zero_weight_episode_fraction'], 1.0)
        self.assertIsNone(candidate['difference_vs_baseline']['estimate'])
        self.assertEqual(self.result['recommendation']['direct']['decision'], 'reject_offline')

    def test_middle_step_null_reward_is_structural_zero(self):
        mode, units = prepare_units(self.document)
        self.assertEqual(mode, 'trajectory')
        # 全部轨迹都已结算（reward_known_at=80 <= cutoff=100），中间步 null 不是未结算。
        self.assertTrue(all(unit['settled'] for unit in units))
        self.assertTrue(all(unit['reward'] in (0.0, 1.0) for unit in units))


class ModeMismatchTests(unittest.TestCase):
    def test_single_step_rejects_multi_step_request(self):
        document = load('sequential.json')
        document['mode'] = 'single_step'
        with self.assertRaises(ValueError):
            evaluate(document)

    def test_unknown_mode_rejected(self):
        document = load('study.json')
        document['mode'] = 'something_else'
        with self.assertRaises(ValueError):
            evaluate(document)


class CandidateValidationTests(unittest.TestCase):
    def setUp(self):
        self.document = load('study.json')

    def test_missing_decision_entry(self):
        policy = copy.deepcopy(self.document['candidate_policies']['cautious'])
        first_id = self.document['decisions'][0]['decision_id']
        del policy[first_id]
        with self.assertRaises(ValueError):
            validate_candidate('broken', self.document['decisions'], policy)

    def test_non_normalized_probabilities(self):
        policy = copy.deepcopy(self.document['candidate_policies']['cautious'])
        for entry in policy.values():
            entry['direct'] += 0.1
        with self.assertRaises(ValueError):
            validate_candidate('broken', self.document['decisions'], policy)

    def test_version_fingerprint_stable_and_distinct(self):
        policies = self.document['candidate_policies']
        versions = {name: candidate_version(policy) for name, policy in policies.items()}
        self.assertEqual(len(set(versions.values())), len(versions))
        # 同一概率表重复计算指纹一致。
        self.assertEqual(candidate_version(policies['direct']), versions['direct'])


class ProvenanceTests(unittest.TestCase):
    def test_reward_and_candidate_versions_frozen(self):
        result = evaluate(load('study.json'))
        self.assertEqual(result['provenance']['reward_version'], 'net-v1')
        self.assertEqual(set(result['provenance']['candidate_versions']),
                         set(result['candidates']))
        self.assertEqual(result['estimand']['method_version'],
                         result['provenance']['method_version'])


if __name__ == '__main__':
    unittest.main()
