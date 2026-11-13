"""用已知生成概率的本地模拟，验证离线评估在三种情形下的行为。

演示一：只按实际动作分组比较会得到错误结论（选择偏差/辛普森式反转）。
演示二：候选动作在日志策略下概率为零，没有历史支持，必须拒绝。
演示三：两步决策不能降格为单步结果，整段价值只能用联合似然比识别。

用法：
    python3 -I tools/run_simulations.py --data-dir /tmp/prb-sim-data \
        --output-dir /tmp/prb-sim-out

脚本只生成虚构数据并调用离线评估，不连接任何外部系统，不代替用户执行操作。
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# 隔离模式（python3 -I）下 cwd 不在 sys.path 中；按脚本位置显式加入项目根，
# 数据目录仍通过参数传入，解释器不会从数据目录加载任何模块。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

ACTIONS_3 = ['direct', 'clarify', 'human']
ACTIONS_2 = ['A', 'B']


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


# ---------------------------------------------------------------- 演示一
def make_selection_bias(path, seed=101, n_users=120, per_user=6, cutoff=100):
    """上下文分布与日志策略共同制造选择偏差。

    真实结构：easy 上下文占多数且 direct 更优；hard 上下文里 clarify 更优。
    日志策略在 easy 上偏好 direct、在 hard 上偏好 clarify，于是按实际动作分组
    得到的事实均值会把上下文效应错记到动作头上，甚至反转真实动作排序。
    """
    rng = np.random.default_rng(seed)
    contexts = [
        {'id': 'easy_mid', 'difficulty': 'easy', 'tier': 'mid',
         'propensity': {'direct': 0.8, 'clarify': 0.15, 'human': 0.05},
         'mean': {'direct': 0.9, 'clarify': 0.5, 'human': 0.3}, 'p': 0.7},
        {'id': 'hard_entry', 'difficulty': 'hard', 'tier': 'entry',
         'propensity': {'direct': 0.1, 'clarify': 0.7, 'human': 0.2},
         'mean': {'direct': 0.15, 'clarify': 0.8, 'human': 0.4}, 'p': 0.3},
    ]
    # 真实策略价值（上下文分布下的期望收益），用于核验。
    true_direct = sum(c['p'] * c['mean']['direct'] for c in contexts)
    true_logging = sum(c['p'] * sum(c['propensity'][a] * c['mean'][a] for a in ACTIONS_3)
                       for c in contexts)
    decisions = []
    for i in range(n_users * per_user):
        cid = rng.choice(len(contexts), p=[c['p'] for c in contexts])
        context = contexts[cid]
        probs = np.array([context['propensity'][a] for a in ACTIONS_3])
        action = ACTIONS_3[rng.choice(3, p=probs)]
        reward = float(np.clip(context['mean'][action] + rng.normal(0, 0.15), -1.0, 1.0))
        # 约 8% 的结算在观察截止之后到达：不能置零。
        known_at = 140 if rng.random() < 0.08 else 60
        decisions.append({
            'decision_id': 'd%d' % i, 'request_id': 'r%d' % i,
            'user_id': 'u%d' % (i % n_users), 'step': 0,
            'context': {k: context[k] for k in ('difficulty', 'tier')},
            'available': ACTIONS_3, 'propensity': context['propensity'],
            'action': action, 'reward': reward, 'reward_known_at': known_at,
        })
    # 候选一：始终 direct（确定性策略）；候选二：温和地偏向 clarify。
    candidate = {
        'always_direct': {row['decision_id']: {'direct': 1.0, 'clarify': 0.0, 'human': 0.0}
                          for row in decisions},
        'mild_clarify': {row['decision_id']: {'direct': 0.3, 'clarify': 0.5, 'human': 0.2}
                         for row in decisions},
    }
    document = {'mode': 'single_step', 'cutoff': cutoff, 'seed': seed,
                'reward_version': 'sim-net-v1', 'decisions': decisions,
                'candidate_policies': candidate}
    write_json(path, document)
    return {'true_direct_value': true_direct, 'true_logging_value': true_logging,
            'true_diff_always_direct': true_direct - true_logging}


def naive_group_means(document):
    """错误做法：只按实际动作分组求事实均值（不做任何重权重）。"""
    cutoff = document['cutoff']
    sums = defaultdict(float)
    counts = defaultdict(int)
    for row in document['decisions']:
        if row['reward_known_at'] > cutoff:
            continue  # 正确地不置零，但分组本身无法纠正选择偏差
        sums[row['action']] += row['reward']
        counts[row['action']] += 1
    return {action: sums[action] / counts[action] for action in sorted(counts)}


# ---------------------------------------------------------------- 演示二
def make_unsupported_action(path, seed=202, n_users=40, per_user=5, cutoff=100):
    """日志策略从不选择 human（概率为零），候选却把大量概率放在 human 上。"""
    rng = np.random.default_rng(seed)
    decisions = []
    for i in range(n_users * per_user):
        difficulty = rng.choice(['easy', 'hard'])
        propensity = {'direct': 0.5, 'clarify': 0.5, 'human': 0.0}
        action = 'direct' if rng.random() < 0.5 else 'clarify'
        mean = 0.85 if action == 'direct' else 0.6
        reward = float(np.clip(mean + rng.normal(0, 0.1), -1.0, 1.0))
        decisions.append({
            'decision_id': 'd%d' % i, 'request_id': 'r%d' % i,
            'user_id': 'u%d' % (i % n_users), 'step': 0,
            'context': {'difficulty': difficulty},
            'available': ACTIONS_3, 'propensity': propensity,
            'action': action, 'reward': reward, 'reward_known_at': 60,
        })
    candidate = {
        'covered': {row['decision_id']: {'direct': 0.6, 'clarify': 0.4, 'human': 0.0}
                    for row in decisions},
        'human_heavy': {row['decision_id']: {'direct': 0.2, 'clarify': 0.2, 'human': 0.6}
                        for row in decisions},
    }
    document = {'mode': 'single_step', 'cutoff': cutoff, 'seed': seed,
                'reward_version': 'sim-net-v1', 'decisions': decisions,
                'candidate_policies': candidate}
    write_json(path, document)


# ---------------------------------------------------------------- 演示三
def make_two_step(path, seed=303, n_users=80, per_user=6, cutoff=100):
    """两步 MDP：第一步动作改变第二步状态；收益只在第二步结算。

    第一步收益始终为 null（结构性零收益），最终收益在第二步到达。
    """
    rng = np.random.default_rng(seed)
    # 第二步状态由第一步动作决定：A 以 0.8 进入好状态 G，B 以 0.7 进入差状态 H。
    step1_means = {'G': {'A': 0.9, 'B': 0.7}, 'H': {'A': 0.5, 'B': 0.3}}
    # 日志策略给两个动作都足够的探索概率，最大逐步重要性比为 1/0.3。
    log_p0 = {'A': 0.5, 'B': 0.5}
    log_p1 = {'A': 0.7, 'B': 0.3}
    decisions = []
    for episode in range(n_users * per_user):
        user = episode % n_users
        a0 = 'A' if rng.random() < log_p0['A'] else 'B'
        if a0 == 'A':
            state1 = 'G' if rng.random() < 0.8 else 'H'
        else:
            state1 = 'H' if rng.random() < 0.7 else 'G'
        a1 = 'A' if rng.random() < log_p1['A'] else 'B'
        final_reward = float(np.clip(step1_means[state1][a1] + rng.normal(0, 0.1), 0.0, 1.0))
        decisions.append({
            'decision_id': 't%d-0' % episode, 'request_id': 't%d' % episode,
            'user_id': 'u%d' % user, 'step': 0,
            'context': {'previous_action': None}, 'available': ACTIONS_2,
            'propensity': dict(log_p0), 'action': a0,
            'reward': None, 'reward_known_at': 60,
        })
        decisions.append({
            'decision_id': 't%d-1' % episode, 'request_id': 't%d' % episode,
            'user_id': 'u%d' % user, 'step': 1,
            'context': {'previous_action': a0, 'state': state1}, 'available': ACTIONS_2,
            'propensity': dict(log_p1), 'action': a1,
            'reward': final_reward, 'reward_known_at': 60,
        })
    candidate = {
        'always_A': {row['decision_id']: {'A': 1.0, 'B': 0.0} for row in decisions},
    }
    document = {'mode': 'trajectory', 'cutoff': cutoff, 'seed': seed,
                'reward_version': 'sim-final-v1', 'decisions': decisions,
                'candidate_policies': candidate}
    write_json(path, document)

    # 真实价值（由生成机制直接计算，供核验）。
    def state_dist(first_action):
        if first_action == 'A':
            return {'G': 0.8, 'H': 0.2}
        return {'G': 0.3, 'H': 0.7}

    def value(first_action, second_probs):
        total = 0.0
        for state, probability in state_dist(first_action).items():
            total += probability * sum(second_probs[a] * step1_means[state][a] for a in ACTIONS_2)
        return total

    true_candidate = value('A', {'A': 1.0, 'B': 0.0})
    true_logging = sum(log_p0[a0] * value(a0, log_p1) for a0 in ACTIONS_2)
    return {'true_candidate_value': true_candidate,
            'true_logging_value': true_logging,
            'true_diff': true_candidate - true_logging}


def naive_flattened_ipw(document, candidate_name):
    """错误做法：把两步决策拆成独立曝光，用单步概率乘最终收益。

    第一步记录的 reward 是 null（按零处理），第二步记录承担整段收益；
    权重不再是联合似然比，估计量被系统性扭曲。
    """
    policy = document['candidate_policies'][candidate_name]
    numerator, denominator = 0.0, 0.0
    for row in document['decisions']:
        if row['reward_known_at'] > document['cutoff']:
            continue
        ratio = policy[row['decision_id']][row['action']] / row['propensity'][row['action']]
        reward = row['reward'] if row['reward'] is not None else 0.0
        numerator += ratio * reward
        denominator += 1.0
    return numerator / denominator


def trajectory_ipw(document, candidate_name):
    """正确做法：按整段轨迹的联合似然比配整段最终收益（Horvitz-Thompson）。"""
    policy = document['candidate_policies'][candidate_name]
    by_request = defaultdict(list)
    for row in document['decisions']:
        by_request[row['request_id']].append(row)
    numerator, denominator = 0.0, 0
    for rows in by_request.values():
        if any(row['reward_known_at'] > document['cutoff'] for row in rows):
            continue
        weight = 1.0
        for row in sorted(rows, key=lambda item: item['step']):
            weight *= policy[row['decision_id']][row['action']] / row['propensity'][row['action']]
        total_reward = sum(row['reward'] for row in rows if row['reward'] is not None)
        numerator += weight * total_reward
        denominator += 1
    return numerator / denominator


def trajectory_snipw(document, candidate_name):
    """正确做法的自归一化版本：与评估模块的主估计口径一致。"""
    policy = document['candidate_policies'][candidate_name]
    by_request = defaultdict(list)
    for row in document['decisions']:
        by_request[row['request_id']].append(row)
    numerator, weight_sum = 0.0, 0.0
    for rows in by_request.values():
        if any(row['reward_known_at'] > document['cutoff'] for row in rows):
            continue
        weight = 1.0
        for row in sorted(rows, key=lambda item: item['step']):
            weight *= policy[row['decision_id']][row['action']] / row['propensity'][row['action']]
        total_reward = sum(row['reward'] for row in rows if row['reward'] is not None)
        numerator += weight * total_reward
        weight_sum += weight
    return numerator / weight_sum


def main():
    parser = argparse.ArgumentParser(description='生成三组已知机制的模拟数据并核验离线评估。')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(data_dir.iterdir()):
        raise SystemExit('数据目录必须为空：%s' % data_dir)

    # 延迟导入：脚本目录与数据目录隔离，项目包由调用方的 sys.path 提供。
    from policy_replay_book.evaluate import evaluate
    from policy_replay_book.io import read_json, sanitize

    results = {}

    # ---- 演示一：分组均值反转 ----
    path1 = data_dir / 'sim1_selection_bias.json'
    truth1 = make_selection_bias(path1)
    doc1 = read_json(path1)
    groups = naive_group_means(doc1)
    eval1 = evaluate(doc1)
    write_json(output_dir / 'sim1_evaluation.json', sanitize(eval1))
    direct_candidate = eval1['candidates']['always_direct']
    check1 = {
        'naive_group_means': groups,
        'naive_conclusion': 'direct 看起来优于 clarify（分组均值）',
        'true_values': truth1,
        'true_ordering': 'hard 上下文里 clarify 更优；上下文加权后 always_direct 相对日志为负',
        'ope_diff_estimate': direct_candidate['difference_vs_baseline']['estimate'],
        'ope_diff_ci': direct_candidate['difference_vs_baseline']['ci'],
        'naive_is_wrong': bool(groups['direct'] > groups['clarify'] and truth1['true_diff_always_direct'] < 0),
        'ope_recovers_negative': bool(direct_candidate['difference_vs_baseline']['estimate'] < 0
                                      and direct_candidate['difference_vs_baseline']['ci'][1] <= 0),
    }
    results['demo1_selection_bias'] = check1

    # ---- 演示二：零支持动作 ----
    path2 = data_dir / 'sim2_unsupported_action.json'
    make_unsupported_action(path2)
    doc2 = read_json(path2)
    eval2 = evaluate(doc2)
    write_json(output_dir / 'sim2_evaluation.json', sanitize(eval2))
    heavy = eval2['candidates']['human_heavy']
    check2 = {
        'identifiable': heavy['identifiable'],
        'hard_positivity_violations': len(heavy['support']['hard_positivity_violations']),
        'candidate_mass_on_zero_logging_action':
            heavy['support']['candidate_mass_on_zero_logging_action'],
        'recommendation': eval2['recommendation']['human_heavy'],
        'correctly_rejected': (not heavy['identifiable'] and
                               eval2['recommendation']['human_heavy']['decision'] == 'reject_offline'),
    }
    results['demo2_unsupported_action'] = check2

    # ---- 演示三：多步不能降格为单步 ----
    path3 = data_dir / 'sim3_two_step.json'
    truth3 = make_two_step(path3)
    doc3 = read_json(path3)
    eval3 = evaluate(doc3)
    write_json(output_dir / 'sim3_evaluation.json', sanitize(eval3))
    proper = trajectory_snipw(doc3, 'always_A')
    raw_ipw = trajectory_ipw(doc3, 'always_A')
    flattened = naive_flattened_ipw(doc3, 'always_A')
    candidate = eval3['candidates']['always_A']
    check3 = {
        'true_values': truth3,
        'trajectory_snipw_value': proper,
        'trajectory_raw_ipw_value': raw_ipw,
        'naive_flattened_value': flattened,
        'naive_bias_vs_true': flattened - truth3['true_candidate_value'],
        'snipw_bias_vs_true': proper - truth3['true_candidate_value'],
        'ope_snipw_value': candidate['estimators']['snipw']['value'],
        'ope_diff_estimate': candidate['difference_vs_baseline']['estimate'],
        'trajectory_estimator_close': abs(proper - truth3['true_candidate_value']) < 0.08,
        'naive_estimator_biased': abs(flattened - truth3['true_candidate_value']) > 0.08,
        'identifiable': candidate['identifiable'],
    }
    results['demo3_multistep'] = check3

    # ---- 汇总断言：任一不成立则以非零状态退出 ----
    assertions = {
        'demo1 分组均值错误且 OPE 纠正': check1['naive_is_wrong'] and check1['ope_recovers_negative'],
        'demo2 零支持动作被拒绝': check2['correctly_rejected'],
        'demo3 联合似然比准确而扁平化估计有偏': (check3['trajectory_estimator_close'] and
                                              check3['naive_estimator_biased']),
    }
    results['assertions'] = assertions
    write_json(output_dir / 'simulation_summary.json', results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not all(assertions.values()):
        raise SystemExit(1)
    print('全部模拟核验通过。')


if __name__ == '__main__':
    sys.exit(main())
