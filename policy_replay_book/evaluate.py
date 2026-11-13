"""候选澄清策略的离线决策支持。

输出只回答“某候选策略相对当前日志策略是否值得进入小流量”，不代替用户执行操作。
结论与输入的奖励口径（reward_version）、候选版本指纹、评估阈值一并固化保存。
"""

import argparse
import hashlib

import numpy as np

from .decisions import validate_decisions
from .offpolicy import (
    CLIP_LEVELS,
    EMPIRICAL_GAP_MASS,
    MAX_CENSOR_FRACTION,
    MAX_WEIGHT_CAP,
    MIN_ESS,
    MIN_ESS_RATIO,
    N_BOOTSTRAP,
    N_FOLDS,
    candidate_version,
    evaluate_single_step,
    evaluate_trajectory,
    prepare_units,
    validate_candidate,
)

METHOD_VERSION = 'offpolicy-eval-v1'
IDENTIFIES = {
    'single_step': '在日志策略的上下文分布下，单步决策规则变化的期望单步结算收益；'
                   '不包含该决策对后续状态与后续决策的影响。',
    'trajectory': '整段轨迹策略变化的期望整段结算总收益，权重为逐步重要性比连乘；'
                  '可识别路径层面的总效应，但要求路径上每一步都满足正性，权重方差随步数累积。',
}


def _candidate_rng(seed, candidate_name):
    digest = hashlib.sha256((str(seed) + '|' + candidate_name).encode('utf-8')).digest()
    integer = int.from_bytes(digest[:8], 'little')
    return np.random.default_rng(integer)


def _recommendation(result, censoring, diff):
    """建议规则：支持不足、截断不确定或区间不显著为正时拒绝/暂缓。"""
    if not result['identifiable'] or result['support']['hard_positivity_violations']:
        return {
            'decision': 'reject_offline',
            'reason': '存在正性违反或有效支持不足，离线数据无法识别该策略价值',
        }
    support = result['support']
    if support['ess'] < MIN_ESS or (support['ess_ratio'] or 0.0) < MIN_ESS_RATIO:
        return {'decision': 'reject_offline', 'reason': '有效样本量 ESS 低于阈值'}
    if (support.get('max_weight') or support.get('max_episode_weight') or 0.0) > MAX_WEIGHT_CAP:
        return {'decision': 'hold', 'reason': '极端权重超过上限，估计对少数轨迹高度敏感'}
    if 'empirical_gap_mass' in support and support['empirical_gap_mass'] > EMPIRICAL_GAP_MASS:
        return {'decision': 'hold', 'reason': '候选在无历史覆盖的上下文-动作格上投放概率质量过大'}
    if support.get('zero_weight_episode_fraction') == 1.0:
        return {'decision': 'reject_offline', 'reason': '日志中不存在候选策略的任何支持轨迹'}
    if censoring['censor_fraction'] > MAX_CENSOR_FRACTION or censoring['bounds_straddle_zero']:
        return {
            'decision': 'hold',
            'reason': '未结算比例过高或截断界跨越零，净收益方向不能确定',
        }
    if diff is None or diff.get('ci') is None or diff['ci'][0] is None:
        return {'decision': 'hold', 'reason': '无法构造净收益差的置信区间'}
    low, high = diff['ci']
    if low > 0:
        return {
            'decision': 'proceed_small_traffic',
            'reason': '净收益差的簇 bootstrap 95% 置信区间整体为正，且支持度与截断检查通过',
        }
    if high <= 0:
        return {'decision': 'do_not_proceed', 'reason': '净收益差的置信区间整体非正'}
    return {'decision': 'hold', 'reason': '净收益差置信区间跨越零，证据不足'}


def evaluate(document):
    rows = validate_decisions(document['decisions'])
    mode, units = prepare_units(document)
    candidate_policies = document['candidate_policies']
    if not isinstance(candidate_policies, dict) or not candidate_policies:
        raise ValueError('缺少候选策略')

    candidates_out = {}
    support_summary = {}
    for name in sorted(candidate_policies):
        policy_map = candidate_policies[name]
        validate_candidate(name, rows, policy_map)
        rng = _candidate_rng(document['seed'], name)
        if mode == 'single_step':
            result = evaluate_single_step(document, units, name, policy_map, rng)
        else:
            result = evaluate_trajectory(document, units, name, policy_map, rng)
        result['candidate_version'] = candidate_version(policy_map)
        result['recommendation'] = _recommendation(
            result, result['censoring'], result['difference_vs_baseline'])
        candidates_out[name] = result
        support_summary[name] = {
            'identifiable': result['identifiable'],
            'ess': result['support']['ess'],
            'ess_ratio': result['support']['ess_ratio'],
            'hard_positivity_violations': len(result['support']['hard_positivity_violations']),
            'candidate_mass_on_zero_logging_action':
                result['support'].get('candidate_mass_on_zero_logging_action', 0.0),
            'max_weight': result['support'].get('max_weight')
                          or result['support'].get('max_episode_weight'),
            'censor_fraction': result['censoring']['censor_fraction'],
        }

    baseline = next(iter(candidates_out.values()))['baseline_bootstrap']
    output = {
        'estimand': {
            'mode': mode,
            'what_can_be_identified': IDENTIFIES[mode],
            'unit': 'decision' if mode == 'single_step' else 'trajectory',
            'reward_version': document['reward_version'],
            'cutoff': document['cutoff'],
            'method_version': METHOD_VERSION,
            'n_decisions': len(rows),
            'n_units': len(units),
            'n_users': len({row['user_id'] for row in rows}),
            'censored_not_zero': '观察截止后结算的单元按未结算处理，不置零；'
                                 '主估计仅使用已结算单元，并附 Manski 界与 IPCW（单步）。',
        },
        'baseline': {
            'logging_policy_factual_mean': baseline['mean'],
            'ci95_cluster_bootstrap': baseline['ci'],
            'clustered_se': baseline['clustered_se'],
            'check': '事实均值是可核验基准：可直接按已结算记录求和复核',
        },
        'candidates': candidates_out,
        'support': {
            'summary': support_summary,
            'thresholds': {
                'min_ess': MIN_ESS,
                'min_ess_ratio': MIN_ESS_RATIO,
                'max_weight_cap': MAX_WEIGHT_CAP,
                'max_censor_fraction': MAX_CENSOR_FRACTION,
                'empirical_gap_mass': EMPIRICAL_GAP_MASS,
            },
            'rule': '硬正性违反（候选质量落在日志零概率动作上）直接拒绝；'
                    'ESS、极端权重或经验外推不达标时不给上线结论；不从近邻用户借值。',
        },
        'uncertainty': {
            'method': '用户分簇百分位置信区间（bootstrap %d 次）+ 簇稳健影响函数标准误' % N_BOOTSTRAP,
            'n_folds': N_FOLDS if mode == 'single_step' else None,
            'weight_clipping_levels': list(CLIP_LEVELS),
            'note': '多步模式只提供重权重估计，不提供单步收益模型；截断估计逐档展示以暴露敏感度。',
        },
        'recommendation': {
            name: candidates_out[name]['recommendation'] for name in candidates_out
        },
        'provenance': {
            'reward_version': document['reward_version'],
            'method_version': METHOD_VERSION,
            'candidate_versions': {name: candidates_out[name]['candidate_version']
                                   for name in candidates_out},
            'input_seed': document['seed'],
            'frozen_warning': '结论绑定以上奖励口径与候选版本指纹；更换候选文件不会改写本文件，'
                              '应另存新分析。',
        },
    }
    return output


def _module_main():
    import sys
    from .io import read_json, write_json, sanitize

    parser = argparse.ArgumentParser(
        description='候选澄清策略离线决策支持：估计相对当前日志策略的净收益差，'
                    '不代替用户执行操作。')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        document = read_json(args.input)
        write_json(args.output, sanitize(evaluate(document)))
    except (ValueError, KeyError, TypeError, OSError) as error:
        print('处理失败：' + str(error), file=sys.stderr)
        return 2
    print('结果已保存：' + args.output)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(_module_main())
