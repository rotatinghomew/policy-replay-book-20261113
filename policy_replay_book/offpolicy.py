"""离线策略评估（OPE）的核心计算。

只做离线决策支持，不代替用户执行任何动作。设计原则：

1. 日志策略（logging policy）给过正概率的动作才可重权重识别，未覆盖的动作
   不从近邻用户或相似上下文借值。
2. 观察截止之后才结算的结果是“未结算”，不是零收益；主估计只在已结算单元上
   定义，未结算规模与边界单独报告。
3. 单步模式的估计单元是独立决策；多步模式的估计单元是整段轨迹，权重为逐步
   重要性比的连乘，收益为整段结算收益之和。两种模式的可识别结论不同。
4. 收益预测模型按用户分簇交叉拟合，同一条结果不会同时用于拟合和评价。
"""

import hashlib
import json
import math
from collections import defaultdict

import numpy as np

# 支持度与建议阈值（全部在输出中显式声明，便于复核与调整）。
MIN_ESS = 30.0
MIN_ESS_RATIO = 0.1
MAX_WEIGHT_CAP = 50.0
MAX_CENSOR_FRACTION = 0.2
CLIP_LEVELS = (5.0, 10.0, 20.0, 50.0)
N_FOLDS = 5
N_BOOTSTRAP = 1000
# 候选在“从未被日志实际覆盖的上下文-动作格”上投放的概率质量超过该值，
# 视为经验外推，不给出上线结论。
EMPIRICAL_GAP_MASS = 0.01


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def candidate_version(policy_map):
    """候选版本指纹：概率表的规范化序列化后取 SHA-256。

    结论与候选版本绑定，下一周换候选文件不会改变已保存的结论。
    """
    payload = json.dumps(policy_map, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def validate_candidate(name, rows, policy_map):
    """核对候选概率表：编号齐全、动作集合一致、概率有限且归一。"""
    if not isinstance(policy_map, dict) or not policy_map:
        raise ValueError('候选策略 %s 缺少概率表' % name)
    for row in rows:
        entry = policy_map.get(row['decision_id'])
        if entry is None:
            raise ValueError('候选策略 %s 缺少决策 %s 的概率' % (name, row['decision_id']))
        actions = set(row['available'])
        if set(entry) != actions:
            raise ValueError('候选策略 %s 在决策 %s 上动作集合不一致' % (name, row['decision_id']))
        if any(not _finite(value) or value < 0 or value > 1 for value in entry.values()):
            raise ValueError('候选策略 %s 存在越界概率' % name)
        if not math.isclose(sum(entry.values()), 1.0, abs_tol=1e-9):
            raise ValueError('候选策略 %s 的概率之和不为一' % name)


def _context_key(row):
    return json.dumps(row['context'], sort_keys=True, separators=(',', ':'))


def _settled(row, cutoff):
    return row['reward_known_at'] <= cutoff


def prepare_units(document):
    """按输入声明的模式构造估计单元，并完成模式一致性检查。"""
    mode = document['mode']
    rows = document['decisions']
    cutoff = document['cutoff']
    by_request = defaultdict(list)
    for row in rows:
        by_request[row['request_id']].append(row)
    for request_rows in by_request.values():
        request_rows.sort(key=lambda item: item['step'])

    if mode == 'single_step':
        for request_rows in by_request.values():
            if len(request_rows) != 1 or request_rows[0]['step'] != 0:
                raise ValueError('single_step 模式下每个请求只能有一次决策')
        units = []
        for row in rows:
            settled = _settled(row, cutoff)
            units.append({
                'decision_id': row['decision_id'],
                'request_id': row['request_id'],
                'cluster': row['user_id'],
                'context_key': _context_key(row),
                'action': row['action'],
                'settled': settled,
                'reward': float(row['reward']) if (settled and row['reward'] is not None) else 0.0,
                'row': row,
            })
        return mode, units

    if mode != 'trajectory':
        raise ValueError('未知评估模式：%r' % (mode,))

    units = []
    for request_id, request_rows in sorted(by_request.items()):
        if [item['step'] for item in request_rows] != list(range(len(request_rows))):
            raise ValueError('请求 %s 步骤不连续' % request_id)
        users = {item['user_id'] for item in request_rows}
        if len(users) != 1:
            raise ValueError('请求 %s 的用户归属不一致' % request_id)
        settled_flags = [_settled(item, cutoff) for item in request_rows]
        all_settled = all(settled_flags)
        # 已结算步骤上 reward 为 null 表示结构性零收益（多步中的中间步）。
        reward = (sum(float(item['reward']) for item in request_rows
                      if item['reward'] is not None)
                  if all_settled else 0.0)
        units.append({
            'request_id': request_id,
            'cluster': request_rows[0]['user_id'],
            'settled': all_settled,
            'reward': reward,
            'rows': request_rows,
        })
    return mode, units


def _assign_folds(clusters, seed):
    """把用户（簇）稳定地分到 K 折，同一用户的全部记录落在同一折。"""
    ordered = sorted(clusters)
    rng = np.random.default_rng(seed)
    shuffle = rng.permutation(len(ordered))
    fold_of_cluster = {}
    for position, index in enumerate(shuffle):
        fold_of_cluster[ordered[index]] = int(position % N_FOLDS)
    return fold_of_cluster


def _cell_action_means(indices, units):
    sums = defaultdict(float)
    counts = defaultdict(int)
    for index in indices:
        unit = units[index]
        if not unit['settled']:
            continue
        key = (unit['context_key'], unit['action'])
        sums[key] += unit['reward']
        counts[key] += 1
    return {key: sums[key] / counts[key] for key in counts}


def _dm_value(model, unit, probs):
    """直接法价值 E_pi[Q]；未覆盖的格没有模型值时返回 None（不借值）。"""
    total = 0.0
    for action, probability in probs.items():
        if probability == 0.0:
            continue
        prediction = model.get((unit['context_key'], action))
        if prediction is None:
            return None
        total += probability * prediction
    return total


def _percentile(samples, q):
    clean = [value for value in samples if value is not None and math.isfinite(value)]
    if not clean:
        return None
    return float(np.percentile(np.asarray(clean, dtype=float), q))


def _ci(samples):
    if not isinstance(samples, list) or len(samples) < 2:
        return [None, None]
    low, high = _percentile(samples, 2.5), _percentile(samples, 97.5)
    if low is None or high is None:
        return [None, None]
    return [low, high]


def _straddles_zero(low, high):
    return low is not None and high is not None and low <= 0 <= high


def _clustered_se(per_unit_terms, units, settled_units_only):
    """簇稳健标准误：影响项先按用户聚合，再除以簇数平方根。"""
    per_cluster = defaultdict(float)
    for i, term in enumerate(per_unit_terms):
        if settled_units_only and not units[i]['settled']:
            continue
        per_cluster[units[i]['cluster']] += term
    values = np.array(list(per_cluster.values()), dtype=float)
    n_clusters = len(values)
    if n_clusters < 2:
        return None
    return float(math.sqrt(np.sum((values - values.mean()) ** 2)) / n_clusters)


def _resample_clusters(units, rng):
    """按用户簇有放回重采样，返回单元下标列表。"""
    clusters = sorted({unit['cluster'] for unit in units})
    by_cluster = defaultdict(list)
    for i, unit in enumerate(units):
        by_cluster[unit['cluster']].append(i)
    picked = rng.choice(len(clusters), size=len(clusters), replace=True)
    index_list = []
    for c_index in picked:
        index_list.extend(by_cluster[clusters[c_index]])
    return index_list


def evaluate_single_step(document, units, candidate_name, policy_map, rng):
    cutoff = document['cutoff']
    n = len(units)
    settled_index = [i for i, unit in enumerate(units) if unit['settled']]
    n_settled = len(settled_index)
    n_censored = n - n_settled
    settled_mask = np.array([unit['settled'] for unit in units])

    # ---- 支持度：硬正性检查覆盖候选可能选择的全部动作，而不只是实际动作 ----
    hard_violations = []
    zero_logging_mass = 0.0       # 候选放在日志零概率动作上的概率质量
    empirical_gap_mass = 0.0      # 候选放在无历史观测格上的概率质量
    observed_cells = {(unit['context_key'], unit['action']) for unit in units if unit['settled']}
    weights = np.zeros(n)
    for i, unit in enumerate(units):
        row = unit['row']
        probs = policy_map[unit['decision_id']]
        for action, probability in probs.items():
            if probability <= 0:
                continue
            if row['propensity'][action] <= 0:
                zero_logging_mass += probability / n
                hard_violations.append({'decision_id': unit['decision_id'], 'action': action,
                                        'candidate_probability': probability})
            if (unit['context_key'], action) not in observed_cells:
                empirical_gap_mass += probability / n
        weights[i] = probs[unit['action']] / row['propensity'][unit['action']]

    w = weights
    w_s = w[settled_mask]
    rewards = np.array([unit['reward'] for unit in units])
    y_s = rewards[settled_mask]
    mean_reward = float(np.mean(y_s))
    reward_min, reward_max = float(np.min(y_s)), float(np.max(y_s))

    ipw_value = float(np.mean(w_s * y_s))
    denom = float(np.mean(w_s))
    snipw_value = float(np.mean(w_s * y_s) / denom) if denom > 0 else None

    clipped = {}
    for cap in CLIP_LEVELS:
        wc = np.minimum(w_s, cap)
        clipped[str(cap)] = {
            'cap': cap,
            'value': float(np.mean(wc * y_s) / np.mean(wc)),
            'fraction_at_cap': float(np.mean(w_s >= cap)),
        }

    # ---- 用户分簇交叉拟合的收益模型与双重稳健估计 ----
    clusters = sorted({unit['cluster'] for unit in units})
    fold_of_cluster = _assign_folds(clusters, document['seed'])
    folds = np.array([fold_of_cluster[unit['cluster']] for unit in units])

    def cross_fit(index_list, probs_lookup):
        """在给定单元上按用户簇交叉拟合，返回（直接法预测, Q(s,a_taken) 预测）。"""
        sub = [units[i] for i in index_list]
        sub_clusters = sorted({unit['cluster'] for unit in sub})
        sub_fold = _assign_folds(sub_clusters, document['seed'])
        fold_arr = np.array([sub_fold[unit['cluster']] for unit in sub])
        dm_arr = np.full(len(sub), np.nan)
        q_arr = np.full(len(sub), np.nan)
        for fold in range(N_FOLDS):
            train_idx = [j for j in range(len(sub)) if fold_arr[j] != fold and sub[j]['settled']]
            model = _cell_action_means(train_idx, sub)
            for j in range(len(sub)):
                if fold_arr[j] != fold:
                    continue
                unit = sub[j]
                dm = _dm_value(model, unit, probs_lookup(unit))
                if dm is not None:
                    dm_arr[j] = dm
                q = model.get((unit['context_key'], unit['action']))
                if q is not None:
                    q_arr[j] = q
        return sub, dm_arr, q_arr

    def dr_terms_for(sub, ww, dm_arr, q_arr):
        """双重稳健项 dm(s) + w·(y - Q(s,a_taken))。

        无模型格不借值：dm 与 Q 同时缺失时退回纯重权重 w·y；仅 Q 缺失时保留 dm
        （此时 dm 已对候选动作求和，缺少的格按零贡献），维持恒等式不被平移。
        """
        terms = np.zeros(len(sub))
        for j, unit in enumerate(sub):
            if not unit['settled']:
                continue
            dm_missing = math.isnan(dm_arr[j])
            q_missing = math.isnan(q_arr[j])
            if dm_missing and q_missing:
                terms[j] = ww[j] * unit['reward']
            elif q_missing:
                terms[j] = dm_arr[j]
            else:
                terms[j] = dm_arr[j] + ww[j] * (unit['reward'] - q_arr[j])
        return terms

    cand_lookup = lambda unit: policy_map[unit['decision_id']]
    sub_full, dm_cross, q_cross = cross_fit(list(range(n)), cand_lookup)
    dm_finite = np.isfinite(dm_cross[settled_mask])
    dm_value = float(np.mean(dm_cross[settled_mask][dm_finite])) if dm_finite.any() else None
    missing_model_mass = float(np.mean([math.isnan(dm_cross[i]) for i in range(n) if units[i]['settled']]))
    dr_terms = dr_terms_for(units, w, dm_cross, q_cross)
    dr_value = float(np.mean(dr_terms[settled_mask]))

    # ---- 有效支持度 ----
    ess = float(np.sum(w_s) ** 2 / np.sum(w_s ** 2)) if np.sum(w_s) > 0 else 0.0
    support = {
        'hard_positivity_violations': hard_violations,
        'candidate_mass_on_zero_logging_action': zero_logging_mass,
        'empirical_gap_mass': empirical_gap_mass,
        'max_weight': float(np.max(w_s)),
        'p99_weight': _percentile(w_s, 99),
        'ess': ess,
        'ess_ratio': ess / n_settled if n_settled else None,
        'missing_model_unit_fraction': missing_model_mass,
        'n_settled': n_settled,
        'n_censored': n_censored,
    }
    identifiable = (not hard_violations and ess >= MIN_ESS and
                    (support['ess_ratio'] or 0.0) >= MIN_ESS_RATIO and
                    support['max_weight'] <= MAX_WEIGHT_CAP and
                    empirical_gap_mass <= EMPIRICAL_GAP_MASS)

    # ---- 用户簇配对 bootstrap：每次重采样同时重算基准、SNIPW、DR ----
    boot_sn, boot_ip, boot_dr, boot_base, boot_diff_dr, boot_diff_sn = [], [], [], [], [], []
    for _ in range(N_BOOTSTRAP):
        index_list = _resample_clusters(units, rng)
        sub, dm_arr, q_arr = cross_fit(index_list, cand_lookup)
        ww = np.array([w[i] for i in index_list])
        sm = np.array([unit['settled'] for unit in sub])
        ww_s = ww[sm]
        yy = np.array([unit['reward'] for unit in sub])
        yy_s = yy[sm]
        if ww_s.size == 0 or np.sum(ww_s) <= 0:
            continue
        sn = float(np.mean(ww_s * yy_s) / np.mean(ww_s))
        ip = float(np.mean(ww_s * yy_s))
        terms = dr_terms_for(sub, ww, dm_arr, q_arr)
        dr = float(np.mean(terms[sm]))
        base = float(np.mean(yy_s))
        if not all(math.isfinite(v) for v in (sn, ip, dr, base)):
            continue
        boot_sn.append(sn)
        boot_ip.append(ip)
        boot_dr.append(dr)
        boot_base.append(base)
        boot_diff_dr.append(dr - base)
        boot_diff_sn.append(sn - base)

    # ---- 簇稳健解析标准误（影响函数按用户聚合后除以簇数平方根）----
    sn_terms = np.zeros(n)
    for i in settled_index:
        sn_terms[i] = (w[i] * (units[i]['reward'] - snipw_value)) / denom if denom > 0 else 0.0
    dr_center = float(np.mean([dr_terms[i] for i in settled_index]))
    dr_centered = np.zeros(n)
    base_terms = np.zeros(n)
    for i in settled_index:
        dr_centered[i] = dr_terms[i] - dr_center
        base_terms[i] = units[i]['reward'] - mean_reward

    # ---- 未结算：IPCW（按动作独立删失）与 Manski 截断界 ----
    settled_by_action = defaultdict(list)
    for i in settled_index:
        settled_by_action[units[i]['action']].append(i)
    total_by_action = defaultdict(int)
    for unit in units:
        total_by_action[unit['action']] += 1
    settle_prob = {action: len(indices) / total_by_action[action]
                   for action, indices in settled_by_action.items()}
    ipcw_assumable = (all(p > 0 for p in settle_prob.values()) and
                      len(settle_prob) == len(total_by_action))
    if ipcw_assumable:
        # 自归一化 IPCW：删失权重 g=1/P(settled|action) 与策略权重 w 一起归一，
        # 与 SNIPW 同口径；假设是“动作条件下的独立删失”。
        g = np.array([1.0 / settle_prob[units[i]['action']] if units[i]['settled'] else 0.0
                      for i in range(n)])
        ipcw_value = float(np.sum(g[settled_mask] * w[settled_mask] * y_s) /
                           np.sum(g[settled_mask] * w[settled_mask]))
        ipcw_base = float(np.sum(g[settled_mask] * y_s) / np.sum(g[settled_mask]))
    else:
        ipcw_value, ipcw_base = None, None

    censor_fraction = n_censored / n if n else 0.0
    base_low = (n_settled * mean_reward + n_censored * reward_min) / n
    base_high = (n_settled * mean_reward + n_censored * reward_max) / n
    if n_censored and denom > 0:
        w_cens = w[~settled_mask]
        cand_low = float((np.sum(w_s * y_s) + np.sum(w_cens) * reward_min) /
                         (np.sum(w_s) + np.sum(w_cens)))
        cand_high = float((np.sum(w_s * y_s) + np.sum(w_cens) * reward_max) /
                          (np.sum(w_s) + np.sum(w_cens)))
    else:
        cand_low, cand_high = snipw_value, snipw_value

    return {
        'identifiable': identifiable,
        'support': support,
        'censoring': {
            'n_censored': n_censored,
            'censor_fraction': censor_fraction,
            'settlement_probability_by_action': {k: float(v) for k, v in settle_prob.items()},
            'reward_bounds_observed': [reward_min, reward_max],
            'manski_bounds_baseline': [base_low, base_high],
            'manski_bounds_candidate_snipw': [cand_low, cand_high],
            'bounds_straddle_zero': bool(_straddles_zero(base_low, base_high) or
                                        _straddles_zero(cand_low, cand_high)),
            'ipcw_value': ipcw_value,
            'ipcw_baseline': ipcw_base,
            'ipcw_assumption': 'censoring_independent_within_action',
        },
        'estimators': {
            'ipw': {'value': ipw_value, 'ci': _ci(boot_ip)},
            'snipw': {'value': snipw_value, 'ci': _ci(boot_sn),
                      'clustered_se': _clustered_se(sn_terms, units, True)},
            'clipped_snipw': clipped,
            'dm_cross_fitted': {'value': dm_value},
            'doubly_robust': {'value': dr_value, 'ci': _ci(boot_dr),
                              'clustered_se': _clustered_se(dr_centered, units, True)},
        },
        'difference_vs_baseline': {
            'method': 'doubly_robust_paired_cluster_bootstrap',
            'estimate': dr_value - mean_reward,
            'ci': _ci(boot_diff_dr),
            'clustered_se': _clustered_se(dr_centered - base_terms, units, True),
            'snipw_estimate': (snipw_value - mean_reward) if snipw_value is not None else None,
            'snipw_ci': _ci(boot_diff_sn),
        },
        'baseline_bootstrap': {'mean': mean_reward, 'ci': _ci(boot_base),
                               'clustered_se': _clustered_se(base_terms, units, True)},
    }


def evaluate_trajectory(document, units, candidate_name, policy_map, rng):
    n = len(units)
    settled_positions = [i for i, unit in enumerate(units) if unit['settled']]
    n_settled = len(settled_positions)
    n_censored = n - n_settled
    settled_mask = np.array([unit['settled'] for unit in units])
    rewards = np.array([unit['reward'] for unit in units])
    y_s = rewards[settled_mask]
    mean_reward = float(np.mean(y_s)) if n_settled else None
    reward_min = float(np.min(y_s)) if n_settled else None
    reward_max = float(np.max(y_s)) if n_settled else None

    # ---- 联合重要性比：逐步似然比连乘，整段权重 ----
    # 硬正性检查覆盖候选在每一步所有可行动作上的概率：候选可能要求日志从不
    # 采样的动作，即便它没有出现在实际路径上，同样不可识别。
    hard_violations = []
    weights = np.zeros(n)
    zero_weight_episodes = 0
    per_step_max = 0.0
    for i, unit in enumerate(units):
        product = 1.0
        for row in unit['rows']:
            probs = policy_map[row['decision_id']]
            for action, probability in probs.items():
                if probability > 0 and row['propensity'][action] <= 0:
                    hard_violations.append({'decision_id': row['decision_id'], 'action': action,
                                            'candidate_probability': probability})
            taken = row['action']
            candidate_prob = probs[taken]
            logging_prob = row['propensity'][taken]
            if logging_prob > 0:
                step_ratio = candidate_prob / logging_prob
                product *= step_ratio
                per_step_max = max(per_step_max, step_ratio)
            else:
                product = 0.0
        weights[i] = product
        if product == 0.0:
            zero_weight_episodes += 1

    w_s = weights[settled_mask]
    ipw_value = float(np.mean(w_s * y_s)) if n_settled else None
    denom = float(np.mean(w_s)) if n_settled else 0.0
    snipw_value = float(np.mean(w_s * y_s) / denom) if denom > 0 else None
    ess = float(np.sum(w_s) ** 2 / np.sum(w_s ** 2)) if np.sum(w_s) > 0 else 0.0

    clipped = {}
    for cap in CLIP_LEVELS:
        wc = np.minimum(w_s, cap)
        clipped[str(cap)] = {
            'cap': cap,
            'value': float(np.mean(wc * y_s) / np.mean(wc)) if np.mean(wc) > 0 else None,
            'fraction_at_cap': float(np.mean(w_s >= cap)),
        }

    support = {
        'hard_positivity_violations': hard_violations,
        'zero_weight_episode_fraction': zero_weight_episodes / n if n else None,
        'max_per_step_weight': per_step_max,
        'max_episode_weight': float(np.max(w_s)) if n_settled else None,
        'ess': ess,
        'ess_ratio': ess / n_settled if n_settled else None,
        'n_settled_episodes': n_settled,
        'n_censored_episodes': n_censored,
    }
    # 部分轨迹权重为零是确定性候选的正常现象：只要候选要求的动作在日志中都有
    # 正概率（无硬正性违反），SNIPW 就在日志分布的支持子集上识别候选价值。
    # 全部轨迹权重为零才是完全无支持；部分缺口由 ESS 与 zero_weight_episode_fraction 把关。
    no_support_at_all = zero_weight_episodes == n
    identifiable = (not hard_violations and not no_support_at_all and ess >= MIN_ESS and
                    (support['ess_ratio'] or 0.0) >= MIN_ESS_RATIO and
                    (support['max_episode_weight'] or 0.0) <= MAX_WEIGHT_CAP)

    # 用户簇配对 bootstrap（估计单元是整段轨迹；同用户的重复请求不拆开）
    boot_sn, boot_ip, boot_base, boot_diff, boot_max_weight = [], [], [], [], []
    for _ in range(N_BOOTSTRAP):
        index_list = _resample_clusters(units, rng)
        ww = np.array([weights[i] for i in index_list])
        yy = np.array([units[i]['reward'] for i in index_list])
        sm = np.array([units[i]['settled'] for i in index_list])
        ww_s, yy_s = ww[sm], yy[sm]
        # 基准（日志事实均值）不依赖候选权重，始终可估计。
        if yy_s.size == 0:
            continue
        base = float(np.mean(yy_s))
        boot_base.append(base)
        if np.sum(ww_s) <= 0:
            continue
        sn = float(np.mean(ww_s * yy_s) / np.mean(ww_s))
        if not (math.isfinite(sn) and math.isfinite(base)):
            continue
        boot_sn.append(sn)
        boot_ip.append(float(np.mean(ww_s * yy_s)))
        boot_diff.append(sn - base)
        boot_max_weight.append(float(np.max(ww_s)))

    sn_terms = np.zeros(n)
    base_terms = np.zeros(n)
    if denom > 0 and n_settled:
        for i in settled_positions:
            sn_terms[i] = weights[i] * (units[i]['reward'] - snipw_value) / denom
            base_terms[i] = units[i]['reward'] - mean_reward

    censor_fraction = n_censored / n if n else 0.0
    if n_censored and n_settled and denom > 0:
        w_cens = weights[~settled_mask]
        cand_low = float((np.sum(w_s * y_s) + np.sum(w_cens) * reward_min) /
                         (np.sum(w_s) + np.sum(w_cens)))
        cand_high = float((np.sum(w_s * y_s) + np.sum(w_cens) * reward_max) /
                          (np.sum(w_s) + np.sum(w_cens)))
        base_low = (n_settled * mean_reward + n_censored * reward_min) / n
        base_high = (n_settled * mean_reward + n_censored * reward_max) / n
    else:
        cand_low, cand_high = snipw_value, snipw_value
        base_low, base_high = mean_reward, mean_reward

    difference = (snipw_value - mean_reward) if snipw_value is not None and mean_reward is not None else None

    return {
        'identifiable': identifiable,
        'support': support,
        'censoring': {
            'n_censored_episodes': n_censored,
            'censor_fraction': censor_fraction,
            'reward_bounds_observed': [reward_min, reward_max],
            'manski_bounds_baseline': [base_low, base_high],
            'manski_bounds_candidate_snipw': [cand_low, cand_high],
            'bounds_straddle_zero': bool(_straddles_zero(base_low, base_high) or
                                        _straddles_zero(cand_low, cand_high)),
        },
        'estimators': {
            'ipw': {'value': ipw_value, 'ci': _ci(boot_ip)},
            'snipw': {'value': snipw_value, 'ci': _ci(boot_sn),
                      'clustered_se': _clustered_se(sn_terms, units, True)},
            'clipped_snipw': clipped,
            'doubly_robust': None,
            'note': '轨迹模式不构造单步收益模型：整段价值只能由联合似然比识别，'
                    '单步概率乘最终收益不是该估计量。',
        },
        'difference_vs_baseline': {
            'method': 'trajectory_snipw_paired_cluster_bootstrap',
            'estimate': difference,
            'ci': _ci(boot_diff),
            'clustered_se': _clustered_se(sn_terms - base_terms, units, True),
        },
        'baseline_bootstrap': {'mean': mean_reward, 'ci': _ci(boot_base),
                               'clustered_se': _clustered_se(base_terms, units, True)},
        'bootstrap_max_weight_quantiles': {
            'p50': _percentile(boot_max_weight, 50) if boot_max_weight else None,
            'p95': _percentile(boot_max_weight, 95) if boot_max_weight else None,
            'max': float(np.max(boot_max_weight)) if boot_max_weight else None,
        },
    }
