import math
from collections import defaultdict


def validate_decisions(rows):
    seen = set()
    sequences = defaultdict(list)
    for row in rows:
        if row['decision_id'] in seen:
            raise ValueError('决策编号重复')
        seen.add(row['decision_id'])
        actions = set(row['available'])
        probabilities = row['propensity']
        if not actions or set(probabilities) != actions or row['action'] not in actions:
            raise ValueError('动作集合不一致')
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities.values()):
            raise ValueError('动作概率越界')
        if not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-9):
            raise ValueError('动作概率之和不为一')
        if probabilities[row['action']] <= 0:
            raise ValueError('实际动作缺少正曝光概率')
        reward = row.get('reward')
        if reward is not None and not math.isfinite(reward):
            raise ValueError('收益不是有限数')
        sequences[row['request_id']].append(row)
    for sequence in sequences.values():
        if sorted(row['step'] for row in sequence) != list(range(len(sequence))):
            raise ValueError('请求步骤不连续')
        if len({row['user_id'] for row in sequence}) != 1:
            raise ValueError('请求的用户归属不一致')
    return rows


def factual_summary(document):
    rows = validate_decisions(document['decisions'])
    groups = defaultdict(lambda: {'decisions': 0, 'settled': 0, 'reward_sum': 0.0})
    pending = 0
    for row in rows:
        group = groups[row['action']]
        group['decisions'] += 1
        if row.get('reward') is None or row['reward_known_at'] > document['cutoff']:
            pending += 1
            continue
        group['settled'] += 1
        group['reward_sum'] += row['reward']
    for group in groups.values():
        group['observed_mean'] = group['reward_sum'] / group['settled'] if group['settled'] else None
    return {'actions': dict(sorted(groups.items())), 'pending': pending,
            'users': len({row['user_id'] for row in rows}), 'requests': len({row['request_id'] for row in rows})}


def inspect(document):
    return factual_summary(document)
