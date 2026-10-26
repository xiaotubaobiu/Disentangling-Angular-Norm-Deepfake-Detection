"""Additional, preregistered E3 gates; never used for model selection."""
from dataclasses import asdict

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from evaluation.e2_gates import DOMAINS, SEEDS, paired_seed_video_bootstrap


def _seeds(frame):
    if frame.empty or set(frame.seed) != set(SEEDS):
        raise ValueError('all three preregistered seeds are required')


def _ba(frame, task):
    if task == 'T5':
        classes = np.array(['camera', 'swap', 'reenact'])
        truth = frame.method_family.to_numpy()
        if set(truth) != set(classes):
            raise ValueError('composition family gate needs all three classes')
        probabilities = frame[[f'p_{c}' for c in classes]].to_numpy(float)
        predicted = classes[probabilities.argmax(1)]
    else:
        truth = frame.label.to_numpy(int)
        if set(truth) != {0, 1}:
            raise ValueError('binary gate needs both physical labels')
        probabilities = frame.prob_fake.to_numpy(float)
        predicted = probabilities >= .5
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError('gate input has invalid probabilities')
    return float(balanced_accuracy_score(truth, predicted))


def negation_results(frame, config):
    rows = []
    spec = config['negation']
    pairs = sorted({q['pair'] for q in spec['questions']})
    for domain in DOMAINS:
        domain_rows = frame[frame.domain.eq(domain)]
        _seeds(domain_rows)
        for question in spec['questions']:
            if not question['negated']:
                continue
            subset = domain_rows[domain_rows.question_id.eq(question['id'])]
            _seeds(subset)
            scores = [_ba(subset[subset.seed.eq(seed)], 'T2') for seed in SEEDS]
            threshold = spec['requirements']['negative_question_balanced_accuracy_min']
            rows.append({'domain': domain, 'question_id': question['id'], 'metric': 'negative_balanced_accuracy',
                         'per_seed': scores, 'value': float(np.mean(scores)), 'minimum': threshold,
                         'passed': bool(np.mean(scores) >= threshold)})
        for pair in pairs:
            positive = next(q for q in spec['questions'] if q['pair'] == pair and not q['negated'])
            negative = next(q for q in spec['questions'] if q['pair'] == pair and q['negated'])
            scores, errors, positive_bas = [], [], []
            for seed in SEEDS:
                subset = domain_rows[domain_rows.seed.eq(seed)]
                left = subset[subset.question_id.eq(positive['id'])]
                right = subset[subset.question_id.eq(negative['id'])]
                joined = left.merge(right, on=['seed', 'domain', 'video_key', 'label'],
                                    suffixes=('_positive', '_negative'), how='outer',
                                    validate='one_to_one', indicator=True)
                if joined.empty or not joined._merge.eq('both').all():
                    raise ValueError('negation questions do not cover identical physical videos')
                values = joined[['p_yes_positive', 'p_yes_negative']].to_numpy(float)
                if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                    raise ValueError('invalid negation probabilities')
                scores.append(float(np.mean((values[:, 0] >= .5) != (values[:, 1] >= .5))))
                errors.append(float(np.mean(np.abs(values.sum(1) - 1))))
                positive_bas.append(_ba(left, 'T2'))
            threshold = spec['requirements']['positive_negative_hard_answer_complement_consistency_min']
            rows.append({'domain': domain, 'pair': pair, 'metric': 'complement_consistency',
                         'per_seed': scores, 'value': float(np.mean(scores)), 'minimum': threshold,
                         'mean_absolute_probability_complement_error': float(np.mean(errors)),
                         'positive_question_balanced_accuracy': float(np.mean(positive_bas)),
                         'passed': bool(np.mean(scores) >= threshold)})
    return rows


def composition_results(frame, heldout, config):
    rows = []
    requirements = config['composition']['requirements']
    for domain in DOMAINS:
        for task in ('T1', 'T5'):
            changed = frame[frame.domain.eq(domain) & frame.task_id.eq(task)]
            original = heldout[heldout.domain.eq(domain) & heldout.task_id.eq(task)]
            _seeds(changed)
            _seeds(original)
            scores = [_ba(changed[changed.seed.eq(seed)], task) for seed in SEEDS]
            references = [_ba(original[original.seed.eq(seed)], task) for seed in SEEDS]
            minimum = requirements['T1_balanced_accuracy_min' if task == 'T1' else 'T5_macro_balanced_accuracy_min']
            drop = float(np.mean(references) - np.mean(scores))
            rows.append({'domain': domain, 'task_id': task, 'metric': 'balanced_accuracy' if task == 'T1' else 'macro_balanced_accuracy',
                         'per_seed': scores, 'heldout_per_seed': references, 'value': float(np.mean(scores)),
                         'minimum': minimum, 'heldout_drop': drop,
                         'heldout_drop_max': requirements['per_task_drop_from_original_heldout_max'],
                         'passed': bool(np.mean(scores) >= minimum and drop <= requirements['per_task_drop_from_original_heldout_max'])})
    return rows


def matched_control_results(frame, config, *, n_resamples=10000):
    if n_resamples != 10000:
        raise ValueError('final E3 matched controls require 10000 bootstrap draws')
    spec = config['length_matched_controls']
    rows = []
    for domain in DOMAINS:
        for kind, tasks in [('candidates', ('T1', 'T2', 'T3', 'T4', 'T5')), ('questions', ('T2', 'T4'))]:
            condition = f'meaningless_matched_{kind}'
            for task in tasks:
                control = frame[frame.domain.eq(domain) & frame.condition.eq(condition) & frame.task_id.eq(task)].copy()
                _seeds(control)
                meaningful = control.copy()
                for name in control.columns:
                    if name.startswith('meaningful_'):
                        meaningful[name.removeprefix('meaningful_')] = control[name]
                # The supplemental protocol uses BA for every binary task;
                # T2's bootstrap statistic is binary authenticity BA.
                metric = 'T2' if task in ('T1', 'T3') else task
                result = paired_seed_video_bootstrap(control, meaningful, metric,
                    n_resamples=n_resamples, seed=20260922, lower_quantile=.05)
                minimum = spec['requirements']['meaningful_minus_control_balanced_accuracy_min']
                lower_min = spec['requirements']['paired_video_bootstrap_one_sided_95_lower_bound_min_exclusive']
                rows.append({'domain': domain, 'condition': condition, 'task_id': task,
                             'metric': 'macro_balanced_accuracy_drop' if task == 'T5' else 'balanced_accuracy_drop',
                             **asdict(result), 'minimum': minimum, 'lower_min_exclusive': lower_min,
                             'passed': bool(result.point >= minimum and result.lower > lower_min)})
    return rows
