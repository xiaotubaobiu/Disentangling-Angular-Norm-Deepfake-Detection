"""Describe grouping changes as reweighting and score averaging, using fixed outputs.

Four analytical states separate group weighting from replacing member scores by
their mean. Symmetric contributions average both possible orders; they describe
this declared factorization, not a unique causal attribution to training.
"""
from bisect import bisect_left, bisect_right
from fractions import Fraction
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import mpmath as mp
import pandas as pd
from journal_detector import analysis_d, trainer

ROOT = Path('/home/z/workspace/deepfake_journal_20261102')
RUNS = ['D_formal_clip_vit_b16_seed123', 'D_formal_clip_vit_b16_seed2026']


def twice_pair_credit(value, sorted_real):
    # 2 for each correctly ordered pair, 1 for a tie, 0 for an incorrect pair.
    return bisect_left(sorted_real, value) + bisect_right(sorted_real, value)


def decompose(run, role, receipt):
    folder = ROOT / 'analysis/D' / run
    paths = [folder / role / name for name in ['video_scores.csv', 'metrics.csv', 'effects.csv']]
    for path in paths:
        assert trainer.digest(path) == receipt['artifact_sha256'][str(path.relative_to(folder))]
    videos = analysis_d.read_scores(paths[0])
    metrics = pd.read_csv(paths[1], float_precision='round_trip')
    effects = pd.read_csv(paths[2], float_precision='round_trip').set_index('aggregation')
    rows, distributions = [], []
    with mp.workdps(400):
        for aggregation in ['mean_logit', 'mean_probability']:
            for representation in ['raw', 'l2']:
                subset = videos[videos.aggregation.eq(aggregation) & videos.representation.eq(representation)]
                natural = subset[subset.grouping.eq('correct')]
                collision = subset[subset.grouping.eq('collision')]
                assert natural.n_frames.eq(8).all()
                real = natural[natural.label.eq(0)].set_index('video_uid')
                other_real = collision[collision.label.eq(0)].set_index('video_uid')
                assert set(real.index) == set(other_real.index)
                real_maximum_difference = max(abs(real.loc[uid, 'rank_score'] - other_real.loc[uid, 'rank_score']) for uid in real.index)
                assert real_maximum_difference < mp.mpf('1e-12')
                sorted_real = sorted(real.rank_score.tolist())
                other_sorted_real = sorted(other_real.rank_score.tolist())
                fake = natural[natural.label.eq(1)].set_index('video_uid')
                merged = collision[collision.label.eq(1)]
                assert not fake.index.duplicated().any()
                R, N, G = len(real), len(fake), len(merged)
                covered, member_credits, group_mean_credits = set(), [], []
                memberships = []
                max_mean_error = mp.mpf(0)
                for group in merged.itertuples():
                    members = tuple(group.member_video_uids)
                    assert len(members) == len(set(members)) and group.n_frames == 8 * len(members)
                    assert group.n_methods == len(members)
                    assert not covered.intersection(members)
                    covered.update(members)
                    scores = [fake.loc[uid, 'rank_score'] for uid in members]
                    mean = mp.fsum(scores) / len(scores)
                    max_mean_error = max(max_mean_error, abs(mean - group.rank_score))
                    assert abs(mean - group.rank_score) < mp.mpf('1e-12')
                    # The two equivalent score constructions must make identical decisions.
                    mean_credit = twice_pair_credit(mean, sorted_real)
                    assert mean_credit == twice_pair_credit(group.rank_score, sorted_real)
                    assert mean_credit == twice_pair_credit(group.rank_score, other_sorted_real)
                    assert mean_credit == twice_pair_credit(mean, other_sorted_real)
                    member_values = [twice_pair_credit(value, sorted_real) for value in scores]
                    assert member_values == [twice_pair_credit(value, other_sorted_real) for value in scores]
                    member_credit = sum(member_values)
                    memberships.append(len(members))
                    member_credits.append(member_credit)
                    group_mean_credits.append(mean_credit)
                assert covered == set(fake.index) and sum(memberships) == N
                a00 = Fraction(sum(member_credits), 2 * R * N)
                a10 = sum((Fraction(c, n) for c, n in zip(member_credits, memberships)), Fraction(0)) / (2 * R * G)
                a01 = Fraction(sum(n * c for n, c in zip(memberships, group_mean_credits)), 2 * R * N)
                a11 = Fraction(sum(group_mean_credits), 2 * R * G)
                reweight = ((a10 - a00) + (a11 - a01)) / 2
                average = ((a01 - a00) + (a11 - a10)) / 2
                assert reweight + average == a11 - a00
                for grouping, value in [('correct', a00), ('collision', a11)]:
                    reference = metrics[(metrics.grouping == grouping) & (metrics.representation == representation) &
                        (metrics.aggregation == aggregation)].iloc[0]
                    assert reference.real_mass == reference.n_real == R
                    assert reference.fake_mass == reference.n_fake == (N if grouping == 'correct' else G)
                    assert abs(float(value) - reference.pooled_auc) < 1e-12
                exact = {name: {'numerator': str(value.numerator), 'denominator': str(value.denominator)}
                    for name, value in [('A00', a00), ('A10', a10), ('A01', a01), ('A11', a11),
                                        ('symmetric_reweight', reweight), ('symmetric_averaging', average)]}
                rows.append(dict(run_id=run, seed=receipt['seed'], role=role, aggregation=aggregation,
                    representation=representation, n_real=R, natural_fake=N, collision_fake=G,
                    A00_natural_equal=float(a00), A10_group_equal_member_scores=float(a10),
                    A01_member_count_weighted_group_mean=float(a01), A11_group_equal_group_mean=float(a11),
                    grouping_auc_change=float(a11 - a00), symmetric_reweight_contribution=float(reweight),
                    symmetric_averaging_contribution=float(average), exact_fractions=json.dumps(exact),
                    maximum_group_mean_difference=float(max_mean_error),
                    maximum_real_format_difference=float(real_maximum_difference)))
                if aggregation == 'mean_logit' and representation == 'raw':
                    counts = pd.Series(memberships).value_counts().sort_index()
                    distributions.extend(dict(run_id=run, role=role, methods_in_group=int(n),
                        groups=int(count), natural_fake_videos=int(n * count)) for n, count in counts.items())
            pair = {row['representation']: row for row in rows if row['aggregation'] == aggregation}
            interaction = ((pair['l2']['A11_group_equal_group_mean'] - pair['raw']['A11_group_equal_group_mean']) -
                           (pair['l2']['A00_natural_equal'] - pair['raw']['A00_natural_equal']))
            assert abs(interaction - effects.loc[aggregation, 'interaction']) < 1e-12
    return rows, distributions


def main():
    rows, distributions, snapshots = [], [], {}
    for run in RUNS:
        folder = ROOT / 'analysis/D' / run
        marker = folder / 'complete.json'
        snapshots[run] = trainer.digest(marker)
        receipt = json.loads(marker.read_text())
        assert receipt['status'] == 'ANALYSIS_COMPLETE' and receipt['independent_recompute'] == 'PASSED_FOR_ALL_REPORTABLE_CELLS'
        for role in ['source_test', 'target_full']:
            part, count = decompose(run, role, receipt)
            rows.extend(part)
            distributions.extend(count)
        assert trainer.digest(marker) == snapshots[run]
    table = pd.DataFrame(rows)
    assert len(table) == 16
    interaction_rows = []
    for (run, role, aggregation), q in table.groupby(['run_id', 'role', 'aggregation']):
        q = q.set_index('representation')
        interaction = q.loc['l2', 'grouping_auc_change'] - q.loc['raw', 'grouping_auc_change']
        reweight = q.loc['l2', 'symmetric_reweight_contribution'] - q.loc['raw', 'symmetric_reweight_contribution']
        average = q.loc['l2', 'symmetric_averaging_contribution'] - q.loc['raw', 'symmetric_averaging_contribution']
        assert abs(reweight + average - interaction) < 1e-14
        interaction_rows.append(dict(run_id=run, role=role, aggregation=aggregation,
            grouping_by_representation_interaction=interaction, reweight_interaction=reweight,
            averaging_interaction=average))
    outputs = {'grouping_decomposition_cells.csv': table,
        'grouping_decomposition_interactions.csv': pd.DataFrame(interaction_rows),
        'collision_group_member_counts.csv': pd.DataFrame(distributions)}
    directory = ROOT / 'reports/grouping_decomposition'
    directory.mkdir(exist_ok=True)
    for name, data in outputs.items():
        data.to_csv(directory / name, index=False)
    result = dict(status='PASS', created=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        producer_script_sha256=trainer.digest(__file__),
        scope='Descriptive fixed-output metric decomposition for two CLIP seeds only; no training, inference, or recipe change.',
        model_count=2, complete_five_seed_summary=False, analysis_complete_snapshots=snapshots,
        precision='Stored high-precision rank scores with400decimal working precision; pair credits and contribution arithmetic exact rational.',
        factorization='original member-count vs equal-group weighting; separate member scores vs their group mean',
        contribution_definition='Average the contribution over the two possible factor orders.',
        original_correct_and_collision_AUC_reconstructed=True,
        all_group_means_and_memberships_verified=True,
        different_saved_real_scalar_formats_allowed_only_when_every_pair_credit_is_unchanged=True,
        exact_contributions_sum_to_total=True,
        not_unique_training_causal_attribution=True,
        output_sha256={name: trainer.digest(directory / name) for name in outputs})
    trainer.save_json(ROOT / 'operations/grouping_decomposition_two_CLIP_seeds.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
