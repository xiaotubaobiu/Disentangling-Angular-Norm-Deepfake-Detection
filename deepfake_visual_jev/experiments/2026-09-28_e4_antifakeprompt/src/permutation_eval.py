"""Supplementary order-only audit; does not replace the original G3 audit."""
import itertools
import torch


@torch.no_grad()
def fixed_batch_audit(model, features, questions, candidates, mask):
    if candidates.ndim != 3 or mask.shape != candidates.shape[:2]:
        raise ValueError('invalid candidates/mask')
    counts = mask.sum(1)
    if not torch.all((counts == 2) | (counts == 3)):
        raise ValueError('audit supports two/three candidates')
    reference = model.classify_features(features, questions, candidates, mask).float().softmax(-1)
    maximum, consistent, comparisons = 0., 0, 0
    orders = list(itertools.permutations(range(3))) if (counts == 3).any() else [(0, 1, 2), (1, 0, 2)]
    for index, triple in enumerate(orders):
        order = torch.tensor(triple, device=mask.device).repeat(len(mask), 1)
        order[counts == 2] = torch.tensor((0, 1, 2) if index % 2 == 0 else (1, 0, 2), device=mask.device)
        permuted = candidates.gather(1, order[:, :, None].expand_as(candidates))
        probs = model.classify_features(features, questions, permuted, mask).float().softmax(-1)
        remapped = torch.zeros_like(probs).scatter(1, order, probs)
        included = (counts == 3) | (index < 2)
        if not included.any():
            continue
        maximum = max(maximum, float((remapped[included] - reference[included]).abs().max()))
        consistent += int((remapped[included].argmax(-1) == reference[included].argmax(-1)).sum())
        comparisons += int(included.sum())
    return {'max_probability_error': maximum, 'consistent_count': consistent, 'comparison_count': comparisons,
            'prediction_consistency': consistent / comparisons}
