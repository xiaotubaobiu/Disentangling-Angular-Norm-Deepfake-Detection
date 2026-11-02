"""Score-blind CDF development isolation on the complete candidate graph."""
import json

import pandas as pd

from journal_detector import data_d


def candidates(specifications):
    """Eight qualified frames plus optional RGB bridges outside fixed-eight."""
    rows = []
    for method, video, group, extras in specifications:
        uid = f"DF40/cdf/{method}/{video}"
        for frame in range(8):
            rows.append(dict(domain="cdf", method_id=method, video_id=video,
                video_uid=uid, source_ids="[]", group_id=group, label=int(method != "real"),
                role="target_full", image_path=f"/{method}/{video}/{frame}.png",
                frame_number=frame, decode_ok=1, rgb_sha256=f"{uid}/{frame}",
                exclusion_reason=""))
        for extra in extras:
            rows.append({**rows[-1], "frame_number": 100, "image_path": f"/{uid}/{extra}.png",
                         "rgb_sha256": extra})
    return pd.DataFrame(rows)


def definition(cal, test, **policy):
    return {"cdf_real_split": {"calibration_pool": cal, "heldout_test": test,
        "source_split_sha256": "historic-split", "source_protocol_sha256": "historic-protocol"},
        "cdf_development_policy": {"real_video_target": 1,
            "fake_videos_per_method_cap": 1, "selection_seed": 20261102, **policy}}


def test_cdf_joint_closure_follows_rgb_then_new_identity_token():
    # Removing the full-candidate RGB edge or stopping after one token hop leaks C.
    full = candidates([
        ("real", "id1_0001", "a", []),
        ("fsgan", "id1_driver", "b", ["unselected-bridge"]),
        ("fsgan", "id2_driver", "c", ["unselected-bridge"]),
        ("simswap", "id2_other", "d", []),
        ("real", "id10_0001", "independent", []),
    ])
    graph = data_d.attach_components(full, {})
    assert "cdf_relation_group_id" in graph, "CDF identity proxies need a joint transitive graph"
    by_video = graph.drop_duplicates("video_id").set_index("video_id")
    assert by_video.loc[["id1_0001", "id1_driver", "id2_driver", "id2_other"],
                        "cdf_relation_group_id"].nunique() == 1
    assert by_video.loc["id10_0001", "cdf_relation_group_id"] != by_video.loc["id1_0001", "cdf_relation_group_id"]


def test_cdf_joint_closure_joins_known_groups_and_same_names_across_methods():
    # Losing known group links or exact same-name links permits method renaming leakage.
    full = candidates([
        ("real", "00021", "source-a", []),
        ("fsgan", "arbitrary", "source-a", []),
        ("simswap", "arbitrary", "source-b", []),
        ("real", "00023", "source-c", []),
    ])
    assert hasattr(data_d, "attach_cdf_relations"), "Known groups and proxy names must share one closure"
    graph = data_d.attach_cdf_relations(full)
    by_uid = graph.drop_duplicates("video_uid").set_index("video_uid")
    assert by_uid.loc[["DF40/cdf/real/00021", "DF40/cdf/fsgan/arbitrary",
                       "DF40/cdf/simswap/arbitrary"], "cdf_relation_group_id"].nunique() == 1
    assert by_uid.loc["DF40/cdf/real/00023", "cdf_relation_group_id"] != by_uid.loc["DF40/cdf/real/00021", "cdf_relation_group_id"]


def test_formal_ranking_and_cal_exclude_entire_development_closure(tmp_path):
    # Returning DEV-linked CAL or an unscored companion to ranking leaks development.
    full = candidates([
        ("real", "00021", "dev-group", []),
        ("real", "00011", "dev-group", []),
        ("fsgan", "development", "dev-group", []),
        ("simswap", "development", "proxy-companion", []),
        ("real", "00048", "cal-group", []),
        ("real", "00023", "test-group", []),
        ("fsgan", "formal", "formal-fake", []),
        ("simswap", "formal", "formal-fake", []),
    ])
    selected = full.copy()
    evidence = data_d.cdf_proxy_split(selected, full,
        definition(["00011", "00048"], ["00021", "00023"],
                   seed_real_video_ids=["00021"]), tmp_path)
    assert (tmp_path / "cdf_dev.parquet").exists(), "Independent CDF-DEV must be materialized"
    dev = pd.read_parquet(tmp_path / "cdf_dev.parquet")
    ranking = pd.read_parquet(tmp_path / "target_full.parquet")
    cal = pd.read_parquet(tmp_path / "target_cal.parquet")
    test = pd.read_parquet(tmp_path / "target_test.parquet")
    forbidden = {"DF40/cdf/real/00021", "DF40/cdf/real/00011",
                 "DF40/cdf/fsgan/development", "DF40/cdf/simswap/development"}
    assert set(dev.video_uid) <= forbidden
    for role in [ranking, cal, test]:
        assert not set(role.video_uid) & forbidden
        assert not set(dev.group_id) & set(role.group_id)
        assert not set(dev.cdf_relation_group_id) & set(role.cdf_relation_group_id)
    assert set(cal.video_id) == {"00048"}
    assert set(ranking.video_id) == {"00048", "00023", "formal"}
    ledger = pd.read_parquet(tmp_path / "cdf_development_exclusions.parquet")
    assert set(ledger.video_uid) == forbidden
    assert ledger.development_group_id.notna().all()
    assert evidence["cal_real_videos"] == 1
    assert evidence["available_target_calibration_budgets"] == []


def test_cal_reallocation_rejects_fake_linked_real_even_for_unselected_rgb(tmp_path):
    # Ignoring an unselected RGB bridge would admit a fake-linked real to CAL.
    full = candidates([
        ("real", "id1_0001", "a", ["cal-bridge"]),
        ("fsgan", "id2_driver", "c", ["cal-bridge"]),
        ("simswap", "id2_other", "d", []),
        ("real", "00021", "dev-real", []),
        ("real", "00023", "formal-real", []),
        ("fsgan", "independent", "formal-fake", []),
        ("simswap", "independent", "formal-fake", []),
    ])
    full = data_d.attach_components(full, {})
    selected = full[full.frame_number.lt(8)].copy()
    data_d.cdf_proxy_split(selected, full,
        definition(["id1_0001"], ["00021", "00023"], seed_real_video_ids=["00021"],
                   fake_videos_per_method_cap=0), tmp_path)
    test = pd.read_parquet(tmp_path / "target_test.parquet")
    cal = pd.read_parquet(tmp_path / "target_cal.parquet")
    assert set(cal.video_id) == {"00023"}
    assert set(test.video_id) == {"id1_0001", "id2_driver", "id2_other", "independent"}
    assert not set(cal.cdf_relation_group_id) & set(test.cdf_relation_group_id)


def test_development_does_not_split_a_real_component_to_fill_quota(tmp_path):
    # Taking one real from a multi-real component to hit the quota silently splits sources.
    full = candidates([
        ("real", "00021", "two-real", []),
        ("real", "00023", "two-real", []),
        ("fsgan", "formal", "formal", []),
    ])
    data_d.cdf_proxy_split(full, full,
        definition([], ["00021", "00023"], fake_videos_per_method_cap=0), tmp_path)
    assert (tmp_path / "cdf_development_scope.json").exists(), "Quota infeasibility must be explicit"
    scope = json.loads((tmp_path / "cdf_development_scope.json").read_text())
    dev = pd.read_parquet(tmp_path / "cdf_dev.parquet")
    assert dev.video_uid.nunique() == 0
    assert scope["real_video_target"] == 1
    assert scope["real_video_actual"] == 0
    assert scope["target_met"] is False


def test_smallest_fake_component_scores_capped_subset_and_quarantines_rest(tmp_path):
    # Capping the scored fake subset must not return its unscored relatives to TEST.
    full = candidates([
        ("real", "00021", "small", []),
        ("fsgan", "development-a", "small", []),
        ("fsgan", "development-b", "small", []),
        ("simswap", "development-a", "small", []),
        ("real", "00023", "large", []),
        ("fsgan", "formal-a", "large", []),
        ("simswap", "formal-a", "large", []),
        ("fsgan", "formal-b", "large", []),
        ("simswap", "formal-b", "large", []),
        ("real", "00011", "cal", []),
    ])
    evidence = data_d.cdf_proxy_split(full, full,
        definition(["00011"], ["00021", "00023"]), tmp_path)
    dev = pd.read_parquet(tmp_path / "cdf_dev.parquet")
    test = pd.read_parquet(tmp_path / "target_test.parquet")
    ledger = pd.read_parquet(tmp_path / "cdf_development_exclusions.parquet")
    assert dev.drop_duplicates('video_uid').groupby('method_id').size().to_dict() == {
        "real": 1, "fsgan": 1, "simswap": 1}
    assert set(ledger.video_id) == {"00021", "development-a", "development-b"}
    assert ledger.video_uid.nunique() == 4
    assert not set(test.video_id) & {"00021", "development-a", "development-b"}
    assert evidence['development']['usable_for_epoch_selection'] is True
    assert evidence['development']['unscored_quarantined_candidate_videos'] == 1
