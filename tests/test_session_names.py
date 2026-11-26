from mktracker.detection.session_names import reconcile_names
from mktracker.video_timeline import VideoRace


def test_repeated_previews_repair_unique_noisy_result_names():
    races = [VideoRace(0, preview_players=["Vy Barrett", "Ly Raised", "Vy Azusa"])
             for _ in range(5)]
    races[0].placements = [[1, "Vy Barrette"], [2, "ly Qaised"], [3, "Vy Azusa"]]
    races[0].placement_evidence = {"1": {"status": "confirmed"}}
    assert reconcile_names(races) == 2
    assert races[0].placements == [[1, "Vy Barrett"], [2, "Ly Raised"], [3, "Vy Azusa"]]
    assert races[0].placement_evidence["1"]["raw_name"] == "Vy Barrette"
    assert races[0].placement_evidence["1"]["status"] == "review"
    assert reconcile_names(races) == 0


def test_unknown_and_colliding_names_are_not_rewritten():
    races = [VideoRace(0, preview_players=["Ly Bert", "Vy Axis", "Vy Azusa"])
             for _ in range(5)]
    races[0].placements = [[1, "Unknown Visitor"], [2, "Vy Axix"], [3, "Vy Axis"]]
    assert reconcile_names(races) == 0
    assert races[0].placements == [[1, "Unknown Visitor"], [2, "Vy Axix"], [3, "Vy Axis"]]


def test_one_off_preview_name_cannot_override_result():
    races = [VideoRace(0, preview_players=["One Off"])] + [VideoRace(i) for i in range(1, 5)]
    races[0].placements = [[1, "One Otf"]]
    assert reconcile_names(races) == 0


def test_readable_minor_spelling_does_not_beat_preview_majority():
    races = [VideoRace(i, preview_players=["Vγ Barrett"]) for i in range(4)]
    races += [VideoRace(i + 4, preview_players=["Vy Parrett"]) for i in range(2)]
    races[0].placements = [[1, "Vy Barrette"]]
    assert reconcile_names(races) == 1
    assert races[0].placements == [[1, "Vγ Barrett"]]


def test_clear_alternative_can_resolve_a_short_bad_read():
    races = [VideoRace(i, preview_players=["Vy Will", "Vy Axis"]) for i in range(5)]
    races[0].placements = [[1, "Vy Witi"]]
    races[0].placement_evidence = {"1": {"alternatives": ["Vy Will"]}}
    assert reconcile_names(races) == 1
    assert races[0].placements == [[1, "Vy Will"]]


def test_repeated_result_spelling_resolves_short_name_without_preview():
    races = [VideoRace(i) for i in range(5)]
    for race in races[:4]:
        race.placements = [[1, "uu."], [2, "K4I"]] + [
            [rank, f"Player{rank}"] for rank in range(3, 25)]
    races[4].placements = [[1, "ud."], [2, "K4{"]] + [
        [rank, f"Player{rank}"] for rank in range(3, 25)]
    races[4].placement_evidence = {
        "1": {"status": "review", "alternatives": ["uu."]},
        "2": {"status": "review", "alternatives": ["K4I"]},
    }
    assert reconcile_names(races) == 2
    assert races[4].placements[:2] == [[1, "uu."], [2, "K4I"]]
    assert reconcile_names(races) == 0


def test_rare_name_in_repeated_24_player_results_stays_for_review():
    races = [VideoRace(i) for i in range(4)]
    for race in races:
        race.placements = [[rank, f"Player{rank}"] for rank in range(1, 25)]
        race.placement_evidence = {"6": {"status": "confirmed"}}
    races[0].placements[5][1] = "Ltam"
    reconcile_names(races)
    assert races[0].placement_evidence["6"]["status"] == "review"
    assert races[0].placements[5] == [6, "Ltam"]


def test_repeated_result_name_repairs_unique_long_variant():
    races = [VideoRace(i) for i in range(5)]
    for race in races[:4]:
        race.placements = [[1, "Polar Star"], [2, "torcher"]] + [
            [rank, f"Player{rank}"] for rank in range(3, 25)]
    races[4].placements = [[1, "Polar Sear"], [2, "corcher"]] + [
        [rank, f"Player{rank}"] for rank in range(3, 25)]
    assert reconcile_names(races) == 2
    assert races[4].placements[:2] == [[1, "Polar Star"], [2, "torcher"]]
    assert reconcile_names(races) == 0


def test_missing_24_player_identity_uses_clear_same_rank_alternative():
    races = [VideoRace(i) for i in range(5)]
    for i, race in enumerate(races):
        race.placements = [[rank, f"Player{rank}"] for rank in range(1, 23)]
        race.placements += [[23, "K4I"], [24, f"Visitor{i}"]]
    races[4].placements[22][1] = "Ltam"
    races[4].placement_evidence = {
        "23": {"status": "review", "alternatives": ["K4{"]},
        "24": {"status": "review", "alternatives": ["Other"]},
    }
    assert reconcile_names(races) == 1
    assert races[4].placements[22] == [23, "K4I"]
    assert races[4].placement_evidence["23"]["status"] == "review"


def test_highlighted_player_uses_digit_spelling_seen_in_two_races():
    races = [VideoRace(i) for i in range(4)]
    variants = ["Liam", "L1am", "Lfam", "t ian"]
    ranks = [1, 2, 3, 4]
    for race, name, rank in zip(races, variants, ranks):
        race.placements = [[n, f"Player{n}"] for n in range(1, 25)]
        race.placements[rank - 1][1] = name
        race.highlighted_rank = rank
        race.placement_evidence = {str(rank): {"status": "review"}}
    races[0].placement_evidence["1"]["alternatives"] = ["L1am"]
    assert reconcile_names(races) == 3
    assert [dict(race.placements)[rank] for race, rank in zip(races, ranks)] == ["L1am"] * 4


def test_result_roster_inference_does_not_change_12_player_session():
    races = [VideoRace(i) for i in range(5)]
    for race in races[:4]:
        race.placements = [[rank, f"Player{rank}"] for rank in range(1, 12)] + [
            [12, "Polar Star"]]
    races[4].placements = [[rank, f"Player{rank}"] for rank in range(1, 12)] + [
        [12, "Polar Sear"]]
    assert reconcile_names(races) == 0
    assert races[4].placements[-1] == [12, "Polar Sear"]


def test_complete_24_player_roster_resolves_two_unique_variants():
    races = [VideoRace(i) for i in range(5)]
    for race in races:
        race.placements = [[1, "L1am"], [2, "uu."]] + [
            [rank, f"Player{rank}"] for rank in range(3, 25)]
    races[4].placements[0][1] = "Lfam"
    races[4].placements[1][1] = "aud."
    assert reconcile_names(races) == 2
    assert races[4].placements[:2] == [[1, "L1am"], [2, "uu."]]
    assert reconcile_names(races) == 0


def test_repeated_digit_alternative_corrects_short_roster_name():
    races = [VideoRace(i) for i in range(5)]
    for race in races:
        race.placements = [[1, "Lidz"]] + [
            [rank, f"Player{rank}"] for rank in range(2, 25)]
    for race in races[:3]:
        race.placement_evidence = {"1": {"alternatives": ["Li4z"]}}
    assert reconcile_names(races) == 5
    assert all(race.placements[0] == [1, "Li4z"] for race in races)
    assert reconcile_names(races) == 0


def test_single_digit_alternative_does_not_change_short_name():
    races = [VideoRace(i) for i in range(5)]
    for race in races:
        race.placements = [[1, "Lidz"]] + [
            [rank, f"Player{rank}"] for rank in range(2, 25)]
    races[0].placement_evidence = {"1": {"alternatives": ["Li4z"]}}
    assert reconcile_names(races) == 0
    assert all(race.placements[0] == [1, "Lidz"] for race in races)
