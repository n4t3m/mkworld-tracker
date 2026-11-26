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
