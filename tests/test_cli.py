from __future__ import annotations

import json
from pathlib import Path

from trilateration.cli import main


def test_cli_solves_json_example(capsys) -> None:
    exit_code = main(["solve", "examples/anchors.json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert abs(payload["lat_deg"] - 50.4529) < 1e-9
    assert abs(payload["lon_deg"] - 30.5268) < 1e-9
    assert payload["alt_m"] == 183.5


def test_cli_solves_csv_example(capsys) -> None:
    exit_code = main(["solve", "examples/anchors.csv", "--target-alt", "183.5", "--input-format", "csv"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert abs(payload["lat_deg"] - 50.4529) < 1e-9
    assert abs(payload["lon_deg"] - 30.5268) < 1e-9


def test_cli_reports_error_for_missing_target_alt(tmp_path: Path, capsys) -> None:
    path = tmp_path / "anchors.json"
    path.write_text("[]", encoding="utf-8")

    exit_code = main(["solve", str(path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert "error" in payload


def test_cli_writes_plot(tmp_path: Path, capsys) -> None:
    output = tmp_path / "plot.png"

    exit_code = main(["plot", "examples/anchors.json", "--output", str(output)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["output"] == str(output)
    assert output.read_bytes().startswith(b"\x89PNG")
