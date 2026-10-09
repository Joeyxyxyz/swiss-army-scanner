"""Recipe library (engine/recipes/library.py), MOCK only: selection, names and history,
the same rules as the preset library (test_presets_lib)."""

import json

import pytest

from engine import api
from engine.recipes.schema import RecipeError
from ui import cli

from conftest import ROOT


@pytest.fixture
def eng(tmp_path):
    """Mock engine on a throwaway library with the repo's presets (the recipes pin them)."""
    s = api.configure(tmp_path / "library", mock=True)
    yield s
    api._session = None


def test_select_refuses_a_recipe_file_and_keeps_the_selection(eng, tmp_path, capsys):
    """Only a built-in or library recipe name can be selected: a file path used to be
    recorded, report success and silently clear the selection (and its delete guard)."""
    api.recipe_add("my_lid", "box_lid")
    api.recipe_select("my_lid")
    with pytest.raises(RecipeError, match="only a built-in or library recipe"):
        api.recipe_select(str(ROOT / "recipes" / "flat_plate.json"))
    assert api.recipe_active() == "my_lid"
    with pytest.raises(RecipeError, match="selected recipe"):
        api.recipe_delete("my_lid", "x")                 # the guard still holds
    api._session = None
    lib = str(tmp_path / "cli_lib")
    assert cli.main(["--mock", "--library", lib, "recipe-select",
                     str(ROOT / "recipes" / "flat_plate.json")]) == cli.EXIT_REFUSED
    assert "only a built-in or library recipe" in capsys.readouterr().err


def test_unreadable_recipe_selection_lines_are_skipped(eng):
    api.recipe_select("box_lid")
    f = eng.store.root / "recipes" / "active.jsonl"
    with open(f, "ab") as fh:
        fh.write(b'{"name": "caf\xe9"}\n\xff\xfe{\x00}\x00')
    assert api.recipe_active() == "box_lid"
    api.recipe_select("flat_plate")                      # appends on a line of its own
    assert api.recipe_active() == "flat_plate"
    assert json.loads(f.read_bytes().splitlines()[-1])["name"] == "flat_plate"


@pytest.mark.parametrize("name", ["CON", "nul", "com3", "lid.", "...", "lid.json",
                                  "FLAT_PLATE"])
def test_recipe_names_portable_and_no_case_clash(eng, name):
    with pytest.raises(RecipeError):
        api.recipe_add(name, "box_lid")
    assert name not in [i["name"] for i in api.recipe_list()]


@pytest.mark.parametrize("value", ["NaN", "Infinity", "1e999"])
def test_recipe_non_finite_override_refused(eng, value):
    with pytest.raises(RecipeError, match="finite"):
        api.recipe_add("lid_bad", "box_lid", {"feedback.hold_s": value})
