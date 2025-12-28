import pytest
import yaml

from modssc.preprocess.plan import PreprocessPlan, StepConfig, dump_plan, load_plan


def test_load_plan_valid(tmp_path):
    """Test loading a valid plan."""
    content = """
    output_key: features.Z
    steps:
      - id: step1
        params: {a: 1}
        modalities: [text]
        requires_fields: [raw.text]
        enabled: true
      - id: step2
    """
    p = tmp_path / "plan.yaml"
    p.write_text(content)

    plan = load_plan(p)
    assert plan.output_key == "features.Z"
    assert len(plan.steps) == 2

    s1 = plan.steps[0]
    assert s1.step_id == "step1"
    assert s1.params == {"a": 1}
    assert s1.modalities == ("text",)
    assert s1.requires_fields == ("raw.text",)
    assert s1.enabled is True

    s2 = plan.steps[1]
    assert s2.step_id == "step2"
    assert s2.params == {}
    assert s2.modalities == ()
    assert s2.requires_fields == ()
    assert s2.enabled is True


def test_load_plan_invalid_root(tmp_path):
    """Test loading a plan that is not a mapping."""
    p = tmp_path / "plan.yaml"
    p.write_text("- list item")
    with pytest.raises(ValueError, match="must contain a mapping"):
        load_plan(p)


def test_load_plan_invalid_steps_type(tmp_path):
    """Test loading a plan where steps is not a sequence."""
    p = tmp_path / "plan.yaml"
    p.write_text("steps: 123")
    with pytest.raises(ValueError, match="'steps' must be a sequence"):
        load_plan(p)


def test_load_plan_invalid_step_item(tmp_path):
    """Test loading a plan where a step is not a mapping."""
    p = tmp_path / "plan.yaml"
    p.write_text("steps: ['not_a_mapping']")
    with pytest.raises(ValueError, match="Each step must be a mapping"):
        load_plan(p)


def test_load_plan_missing_id(tmp_path):
    """Test loading a plan where a step is missing id."""

    p = tmp_path / "plan.yaml"
    p.write_text("steps: [{id: '', step_id: ''}]")
    with pytest.raises(ValueError, match="Each step must define 'id'"):
        load_plan(p)


def test_load_plan_invalid_params(tmp_path):
    """Test loading a plan where params is not a mapping."""

    p = tmp_path / "plan.yaml"
    p.write_text("steps: [{id: s1, params: [1]}]")
    with pytest.raises(ValueError, match="params for 's1' must be a mapping"):
        load_plan(p)


def test_dump_plan(tmp_path):
    """Test dumping a plan."""
    plan = PreprocessPlan(output_key="out", steps=(StepConfig(step_id="s1", params={"k": "v"}),))
    p = tmp_path / "out.yaml"
    dump_plan(plan, p)

    loaded = yaml.safe_load(p.read_text())
    assert loaded["output_key"] == "out"
    assert len(loaded["steps"]) == 1
    assert loaded["steps"][0]["id"] == "s1"
    assert loaded["steps"][0]["params"] == {"k": "v"}
