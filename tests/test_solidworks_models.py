import pytest
from pydantic import ValidationError

from ragtime.indexer.models import SHELL_TOOL_TYPES, SolidworksConnectionConfig, ToolType


def test_solidworks_type_and_config_defaults():
    assert ToolType("solidworks") is ToolType.SOLIDWORKS
    assert ToolType.SOLIDWORKS not in SHELL_TOOL_TYPES
    config = SolidworksConnectionConfig(host="example", user="alice")
    assert (config.solidworks_version, config.instance_mode, config.max_instances, config.idle_shutdown_seconds) == (None, "background", 1, 900)
    assert config.pdm_tool_id is None
    assert config.pdm_approved_folders == []


def test_solidworks_auto_update_host_defaults_on_and_round_trips():
    assert SolidworksConnectionConfig(host="example", user="alice").auto_update_host is True
    config = SolidworksConnectionConfig.model_validate({"host": "example", "user": "alice", "auto_update_host": False})
    assert config.auto_update_host is False
    assert config.model_dump()["auto_update_host"] is False


def test_solidworks_pdm_folders_normalized():
    config = SolidworksConnectionConfig(host="example", user="alice", pdm_tool_id="pdm", pdm_approved_folders=["/Design/Parts"])
    assert config.pdm_approved_folders == [r"\Design\Parts"]


@pytest.mark.parametrize("folders", [[r"\ "], ["\\"], [r"\A", r"\a"], [r"\A", r"\A\B"], ["relative"], [rf"\A{i}" for i in range(26)]])
def test_solidworks_pdm_folders_reject_invalid(folders):
    with pytest.raises(ValidationError):
        SolidworksConnectionConfig(host="example", user="alice", pdm_approved_folders=folders)


@pytest.mark.parametrize(
    ("field", "lower", "upper"),
    [("solidworks_version", 2022, 2035), ("max_instances", 1, 4), ("idle_shutdown_seconds", 60, 86400)],
)
def test_solidworks_config_ranges(field, lower, upper):
    for value in (lower, upper):
        assert getattr(SolidworksConnectionConfig(host="example", user="alice", **{field: value}), field) == value
    for value in (lower - 1, upper + 1):
        with pytest.raises(ValidationError):
            SolidworksConnectionConfig(host="example", user="alice", **{field: value})


def test_solidworks_config_modes_and_ssh_fields():
    with pytest.raises(ValidationError):
        SolidworksConnectionConfig(host="example", user="alice", instance_mode="invalid")  # pyright: ignore[reportArgumentType] - deliberately invalid mode
    config = SolidworksConnectionConfig(
        host="example",
        user="alice",
        instance_mode="attach",
        password="password",
        key_content="key",
        key_passphrase="passphrase",
        pairing_id="pair",
        transport="tunnel",
        host_keys=["host-key"],
    )
    assert (config.password, config.key_content, config.key_passphrase) == ("password", "key", "passphrase")
    assert (config.pairing_id, config.transport, config.host_keys) == ("pair", "tunnel", ["host-key"])
