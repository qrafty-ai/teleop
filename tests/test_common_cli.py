from teleop_xr.common_cli import CommonCLI
from teleop_xr.config import InputMode


def test_common_cli_defaults():
    cli = CommonCLI()
    assert cli.host == "0.0.0.0"
    assert cli.port == 4443
    assert cli.input_mode == InputMode.CONTROLLER
    assert cli.double_press_ms == 300
    assert cli.long_press_ms == 1000


def test_common_cli_init():
    cli = CommonCLI(
        host="127.0.0.1",
        port=8000,
        input_mode=InputMode.HAND,
        double_press_ms=250,
        long_press_ms=1500,
    )
    assert cli.host == "127.0.0.1"
    assert cli.port == 8000
    assert cli.input_mode == InputMode.HAND
    assert cli.double_press_ms == 250
    assert cli.long_press_ms == 1500


def test_common_cli_event_settings():
    cli = CommonCLI(double_press_ms=225, long_press_ms=1250)

    settings = cli.event_settings()

    assert settings.double_press_threshold_ms == 225
    assert settings.long_press_threshold_ms == 1250
