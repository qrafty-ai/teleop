import pytest
import numpy as np
from dataclasses import is_dataclass
from unittest.mock import MagicMock
from teleop_xr.events import (
    XRButton,
    XRController,
    ButtonEvent,
    ButtonEventType,
    EventSettings,
    button_index_to_enum,
    ButtonDetector,
    EventProcessor,
)


class TestXRButtonEnum:
    def test_all_values_exist(self):
        assert XRButton.TRIGGER == "trigger"
        assert XRButton.SQUEEZE == "squeeze"
        assert XRButton.TOUCHPAD == "touchpad"
        assert XRButton.THUMBSTICK == "thumbstick"
        assert XRButton.BUTTON_PRIMARY == "button_primary"
        assert XRButton.BUTTON_SECONDARY == "button_secondary"
        assert XRButton.MENU == "menu"

    def test_enum_count(self):
        assert len(XRButton) == 7

    def test_is_str_enum(self):
        for button in XRButton:
            assert isinstance(button.value, str)
            assert isinstance(button, str)


class TestXRControllerEnum:
    def test_left_right_values(self):
        assert XRController.LEFT == "left"
        assert XRController.RIGHT == "right"

    def test_enum_count(self):
        assert len(XRController) == 2

    def test_is_str_enum(self):
        for controller in XRController:
            assert isinstance(controller.value, str)
            assert isinstance(controller, str)


class TestButtonEvent:
    def test_creation_with_all_fields(self):
        event = ButtonEvent(
            type=ButtonEventType.BUTTON_DOWN,
            button=XRButton.TRIGGER,
            controller=XRController.LEFT,
            timestamp_ms=1000.0,
            hold_duration_ms=500.0,
        )
        assert event.type == ButtonEventType.BUTTON_DOWN
        assert event.button == XRButton.TRIGGER
        assert event.controller == XRController.LEFT
        assert event.timestamp_ms == 1000.0
        assert event.hold_duration_ms == 500.0

    def test_is_dataclass(self):
        assert is_dataclass(ButtonEvent)


class TestEventSettings:
    def test_default_values(self):
        settings = EventSettings()
        assert settings.double_press_threshold_ms == 300
        assert settings.long_press_threshold_ms == 1000

    def test_custom_values(self):
        settings = EventSettings(
            double_press_threshold_ms=200,
            long_press_threshold_ms=1000,
        )
        assert settings.double_press_threshold_ms == 200
        assert settings.long_press_threshold_ms == 1000

    def test_is_dataclass(self):
        assert is_dataclass(EventSettings)


class TestButtonIndexToEnum:
    def test_valid_indices(self):
        assert button_index_to_enum(0) == XRButton.TRIGGER
        assert button_index_to_enum(1) == XRButton.SQUEEZE
        assert button_index_to_enum(2) == XRButton.TOUCHPAD
        assert button_index_to_enum(3) == XRButton.THUMBSTICK
        assert button_index_to_enum(4) == XRButton.BUTTON_PRIMARY
        assert button_index_to_enum(5) == XRButton.BUTTON_SECONDARY
        assert button_index_to_enum(6) == XRButton.MENU

    def test_invalid_indices_return_none(self):
        assert button_index_to_enum(-1) is None
        assert button_index_to_enum(7) is None


class TestButtonDetector:
    @pytest.fixture
    def detector(self):
        settings = EventSettings(
            double_press_threshold_ms=300,
            long_press_threshold_ms=1000,
        )
        return ButtonDetector(settings)

    def test_button_down_detection(self, detector):
        events = detector.update(XRButton.TRIGGER, XRController.LEFT, True, 1000.0)
        assert len(events) == 1
        assert events[0].type == ButtonEventType.BUTTON_DOWN
        assert events[0].button == XRButton.TRIGGER
        assert events[0].controller == XRController.LEFT
        assert events[0].timestamp_ms == 1000.0

        events = detector.update(XRButton.TRIGGER, XRController.LEFT, True, 1010.0)
        assert len(events) == 0

    def test_button_up_with_duration(self, detector):
        detector.update(XRButton.TRIGGER, XRController.LEFT, True, 1000.0)
        events = detector.update(XRButton.TRIGGER, XRController.LEFT, False, 1500.0)

        assert len(events) == 1
        assert events[0].type == ButtonEventType.BUTTON_UP
        assert events[0].button == XRButton.TRIGGER
        assert events[0].controller == XRController.LEFT
        assert events[0].timestamp_ms == 1500.0
        assert events[0].hold_duration_ms == 500.0

        events = detector.update(XRButton.TRIGGER, XRController.LEFT, False, 1510.0)
        assert len(events) == 0

    def test_double_press_detection(self, detector):
        detector.update(XRButton.TRIGGER, XRController.LEFT, True, 1000.0)
        detector.update(XRButton.TRIGGER, XRController.LEFT, False, 1100.0)

        events = detector.update(XRButton.TRIGGER, XRController.LEFT, True, 1200.0)

        assert any(e.type == ButtonEventType.BUTTON_DOWN for e in events)
        assert any(e.type == ButtonEventType.DOUBLE_PRESS for e in events)

    def test_double_press_debounce(self, detector):
        detector.update(XRButton.TRIGGER, XRController.LEFT, True, 1000.0)
        detector.update(XRButton.TRIGGER, XRController.LEFT, False, 1100.0)
        detector.update(XRButton.TRIGGER, XRController.LEFT, True, 1200.0)
        detector.update(XRButton.TRIGGER, XRController.LEFT, False, 1300.0)

        events = detector.update(XRButton.TRIGGER, XRController.LEFT, True, 1400.0)
        assert not any(e.type == ButtonEventType.DOUBLE_PRESS for e in events)

    def test_long_press_detection(self, detector):
        detector.update(XRButton.TRIGGER, XRController.LEFT, True, 1000.0)

        events = detector.update(XRButton.TRIGGER, XRController.LEFT, True, 1900.0)
        assert len(events) == 0

        events = detector.update(XRButton.TRIGGER, XRController.LEFT, True, 2000.0)
        assert len(events) == 1
        assert events[0].type == ButtonEventType.LONG_PRESS

        events = detector.update(XRButton.TRIGGER, XRController.LEFT, True, 2100.0)
        assert len(events) == 0

    def test_independent_tracking(self, detector):
        detector.update(XRButton.TRIGGER, XRController.LEFT, True, 1000.0)
        detector.update(XRButton.TRIGGER, XRController.LEFT, False, 1100.0)

        events = detector.update(XRButton.TRIGGER, XRController.RIGHT, True, 1200.0)
        assert not any(e.type == ButtonEventType.DOUBLE_PRESS for e in events)
        assert any(e.type == ButtonEventType.BUTTON_DOWN for e in events)


class TestEventProcessor:
    @pytest.fixture
    def processor(self):
        settings = EventSettings()
        return EventProcessor(settings)

    def test_callback_registration(self, processor):
        cb = MagicMock()
        processor.on_button_down(XRButton.TRIGGER, XRController.LEFT, cb)
        processor.on_button_up(callback=cb)
        processor.on_double_press(button=XRButton.BUTTON_PRIMARY, callback=cb)
        processor.on_long_press(controller=XRController.RIGHT, callback=cb)

    def test_callback_invocation_filtering(self, processor):
        cb_trigger_left = MagicMock()
        cb_any_right = MagicMock()
        cb_primary_any = MagicMock()
        cb_wildcard = MagicMock()

        processor.on_button_down(XRButton.TRIGGER, XRController.LEFT, cb_trigger_left)
        processor.on_button_down(controller=XRController.RIGHT, callback=cb_any_right)
        processor.on_button_down(
            button=XRButton.BUTTON_PRIMARY, callback=cb_primary_any
        )
        processor.on_button_down(callback=cb_wildcard)

        event = ButtonEvent(
            type=ButtonEventType.BUTTON_DOWN,
            button=XRButton.TRIGGER,
            controller=XRController.LEFT,
            timestamp_ms=1000.0,
        )
        processor._fire_callbacks(event)
        cb_trigger_left.assert_called_once_with(event)
        cb_wildcard.assert_called_once_with(event)
        cb_any_right.assert_not_called()
        cb_primary_any.assert_not_called()

        cb_trigger_left.reset_mock()
        cb_wildcard.reset_mock()

        event = ButtonEvent(
            type=ButtonEventType.BUTTON_DOWN,
            button=XRButton.SQUEEZE,
            controller=XRController.RIGHT,
            timestamp_ms=1100.0,
        )
        processor._fire_callbacks(event)
        cb_any_right.assert_called_once_with(event)
        cb_wildcard.assert_called_once_with(event)
        cb_trigger_left.assert_not_called()
        cb_primary_any.assert_not_called()

    def test_process_xr_state(self, processor):
        cb = MagicMock()
        processor.on_button_down(XRButton.TRIGGER, XRController.LEFT, cb)

        xr_state = {
            "timestamp_unix_ms": 2000.0,
            "devices": [
                {
                    "role": "controller",
                    "handedness": "left",
                    "gamepad": {
                        "buttons": [
                            {"pressed": True, "touched": True, "value": 1.0},
                            {"pressed": False, "touched": False, "value": 0.0},
                        ],
                        "axes": [0.0, 0.0],
                    },
                }
            ],
        }

        processor.process(np.eye(4), xr_state)
        cb.assert_called_once()
        event = cb.call_args[0][0]
        assert event.type == ButtonEventType.BUTTON_DOWN
        assert event.button == XRButton.TRIGGER
        assert event.controller == XRController.LEFT
        assert event.timestamp_ms == 2000.0
