"""XR event system types and utilities.

This module provides enums and dataclasses for XR button events,
including button identification, controller identification, and event data.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable
import numpy as np


class XRButton(str, Enum):
    """XR controller button identifiers following xr-standard gamepad mapping."""

    TRIGGER = "trigger"
    SQUEEZE = "squeeze"
    TOUCHPAD = "touchpad"
    THUMBSTICK = "thumbstick"
    BUTTON_PRIMARY = "button_primary"
    BUTTON_SECONDARY = "button_secondary"
    MENU = "menu"


class XRController(str, Enum):
    """XR controller hand identifiers."""

    LEFT = "left"
    RIGHT = "right"


class ButtonEventType(str, Enum):
    """Types of button events that can be detected."""

    BUTTON_DOWN = "button_down"
    BUTTON_UP = "button_up"
    DOUBLE_PRESS = "double_press"
    LONG_PRESS = "long_press"


@dataclass
class ButtonEvent:
    """Represents a button event with timing information.

    Attributes:
        type: The type of button event.
        button: The button that triggered the event.
        controller: The controller (left/right) that triggered the event.
        timestamp_ms: Unix timestamp in milliseconds when the event occurred.
        hold_duration_ms: Duration the button was held (for release events), or None.
    """

    type: ButtonEventType
    button: XRButton
    controller: XRController
    timestamp_ms: float
    hold_duration_ms: float | None = None


@dataclass
class EventSettings:
    """Configuration settings for event detection.

    Attributes:
        double_press_threshold_ms: Maximum time between presses to count as double-press.
        long_press_threshold_ms: Minimum hold time to count as long-press.
    """

    double_press_threshold_ms: float = 300
    long_press_threshold_ms: float = 1000


@dataclass
class ButtonState:
    """Internal state for tracking a single button.

    Attributes:
        is_pressed: Whether the button is currently pressed.
        press_time_ms: Timestamp of the last press event.
        last_release_time_ms: Timestamp of the last release event.
        long_press_fired: Whether a long-press event has already been fired for the current press.
        double_press_fired: Whether a double-press event was just fired (for debounce).
    """

    is_pressed: bool = False
    press_time_ms: float = 0.0
    last_release_time_ms: float = 0.0
    long_press_fired: bool = False
    double_press_fired: bool = False


class ButtonDetector:
    """Detects complex button events (double-press, long-press) from raw button states."""

    _settings: EventSettings
    _states: dict[tuple[XRButton, XRController], ButtonState]

    def __init__(self, settings: EventSettings):
        """Initialize the ButtonDetector.

        Args:
            settings: Configuration settings for event detection.
        """
        self._settings = settings
        self._states = {}

    def update(
        self,
        button: XRButton,
        controller: XRController,
        is_pressed: bool,
        timestamp_ms: float,
    ) -> list[ButtonEvent]:
        """Update the state of a button and return any detected events.

        Args:
            button: The button being updated.
            controller: The controller the button belongs to.
            is_pressed: The current raw pressed state of the button.
            timestamp_ms: The timestamp of the update in milliseconds.

        Returns:
            A list of ButtonEvent objects detected during this update.
        """
        key = (button, controller)
        if key not in self._states:
            self._states[key] = ButtonState()

        state = self._states[key]
        events: list[ButtonEvent] = []

        if is_pressed and not state.is_pressed:
            state.is_pressed = True
            state.press_time_ms = timestamp_ms
            state.long_press_fired = False
            state.double_press_fired = False

            events.append(
                ButtonEvent(
                    type=ButtonEventType.BUTTON_DOWN,
                    button=button,
                    controller=controller,
                    timestamp_ms=timestamp_ms,
                )
            )

            time_since_last_release = timestamp_ms - state.last_release_time_ms
            if (
                state.last_release_time_ms > 0
                and time_since_last_release < self._settings.double_press_threshold_ms
            ):
                state.double_press_fired = True
                events.append(
                    ButtonEvent(
                        type=ButtonEventType.DOUBLE_PRESS,
                        button=button,
                        controller=controller,
                        timestamp_ms=timestamp_ms,
                    )
                )

        elif not is_pressed and state.is_pressed:
            state.is_pressed = False
            hold_duration = timestamp_ms - state.press_time_ms

            if state.double_press_fired:
                state.last_release_time_ms = 0.0
            else:
                state.last_release_time_ms = timestamp_ms

            events.append(
                ButtonEvent(
                    type=ButtonEventType.BUTTON_UP,
                    button=button,
                    controller=controller,
                    timestamp_ms=timestamp_ms,
                    hold_duration_ms=hold_duration,
                )
            )

        elif is_pressed and state.is_pressed and not state.long_press_fired:
            hold_duration = timestamp_ms - state.press_time_ms
            if hold_duration >= self._settings.long_press_threshold_ms:
                state.long_press_fired = True
                events.append(
                    ButtonEvent(
                        type=ButtonEventType.LONG_PRESS,
                        button=button,
                        controller=controller,
                        timestamp_ms=timestamp_ms,
                    )
                )

        return events


_BUTTON_INDEX_MAP: dict[int, XRButton] = {
    0: XRButton.TRIGGER,
    1: XRButton.SQUEEZE,
    2: XRButton.TOUCHPAD,
    3: XRButton.THUMBSTICK,
    4: XRButton.BUTTON_PRIMARY,
    5: XRButton.BUTTON_SECONDARY,
    6: XRButton.MENU,
}


def button_index_to_enum(index: int) -> XRButton | None:
    """Convert xr-standard gamepad button index to XRButton enum.

    Args:
        index: Button index from xr-standard gamepad (0-6).

    Returns:
        Corresponding XRButton enum value, or None if index is invalid.
    """
    return _BUTTON_INDEX_MAP.get(index)


class EventProcessor:
    """Orchestrates event detection and manages subscriptions.

    This class parses raw XR state dictionaries, feeds them to a ButtonDetector,
    and invokes registered callbacks when events are detected.
    """

    _detector: ButtonDetector
    _callbacks: dict[
        ButtonEventType, list[tuple[Callable, XRButton | None, XRController | None]]
    ]

    def __init__(self, settings: EventSettings):
        """Initialize the EventProcessor.

        Args:
            settings: Configuration settings for event detection.
        """
        self._detector = ButtonDetector(settings)
        self._callbacks = {t: [] for t in ButtonEventType}

    def on_button_down(
        self,
        button: XRButton | None = None,
        controller: XRController | None = None,
        callback: Callable[[ButtonEvent], None] | None = None,
    ) -> None:
        """Register a callback for button down events.

        Args:
            button: Optional button filter. If set, only events for this button trigger the callback.
            controller: Optional controller filter. If set, only events for this controller trigger the callback.
            callback: The function to call when the event occurs.
        """
        if callback:
            self._callbacks[ButtonEventType.BUTTON_DOWN].append(
                (callback, button, controller)
            )

    def on_button_up(
        self,
        button: XRButton | None = None,
        controller: XRController | None = None,
        callback: Callable[[ButtonEvent], None] | None = None,
    ) -> None:
        """Register a callback for button up events.

        Args:
            button: Optional button filter.
            controller: Optional controller filter.
            callback: The function to call when the event occurs.
        """
        if callback:
            self._callbacks[ButtonEventType.BUTTON_UP].append(
                (callback, button, controller)
            )

    def on_double_press(
        self,
        button: XRButton | None = None,
        controller: XRController | None = None,
        callback: Callable[[ButtonEvent], None] | None = None,
    ) -> None:
        """Register a callback for double press events.

        Args:
            button: Optional button filter.
            controller: Optional controller filter.
            callback: The function to call when the event occurs.
        """
        if callback:
            self._callbacks[ButtonEventType.DOUBLE_PRESS].append(
                (callback, button, controller)
            )

    def on_long_press(
        self,
        button: XRButton | None = None,
        controller: XRController | None = None,
        callback: Callable[[ButtonEvent], None] | None = None,
    ) -> None:
        """Register a callback for long press events.

        Args:
            button: Optional button filter.
            controller: Optional controller filter.
            callback: The function to call when the event occurs.
        """
        if callback:
            self._callbacks[ButtonEventType.LONG_PRESS].append(
                (callback, button, controller)
            )

    def process(self, pose: np.ndarray, xr_state_dict: dict) -> None:
        """Process XR state and trigger events.

        Args:
            pose: The 4x4 transformation matrix (unused, for Teleop.subscribe compatibility).
            xr_state_dict: The raw XR state dictionary.
        """
        timestamp_ms = xr_state_dict.get("timestamp_unix_ms", 0.0)
        devices = xr_state_dict.get("devices", [])

        for device in devices:
            if device.get("role") != "controller":
                continue

            handedness = device.get("handedness")
            try:
                controller = XRController(handedness)
            except ValueError:
                continue

            gamepad = device.get("gamepad")
            if not gamepad:
                continue

            buttons = gamepad.get("buttons", [])
            for i, button_state in enumerate(buttons):
                button_enum = button_index_to_enum(i)
                if not button_enum:
                    continue

                is_pressed = button_state.get("pressed", False)
                detected_events = self._detector.update(
                    button_enum, controller, is_pressed, timestamp_ms
                )

                for event in detected_events:
                    self._fire_callbacks(event)

    def _fire_callbacks(self, event: ButtonEvent) -> None:
        """Fire all callbacks matching the event filters."""
        for callback, b_filter, c_filter in self._callbacks.get(event.type, []):
            if (b_filter is None or b_filter == event.button) and (
                c_filter is None or c_filter == event.controller
            ):
                callback(event)
