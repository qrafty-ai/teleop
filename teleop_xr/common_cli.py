from dataclasses import dataclass
from .config import InputMode


@dataclass
class CommonCLI:
    host: str = "0.0.0.0"
    port: int = 4443
    input_mode: InputMode = InputMode.CONTROLLER
    double_press_ms: float = 300
    long_press_ms: float = 1000

    def event_settings(self):
        """Create EventSettings from CLI configuration.

        Creates an EventSettings instance configured with the timing values
        from this CLI configuration (double_press_ms and long_press_ms).

        Note: This method uses a lazy import of EventSettings to avoid
        circular dependencies.

        Returns:
            EventSettings: Configured event settings for button press detection.
        """
        from .events import EventSettings

        return EventSettings(
            double_press_threshold_ms=self.double_press_ms,
            long_press_threshold_ms=self.long_press_ms,
        )
