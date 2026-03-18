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
        from .events import EventSettings

        return EventSettings(
            double_press_threshold_ms=self.double_press_ms,
            long_press_threshold_ms=self.long_press_ms,
        )
