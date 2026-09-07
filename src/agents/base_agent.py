import logging


class Agent:
    """
    Abstract baseclass for all system Agents.
    Enforces structure and handles color-coded terminal logging.
    """
    # Foreground colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    # Background color
    BG_BLACK = '\033[40m'
    RESET = '\033[0m'

    name: str = "Base Agent"
    color: str = WHITE

    def __init__(self):
        # Creates a dedicated logger for this specific agent name
        self.logger = logging.getLogger(self.name)

    def log(self, message: str):
        """Outputs a colored log identifying the agent."""
        color_code = self.BG_BLACK + self.color
        formatted_message = f"[{self.name}] {message}"

        # Uses the configured logger instead of the generic root logger
        self.logger.info(color_code + formatted_message + self.RESET)