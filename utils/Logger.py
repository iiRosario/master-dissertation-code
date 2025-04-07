import os
from datetime import datetime
from env import *
class Logger:
    def __init__(self):
        """
        Initializes the Logger, creating a log file in the format 'log-DD-MM-YYYY.txt'.
        If the file already exists, it continues writing to it.
        """
        # Resolve the absolute path to ensure correctness
        
        self.log_dir = PATH_LOG
        # Ensure the log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Create the filename based on the current date
        today = datetime.today().strftime("%d-%m-%Y")
        self.log_file = os.path.join(self.log_dir, f"log-{today}.txt")

        # Create the file if it does not exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("=== Log Start ===\n")

    def log(self, message: str):
        """
        Writes a message to the log file, separated by a new line.

        :param message: The message to be written in the log.
        """
        timestamp = datetime.now().strftime("[%H:%M:%S]")  # Adds the message timestamp
        with open(self.log_file, "a") as f:
            f.write(f"{timestamp} {message}\n")

# Usage example
if __name__ == "__main__":
    logger = Logger()  # Logs will be stored in "../logs"
    logger.log("This is a test log entry.")
    logger.log("Another log entry.")
