import sys


class Tee:
    """
    Redirects all print() output to both the console and a log file at the same time.

    Drop this into any script where you want to save the full console output to disk.

    Example
    -------
    Place these three lines at the start of your run function:

        tee = Tee("/path/to/your/run_log.txt")
        sys.stdout = tee

    And this one line at the very end:

        tee.close()

    Everything printed between those two points will appear in the console
    as normal, and will also be written to the log file.
    """

    def __init__(self, filepath):
        # Keep a reference to the real terminal so we can still write to it
        self.terminal = sys.__stdout__
        # Open the log file for writing (creates it if it does not exist)
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        # Write to both the terminal and the log file
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Flush both outputs so nothing gets stuck in a buffer
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        # Some libraries (e.g. transformers) call sys.stdout.isatty() to check
        # whether they are running in a terminal. Since we are writing to a file,
        # we always return False to avoid an AttributeError crash.
        return False

    def close(self):
        # Close the log file and restore sys.stdout to the real terminal
        self.log.close()
        sys.stdout = sys.__stdout__