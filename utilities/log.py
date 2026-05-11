"""
Usage: safe textual output to file
"""

import os


class Logger:
    "Utility class used for logging"

    _logged_text = ""

    @staticmethod
    def log(*args, end: str = "\n") -> None:
        "Log arguments"

        if not isinstance(end, str):
            raise TypeError(f"End argument must be a string ({end} was given)")

        for arg in args:
            text = str(arg)
            Logger._logged_text += text

            print(text, end="")

        Logger._logged_text += end
        print(end, end="")

    @staticmethod
    def save_and_reset_logs(file_name: str) -> None:
        "Saves and resets the current logged text"

        if not isinstance(file_name, str):
            raise TypeError(
                f"File_name argument must be a string ({file_name} was given)"
            )

        file_name = file_name.replace(os.sep, "_")

        with open(f"saved_data/{file_name}", "w") as file:
            file.write(Logger._logged_text)

        Logger._logged_text = ""
