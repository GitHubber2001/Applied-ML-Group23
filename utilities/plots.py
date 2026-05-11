"""
Usage: safe the open matplot plots
"""

import os

import matplotlib.pyplot as plt


def save_open_plots(
    prefix: str = "figure_", postfix: str = "", file_extension: str = ".png"
) -> None:
    """Saves all open plots"""

    if not isinstance(prefix, str):
        raise TypeError(f"Prefix argument must be a string ({prefix} was given)")

    if not isinstance(postfix, str):
        raise TypeError(f"Postfix argument must be a string ({postfix} was given)")

    if not isinstance(file_extension, str):
        raise TypeError(
            f"File extension argument must be a string ({file_extension} was given)"
        )

    for i in plt.get_fignums():
        file_name = (prefix + str(i) + postfix).replace(os.sep, "_")

        plt.figure(i)
        plt.savefig(f"saved_data/{file_name}{file_extension}")
