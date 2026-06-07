"""
Kevin Kuipers (s5051150)
Federico Berdugo Morales (s5363268)
Sían Bos García (s5962277)
Mahmoud Saad (S6175767)
"""

from utilities.timer import TimeManager

with TimeManager("Imports"):
    import random

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


def main():
    print("running")


if __name__ == "__main__":
    with TimeManager("Program", True):
        main()

    # to keep plots open
    plt.show()
