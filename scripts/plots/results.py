# Read Python CSV file and plot results using matplotlib

import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse


class PlotAlgorithmResults:
    args = None
    x_data = None
    y_data = None

    def __init__(self):
        self.parse_cli_arguments()
        self.parse_csv_file()
        self.plot_results()

    def parse_cli_arguments(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--csv",
            type=str,
            help="Path of CSV file with results",
            metavar="",
        )

        parser.add_argument(
            "--label",
            type=str,
            help="Label for plot",
            metavar="",
        )

        self.args = parser.parse_args()

    def parse_csv_file(self):
        # Read CSV file
        with open(self.args.csv, "r") as csvfile:
            plots = csv.DictReader(csvfile, delimiter=",")
            # Skip header
            next(plots)
            # Read data
            x = []
            y = []
            for row in plots:
                x.append(float(row["agent_timesteps_total"]))
                y.append(float(row["episode_reward_mean"]))

            self.x_data = x
            self.y_data = y

    def plot_results(self):
        # Plot data
        plt.plot(
            self.x_data,
            self.y_data,
            label=self.args.label,
        )

        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.title(self.args.label)

        plt.show()


if __name__ == "__main__":
    PlotAlgorithmResults()
