# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""
Script to help evaluate the coverage report.
"""
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def print_summary(file_name, file_data, threshold, verbose_all):
    """
    Print the summary.

    Parameters:
    ----------
    file_name(str)  :  Filename.
    file_data(json)  :  File data.
    threshold(float) :  Percentage to use as threshold.
    verbose_all(bool) :    Specific all verbose option.
    """
    file_summary = file_data["summary"]
    percent_covered = file_summary["percent_covered"]

    if percent_covered < threshold or verbose_all:
        missing_lines = file_data["missing_lines"]
        covered_lns = file_summary["covered_lines"] + file_summary["excluded_lines"]
        print(f"File {file_name} needs additional tests.")
        round_pc = round(percent_covered, 2)
        print(f"- Current coverage percentage: {round_pc:.2f}.")
        total_lines = len(missing_lines) + covered_lns
        print(f"- Covered lines: {covered_lns}/{total_lines}")
        print(f"- See missing lines: {missing_lines}.\n")


def run(report_path, threshold, specific_file, verbose_all):
    """
    Run the helpers script.

    Parameters:
    ----------
    report_path (str) :    Specifies the path to the report file.
    threshold(float) :     Percentage to use as threshold.
    specific_file(str) :   Specific file to check in the report. If None all will be analysed.
    verbose_all(bool) :    Specific all verbose option.
    """
    try:
        with open(report_path, encoding="utf-8") as report:
            report_data = json.load(report)
            totals_percent_covered = report_data["totals"]["percent_covered"]
            round_tpc = round(totals_percent_covered, 2)
            print(f"Total percent covered: {round_tpc:.2f}.\n")

            # If a specific file is None, the full report is analysed.
            if specific_file is None:
                for file_key in report_data["files"]:
                    print_summary(
                        file_key, report_data["files"][file_key], threshold, verbose_all
                    )
            else:
                # Health check
                if specific_file not in report_data["files"]:
                    print(f"File {specific_file} isn't in the coverage report.")
                else:
                    print_summary(
                        specific_file,
                        report_data["files"][specific_file],
                        threshold,
                        verbose_all,
                    )
    except FileNotFoundError:
        print(f"The report file {report_path} is not found.")
        return
    except json.JSONDecodeError:
        print(f"The report file '{report_data}' is not a valid JSON document.")
        return


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--report-path",
        default="coverage.json",
        help="""
            Specify the path to the report file.
            Default is 'coverage.json' in the current directory.
            """,
    )
    parser.add_argument(
        "--threshold-percentage",
        default="90",
        type=float,
        help="""
            Specify the threshold percentage.
            If a file has a lower score will be displayed.
            Default is 90.
            """,
    )
    parser.add_argument(
        "--file",
        default=None,
        type=str,
        help="""
            Specify the specific file to evaluate the coverage.
            If default value (None) all the files in the coverage report will be evaluated.
            """,
    )
    parser.add_argument(
        "--verbose-all", action="store_true", help="Enable all verbose options"
    )

    parsed_args = parser.parse_args()

    run(
        parsed_args.report_path,
        parsed_args.threshold_percentage,
        parsed_args.file,
        parsed_args.verbose_all,
    )
