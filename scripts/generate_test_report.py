from datetime import datetime
from pathlib import Path
import sys
import subprocess


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "testing"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def main():

    print(
        "Running complete test suite..."
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-v",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    report_path = (
        OUTPUT_DIR
        / "test_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "FAKEGUARD AI TEST REPORT\n"
        )

        file.write(
            "=" * 60
        )

        file.write("\n\n")

        file.write(
            f"Generated: "
            f"{datetime.now()}\n\n"
        )

        file.write(
            "Exit code: "
            f"{result.returncode}\n\n"
        )

        file.write(
            result.stdout
        )

        if result.stderr:

            file.write(
                "\n\nERROR/WARNING OUTPUT\n"
            )

            file.write(
                result.stderr
            )

    print(
        "\nReport saved to:"
    )

    print(
        report_path
    )

    if result.returncode == 0:

        print(
            "\nALL TESTS PASSED."
        )

    else:

        print(
            "\nSOME TESTS FAILED."
        )


if __name__ == "__main__":
    main()
