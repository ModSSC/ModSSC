from __future__ import annotations

from modssc.hpo import Space


def main() -> None:
    space = Space.from_dict(
        {
            "method": {
                "params": {
                    "max_iter": [10, 20, 40],
                    "confidence_threshold": [0.7, 0.8, 0.9],
                }
            }
        }
    )

    for trial in space.iter_grid():
        print(f"trial={trial.index} params={trial.params}")


if __name__ == "__main__":
    main()
