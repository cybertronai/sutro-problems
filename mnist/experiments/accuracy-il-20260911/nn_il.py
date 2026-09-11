#!/usr/bin/env python3
"""The existing 1NN submission expressed as nested loops of v4 primitives."""
import argparse
import json
from pathlib import Path
from il import ref, ins, loop, make_program, score


def nearest_neighbor(n_train=600, n_test=600, features=9):
    if not 1 <= features <= 9 or n_train < 1 or n_test < 1:
        raise ValueError("1NN dimensions must be positive; at most 9 features")
    temp, dist, best_dist, best_label, cond = [ref("hot", i) for i in range(5)]
    def distance(row_offset=0, **row_coefficients):
        coefficients = {var: value * features for var, value in row_coefficients.items()}
        return [ins("set", dist, 0), loop("pixel", features, [
            ins("sub", temp, ref("query", pixel=1), ref("train", row_offset * features, pixel=1, **coefficients)),
            ins("mul", temp, temp, temp),
            ins("add", dist, dist, temp),
        ])]
    body = [loop("load_train", n_train * features, [ins("recv", ref("train", load_train=1))]),
            loop("load_label", n_train, [ins("recv", ref("labels", load_label=1))]),
            loop("query_number", n_test, [
                loop("load_query", features, [ins("recv", ref("query", load_query=1))]),
                *distance(), ins("copy", best_dist, dist), ins("copy", best_label, ref("labels")),
                loop("row", n_train - 1, [
                    *distance(row=1), ins("cmp", cond, dist, best_dist),
                    ins("select", best_dist, cond, dist, best_dist),
                    ins("select", best_label, cond, ref("labels", row=1), best_label),
                ], start=1), ins("send", best_label),
            ])]
    return make_program([("query", 9), ("hot", 5), ("train", n_train * features), ("labels", n_train)], body,
                        {"algorithm": "1NN squared Euclidean distance; first index wins ties",
                         "n_train": n_train, "n_test": n_test, "features": features})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("1nn.il.json"))
    args = parser.parse_args()
    document = nearest_neighbor()
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    result = score(document)
    args.output.with_suffix(".score.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
