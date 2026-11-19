"""Generate the demo input manifest from the benchmark test set."""

import argparse
import json
from pathlib import Path


FIXTURES = Path(__file__).resolve().parent
BENCHMARK = FIXTURES.parents[1] / "bench" / "report119"
EXAMPLES = {
    "test-0062": "top",
    "test-0019": "mismatch",
    "test-0074": "outage",
    "test-0393": "rule",
    "test-0395": "past_hazard",
}
# Visually reviewed field photographs, in contact-sheet order.
CURATED_IDS = """
0062 0019 0005 0106 0034 0107 0142 0195 0220 0218
0161 0199 0014 0137 0324 0273 0009 0271 0185 0123
0086 0204 0077 0231 0267 0187 0320 0249 0162 0012
0337 0221 0033 0323 0223 0146 0055 0256 0325 0166
0250 0386 0395 0113 0053 0382 0136 0230 0084 0258
0213 0156 0098 0057 0181 0026 0120 0253 0042 0138
0281 0313 0097 0227 0205 0110 0074 0135 0215 0122
0065 0319 0039 0066 0002 0024 0393 0340 0396 0285
0140 0175 0083 0088 0013 0333 0158 0182 0354 0189
0003 0171 0045 0316 0126 0240 0245 0015 0260 0264
""".split()
REPORT_COUNT = 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the committed manifest.")
    args = parser.parse_args()

    source = json.loads((BENCHMARK / "test119.json").read_text(encoding="utf-8"))
    indexed = {row["id"]: row for row in source}
    if len(indexed) != len(source) or len(source) < REPORT_COUNT:
        raise SystemExit("The test set must contain at least 100 unique IDs.")

    selected = [indexed["test-" + suffix] for suffix in CURATED_IDS]
    assert len(selected) == len({r["image_id"] for r in selected}) == REPORT_COUNT
    assert set(EXAMPLES) <= {r["id"] for r in selected}
    reports = []
    for row in selected:
        report = {key: row[key] for key in ("id", "text", "image_path")}
        if report["id"] in EXAMPLES:
            report["example"] = EXAMPLES[report["id"]]
        reports.append(report)

    output = FIXTURES / "reports.json"
    payload = json.dumps(reports, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != payload:
            raise SystemExit("The committed manifest differs from the benchmark inputs.")
        print(f"Verified {len(reports)} unique benchmark inputs and {len(EXAMPLES)} demo examples.")
    else:
        output.write_text(payload, encoding="utf-8")
        print(f"Wrote {len(reports)} reports to {output.name}.")


if __name__ == "__main__":
    main()
