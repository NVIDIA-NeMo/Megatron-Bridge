#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Produce a `nsys stats -r nvtx_sum --format csv`-compatible CSV from a nsys SQLite db.

This sidesteps needing the `nsys` CLI on the host. It reads the `NVTX_EVENTS` table,
joins range-push with range-pop events by ``rangeId``, sums durations per range
name, and writes the columns that :mod:`print_nsys_leaderboard` cares about
(``Range`` + ``Total Time (ns)``).

Usage::

    python extract_nsys_csv.py PROFILE.sqlite OUT.csv
"""

import csv
import sqlite3
import sys
from collections import defaultdict


def extract(db_path: str, out_csv: str) -> None:
    """Aggregate NVTX range durations from ``db_path`` and write ``out_csv``."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # NsysEventType: 59 = NvtxPushPopRange (start + end on same row),
    # 60 = NvtxStartEndRange. Both have non-null `end` for completed ranges.
    cur.execute(
        """
        SELECT COALESCE(NVTX_EVENTS.text, StringIds.value) AS range_name,
               (NVTX_EVENTS.end - NVTX_EVENTS.start) AS dur_ns
        FROM NVTX_EVENTS
        LEFT JOIN StringIds ON NVTX_EVENTS.textId = StringIds.id
        WHERE NVTX_EVENTS.end IS NOT NULL
          AND NVTX_EVENTS.eventType IN (59, 60, 70, 75)
        """
    )
    totals = defaultdict(lambda: [0, 0])  # name -> [total_ns, count]
    for name, dur in cur:
        if name is None:
            continue
        totals[name][0] += dur or 0
        totals[name][1] += 1
    conn.close()

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Time (%)",
                "Total Time (ns)",
                "Instances",
                "Avg (ns)",
                "Med (ns)",
                "Min (ns)",
                "Max (ns)",
                "StdDev (ns)",
                "Style",
                "Range",
            ]
        )
        grand = sum(t[0] for t in totals.values()) or 1
        for name, (total, n) in sorted(totals.items(), key=lambda kv: -kv[1][0]):
            avg = total / n if n else 0
            pct = 100.0 * total / grand
            w.writerow([f"{pct:.2f}", total, n, f"{avg:.1f}", "", "", "", "", "PushPop", name])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: extract_nsys_csv.py PROFILE.sqlite OUT.csv")
    extract(sys.argv[1], sys.argv[2])
