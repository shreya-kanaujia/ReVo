#!/usr/bin/env python3
"""Build a deterministic, non-overlapping Track B validation CPU partition."""

import argparse
import json


ROLE_WIDTHS = {
    "sender": 2,
    "encoder": 4,
    "receiver": 1,
    "signaling": 1,
    "trace": 1,
}


def parse_cpu_list(value):
    cpus = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = map(int, part.split("-", 1))
            cpus.extend(range(first, last + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def partition(cpus):
    cpus = sorted(set(map(int, cpus)))
    required = sum(ROLE_WIDTHS.values())
    if len(cpus) < required:
        raise RuntimeError(
            f"Track B cgroup isolation requires {required} CPUs; available={cpus}"
        )
    result = {}
    offset = 0
    for role, width in ROLE_WIDTHS.items():
        result[role] = cpus[offset:offset + width]
        offset += width
    result["sender_container"] = result["sender"] + result["encoder"]
    result["available"] = cpus
    return result


def compact(cpus):
    return ",".join(map(str, cpus))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpus", required=True)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    result = partition(parse_cpu_list(args.cpus))
    if args.shell:
        for role in (*ROLE_WIDTHS, "sender_container"):
            print(f"TRACK_B_{role.upper()}_CPUS={compact(result[role])}")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
