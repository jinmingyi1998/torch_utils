import gzip
import re
from pathlib import Path
from typing import List, Optional

import orjson
import typer


def merge_chrome_traces(interesting_paths: List[Path], output_path: Path, start_time_ms: int, end_time_ms: int):
    merged_trace = {"traceEvents": []}

    for path in interesting_paths:
        tp_rank = _get_tp_rank_of_path(path)
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            trace = orjson.loads(f.read())
            print(f"One file: {path=} {list(trace)=}")

            if 'traceEvents' in trace:
                events = trace['traceEvents']
                print(f"{len(events)=}")

                # format: us
                min_ts = min(e["ts"] for e in events)

                ts_interest_start = min_ts + 1000 * start_time_ms
                ts_interest_end = min_ts + 1000 * end_time_ms
                print(f"{ts_interest_start=} {ts_interest_end}")

                events = [e for e in events if ts_interest_start <= e["ts"] <= ts_interest_end]
                print(f"after filtering {len(events)=}")

                for e in events:
                    if e["name"] == "process_sort_index":
                        pid = _maybe_cast_int(e["pid"])
                        if pid is not None and pid < 1000:
                            e["args"]["sort_index"] = 100 * tp_rank + int(e["pid"])
                    # modify in place to speed up
                    e["pid"] = f"[TP{tp_rank:02d}] {e['pid']}"

                merged_trace['traceEvents'].extend(events)
            else:
                print(f"Warning: No 'traceEvents' in {path}")

            for key, value in trace.items():
                if key != 'traceEvents' and key not in merged_trace:
                    print(f"Pick {key=} in {path=}")
                    merged_trace[key] = value

    print(f"Write output to {output_path}")
    with gzip.open(output_path, 'wb') as f:
        f.write(orjson.dumps(merged_trace))


def _maybe_cast_int(x):
    try:
        return int(x)
    except ValueError:
        return None


def _get_tp_rank_of_path(p: Path):
    return int(re.search("TP-(\d+)", p.name).group(1))


def main(
        profile_id: Optional[str] = None,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
):
    dir_data = Path("/Users/tom/temp/temp_sglang_server2local")

    if profile_id is None:
        pattern = re.compile(r"^([0-9.]+)-.*$")
        profile_id = max(
            m.group(1)
            for p in dir_data.glob("*.json.gz")
            if (m := pattern.match(p.name)) is not None
        )
    print(f"{profile_id=}")

    interesting_paths = sorted(
        [p for p in dir_data.glob("*.json.gz") if p.name.startswith(profile_id)],
        key=_get_tp_rank_of_path,
    )
    print(f"{interesting_paths=}")

    output_path = dir_data / f"merged-{profile_id}.trace.json.gz"
    merge_chrome_traces(interesting_paths, output_path, start_time_ms=start_time_ms, end_time_ms=end_time_ms)


if __name__ == "__main__":
    typer.run(main)
