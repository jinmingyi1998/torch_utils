import concurrent.futures
import gzip
import re
import time
from pathlib import Path
from typing import Dict, List

import orjson
import tqdm
import typer
from typing_extensions import Annotated

app = typer.Typer(no_args_is_help=True, add_completion=False)


def process_trace_json(
    path: Path,
    start_time_ms: int,
    end_time_ms: int,
    filter_names: List[str],
) -> Dict:
    tp_rank = _get_tp_rank_of_path(path)
    with gzip.open(str(path), "rt") as f:
        trace = orjson.loads(f.read())
        print(f"One file: {path=} {list(trace)=}")

        if "traceEvents" in trace:
            events = trace["traceEvents"]
            print(f"{len(events)=}")

            # format: us
            min_ts = min(e["ts"] for e in events)

            ts_interest_start = min_ts + 1000 * start_time_ms
            ts_interest_end = min_ts + 1000 * end_time_ms
            print(
                f"{ts_interest_start=} us {ts_interest_end=} us duration={ts_interest_end - ts_interest_start} us"
            )

            events = [
                e for e in events if ts_interest_start <= e["ts"] <= ts_interest_end
            ]
            if filter_names:

                def filter_name_fn(e):
                    for name in filter_names:
                        if name in e["name"]:
                            return True
                    return False

                events = [e for e in events if filter_name_fn(e)]

            print(f"after filtering {len(events)=}")

            for e in events:
                if e["name"] == "process_sort_index":
                    pid = _maybe_cast_int(e["pid"])
                    if pid is not None and pid < 1000:
                        e["args"]["sort_index"] = 100 * tp_rank + int(e["pid"])
                # modify in place to speed up
                e["pid"] = f"[TP{tp_rank:02d}] {e['pid']}"

            trace["traceEvents"] = events

        else:
            print(f"Warning: No 'traceEvents' in {path}")

    return trace


def merge_chrome_traces(
    interesting_paths: List[Path],
    output_path: Path,
    start_time_ms: int,
    end_time_ms: int,
    filter_names: List[str],
):
    merged_trace = {"traceEvents": []}

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                process_trace_json, path, start_time_ms, end_time_ms, filter_names
            )
            for path in interesting_paths
        ]

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(interesting_paths)
        ):
            trace = future.result()
            merged_trace["traceEvents"].extend(trace["traceEvents"])
            for key, value in trace.items():
                if key != "traceEvents" and key not in merged_trace:
                    merged_trace[key] = value

    print(f"Write output to {output_path}")
    t0 = time.time()
    with gzip.open(str(output_path), "wb") as f:
        f.write(orjson.dumps(merged_trace))
    print(f"write file elapsed time: {time.time()-t0} seconds")


def _maybe_cast_int(x):
    try:
        return int(x)
    except ValueError:
        return None


def _get_tp_rank_of_path(p: Path):
    return int(re.search("TP-(\d+)", p.name).group(1))


@app.command()
def main(
    profile_id: Annotated[str, typer.Option(help="prefix name of trace file")],
    data_dir: Annotated[
        Path,
        typer.Option(dir_okay=True, exists=True, help="data dir for input trace file"),
    ] = Path(),
    start_time_ms: Annotated[
        int, typer.Option(help="begin ts offset from start in ms")
    ] = 0,
    end_time_ms: Annotated[
        int, typer.Option(help="end ts offset from start in ms")
    ] = 4294967296,
    filter_name: Annotated[
        List[str],
        typer.Option(
            help="list of name for filter events name: usage example: --filter-name aaa  --filter-name bbb"
        ),
    ] = None,
):
    dir_data = data_dir

    interesting_paths = sorted(
        [p for p in dir_data.glob("*.json.gz") if p.name.startswith(profile_id)],
        key=_get_tp_rank_of_path,
    )
    print(f"{interesting_paths=}")

    output_path = dir_data / f"merged-{profile_id}.trace.json.gz"
    merge_chrome_traces(
        interesting_paths,
        output_path,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        filter_names=filter_name,
    )


if __name__ == "__main__":
    app()
