# Files

This document describes where the harness writes data, the state of each case, the behavior of a re-run, the
safety guards, serialization, and the metrics file.

## 1. Path resolution

1. The harness writes into two directories next to the test module:
   1. `_data/` for dumps and states;
   2. `_metrics/` for the metrics files.
2. The base directory is the directory of the running test module (`request.path.parent` in the group fixture). It
   is not the working directory.
3. The pytest process resolves both directories once. Launchers and workers get absolute paths, so their working
   directory has no effect.
4. The metrics files live in a directory separate from the dumps. Thus a cleanup of the dumps cannot delete them.

## 2. Configuration directory

1. Each configuration writes into its own directory: `_data/<name>/`. `<name>` is the configuration name
   ([test_definition.md](test_definition.md), section 7).
2. The directory holds:
   1. a marker file that the harness creates with the directory;
   2. a lock file;
   3. one state file for each case;
   4. the instance records of A and B;
   5. the progress files of the workers;
   6. the dumps of A and B.

## 3. One owner of the layout

1. One module of the harness defines all paths in the configuration directory. The launchers use this module.
2. The state of each case holds the paths of its data files. The case fixture gives these paths to the test
   function. The test function reads and deletes the files at these paths. It does not compute paths.
3. The layout module maps a case ID and a call address to file names that are safe for the file system.

## 4. State files

1. Each case has one state file.
2. Three writers change the state of a case: the group fixture, the launchers and the test function. Each writer
   replaces the whole file. It writes a temporary file and renames it to the state file. There is no
   read-modify-write of a shared file.
3. The state holds:
   1. the status of the case, and for each call and rank the statuses of A and B;
   2. the reason and the traceback of a failure, if there is one;
   3. the warnings of the dtype audit of B;
   4. the paths of the data files;
   5. the run ID of the run that produced the data.
4. The state file stays after a pass, also when the data files are deleted.

## 5. Fresh run and reuse run

1. At the start, the group fixture reads the state files of the configuration directory.
2. **Fresh run.** If the directory has no state files, or if every recorded case passed, the group fixture cleans
   the directory and runs A and B from the start. UNSUPPORTED counts as passed for this decision.
3. **Reuse run.** In all other cases, the group fixture prints a warning and does not start A or B. Each case gets
   the outcome of its stored state ([statuses.md](statuses.md), section 2).
4. The warning says that the run uses old data. Thus changes of the xDiT code or of the settings since then are
   not tested.
5. There is no fingerprint check of the settings. After a change of the settings, the user must clean the
   directory.
6. When a reuse run passes all remaining cases, all states are PASSED. The next run is then a fresh run.

## 6. Safety guards

1. **Marker file.** Automatic cleaning deletes only a directory that holds the marker file. A directory that is not
   empty and has no marker file is an error. Thus a wrong path cannot delete unrelated files.
2. **Lock.** The pytest session holds an exclusive lock (`fcntl.flock`) on the lock file of the configuration
   directory. A second session on the same directory fails at once. Without the lock, a second session could clean
   the directory while the first session writes into it.

## 7. Serialization

1. Instances exchange data only through files, because B reads the inputs of A in another process.
2. Tensors go into safetensors files.
3. The structure and the primitive values go into JSON: numbers, booleans, `None`, strings, and nested lists,
   tuples and mappings.
4. Each tensor in the JSON structure refers to its entry in the safetensors file.
5. Any other object type cannot be serialized. Its case gets UNSUPPORTED with a message that names the argument
   ([instance_a.md](instance_a.md), section 3).

## 8. Files that stay

1. A passing test deletes the data files of its case.
2. A failing test keeps the inputs and the outputs of A and B.
3. A case that crashed, hung or did not start keeps whatever files exist.

## 9. Metrics file

### 9.1 Location and lifetime

1. Each configuration has one metrics file: `_metrics/<name>.jsonl`.
2. The file only grows. The cleaning of the configuration directory never touches it.
3. Its purpose: it shows how the deviations and the tolerances change over time while the implementation changes.

### 9.2 Format

1. JSON Lines: one JSON object for each line.
2. Timestamps are in UTC, in ISO 8601 format.
3. Each append takes an exclusive `fcntl.flock` lock on the metrics file. The lock of the configuration directory
   (section 6) already stops two sessions on one configuration. The lock of the metrics file is a second guard
   that costs almost nothing, so two writers can never mix their lines.

### 9.3 Lines

1. There is one line for each call, tensor output and metric of a case.
2. Fields of a line:
   1. `timestamp` and `run_id` of the run that wrote the line;
   2. `data_run_id`: the run ID of the run that produced the compared data. In a reuse run, it differs from
      `run_id`. An analysis can use it to drop repeated values;
   3. `xdit_commit` and `xdit_dirty`: the xDiT git commit, and whether the working tree had uncommitted changes;
   4. `gpu_arch`, for example `gfx950`;
   5. `configuration`, `yaml_entry`, `case`, `call` (the call address) and `output`;
   6. `metric` and `values`: a list with one value for each rank, in rank order;
   7. `tolerance`: the tolerance in force;
   8. `outcome`: `passed` or `failed` for a checked metric, `reported` for `cosine`.
3. A case without metrics writes one line with the fields 1 to 5 (without `call` and `output`) and its outcome, and
   without metric values. Examples: a case that crashed, hung, did not start, was unsupported, or stopped at once
   ([checks.md](checks.md), section 2).
4. A case that is skipped because it passed in a previous run writes nothing.
