# Processes

This document describes the processes of one instance, the deadlines, the progress records, and how the harness
attributes crashes and hangs.

## 1. Process tree

1. **pytest process.** It runs the configuration fixture.
2. **Supervisor.** The configuration fixture starts each launcher through `run_in_pty()` from the prototype
   ([terminal.py](../_prototypes/pytest_integration/terminal.py)). The supervisor gives live output and kills every
   leftover process at the end.
3. **Launcher.** One launcher for each instance: `python -m harness.launcher` with the instance name, the
   configuration and the absolute data directory.
4. **torchrun.** The launcher starts `python -m torch.distributed.run` with the same options that the xDiT CLI uses
   ([cli.py:131-142](../../../xfuser/cli.py#L131-L142)), but with the worker module of the harness instead of
   `xfuser.runner`. The number of workers (`--nproc_per_node`) is the `num_gpus` field of the merged entry
   ([test_definition.md](test_definition.md), section 3.1). Without it, the launcher computes the number as the
   xDiT CLI does ([cli.py:122-127](../../../xfuser/cli.py#L122-L127)).
5. **Workers.** One worker for each rank: `python -m harness.worker`.

## 2. Launcher contract

1. The launcher survives a crash or a hang of its workers.
2. The launcher writes the instance status and the case states to files.
3. The launcher exits with 0 when it wrote its records, also after a crash or a hang of the workers.
4. A nonzero exit means that the launcher itself failed and no data exists. The configuration fixture then raises
   an exception, and every case of the configuration gets the pytest outcome "error".

## 3. Environment

1. [conftest.py:4-11](../../conftest.py#L4-L11) sets `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1` in `pytest_configure`.
   The launchers and workers inherit it. Thus a GPU fault does not write a core dump into the working tree.
2. stdin of the workers is `/dev/null`, as in the prototype.

## 4. Deadlines

1. The launcher enforces the deadlines, because it is the parent of torchrun.
2. The defaults are in [test_definition.md](test_definition.md), section 6:
   1. **init:** from the start of the launcher to the record "init done" of the slowest rank;
   2. **pipeline run:** from the record "pipeline start" to the record "pipeline end" of the slowest rank;
   3. **each replayed call:** the time since the last progress record of the slowest rank, during the replays.
3. The per-call deadline applies to each replayed call, not to a whole case. Reason: a case without a call number
   covers all calls of its component, and the number of calls is known only after the pipeline run of A.
4. The per-call defaults (2 minutes for A, 8 minutes for B) are shorter than the 10-minute default timeout of a
   process group in the local torch version (medium confidence). Thus a rank that is stuck in a collective
   operation gets HUNG, before the watchdog of the process group turns the hang into a crash.
5. On a missed deadline, the launcher stops the process tree of torchrun: SIGTERM, then SIGKILL after the grace
   period. Then it records the hang.
6. The grace period is 30 seconds (Not confirmed). GPU processes need longer than the 5-second default of the
   prototype to release their resources.

## 5. Progress records

1. Each worker writes its own progress file: one JSON line for each event, flushed at once.
2. A record holds the time, the rank, the phase, and for a replay the case and the call address.
3. Events:
   1. init start, init done;
   2. pipeline start, pipeline end;
   3. replay start, for each replayed call;
   4. case done, after the synchronization and the barrier at the end of each case.
4. The synchronization and the barrier at the end of each case make sure that an asynchronous GPU fault shows up
   inside the case that caused it. A silent memory corruption that shows up only in a later case is an accepted
   limit.

## 6. Exceptions in a worker

1. The top level of the worker catches every exception.
2. It writes an error record with the traceback, the phase, and for a replay the case and the call address.
3. Then the worker exits with a nonzero status.
4. torchrun runs without restarts, so it stops all other workers when one worker fails (medium confidence; not
   verified with a run).
5. Thus any exception ends the instance like a crash. The status is ERROR_A or ERROR_B.

## 7. Attribution

The launcher reads the progress records and the error records after the end of torchrun.

1. **Failure during a replay.** The case whose replay was running gets CRASHED, HUNG or ERROR of that instance.
   The record of the first rank that failed decides the case.
2. **Failure outside the replays** (init, pipeline run, conversion in B): no case is the cause. Every case without
   complete data gets FAILED_TO_START. The reason names the instance and the phase, and holds the traceback if
   there is one.
3. **Complete data** means the data of A and of B for the case.
4. **Cases with complete data.** A case whose replays in B finished before a failure of B keeps its data. The test
   function checks it as usual.
5. **Cases without complete data.** Every other case gets FAILED_TO_START.
6. **After a failure of A,** the configuration fixture does not start B. Thus no case has complete data: the case whose
   replay was running gets CRASHED_A, HUNG_A or ERROR_A, and every other case gets FAILED_TO_START.

Version 1 does not recover. It does not run B for the cases that A finished, and it does not continue after B
fails. This is future work ([future_work/index.md](future_work/index.md), section 2.5).
