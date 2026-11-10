# Statuses

This document lists every status of a case, which part of the harness sets it, and its pytest outcome. The list is
not fixed. New statuses can be added. Users do not see status identifiers. The harness raises verbose,
human-readable messages instead.

## 1. Statuses of a case

| Status | Set by | Meaning | Data for checks | pytest outcome |
|---|---|---|---|---|
| ABOUT_TO_START | group fixture | A fresh run created the state. No instance has finished the case yet. | no | error, if a later run finds it (section 2) |
| UNSUPPORTED | worker of A or B | The configuration or the case uses a feature that version 1 does not support ([unsupported.md](unsupported.md)). | no | skipped |
| FAILED_TO_START | worker or launcher | The case has no data to finish the test: a path error, or a failure outside the replays. | no | error |
| CRASHED_A, CRASHED_B | launcher | The instance crashed while it replayed this case. | no | failed |
| HUNG_A, HUNG_B | launcher | The instance missed a deadline while it replayed this case. | no | failed |
| ERROR_A, ERROR_B | worker | An exception ended the instance while it replayed this case. The state holds the traceback. | no | failed |
| NON_DETERMINISTIC_A, NON_DETERMINISTIC_B | worker | Two replays of one call gave different outputs. | yes | failed at once without the override; warning with the override |
| REPLAY_MISMATCH_A, REPLAY_MISMATCH_B | worker | Two replays agree, but differ from the capture. | yes, not trusted | failed at once |
| NOT_REACHED_B (Not confirmed) | worker of B | The call addresses of B differ from those of A. | partly | failed at once |
| DATA_COMPLETE | worker of B | A and B finished the case. The test function can compare. | yes | from the checks |
| PASSED | test function | All checks passed. The data files are deleted. | deleted | passed |
| FAILED | test function | At least one deferred check failed. The data files stay. | yes | failed |

The status names are a proposal (Not confirmed), except the names that the user gave: UNSUPPORTED,
FAILED_TO_START, CRASHED, HUNG, ERROR, NON_DETERMINISTIC and REPLAY_MISMATCH.

## 2. Outcomes in a reuse run

A reuse run does not start A or B. It uses the stored states ([files.md](files.md), section 5).

| Stored state | Outcome in the reuse run |
|---|---|
| PASSED | skipped, with "passed in a previous run" |
| UNSUPPORTED | skipped again, with the stored reason |
| FAILED, data present | compared again with the current tolerances |
| a status without data | the stored status and reason again |
| NON_DETERMINISTIC, REPLAY_MISMATCH, NOT_REACHED_B | the stored status again |
| ABOUT_TO_START | error: "the previous run did not finish; clean the directory" |
| no state (the case was not in the original run) | error: "this case was not part of the run that produced the data" |

## 3. Statuses of an instance

The launcher also records one status for the instance as a whole. The group fixture uses it to decide whether to
start B.

1. CLEAN: the instance finished all its steps.
2. UNSUPPORTED: the configuration is not supported. B does not start.
3. CRASHED, HUNG, ERROR: with the phase (init, pipeline run, replay) and, for a replay, the case. B does not start
   after such a status of A.

## 4. Mapping to pytest outcomes

1. **failed:** tolerance failures and the other deferred checks; NON_DETERMINISTIC without the override;
   REPLAY_MISMATCH; NOT_REACHED_B; CRASHED, HUNG and ERROR of A or B.
2. **error:** FAILED_TO_START; ABOUT_TO_START in a reuse run; a case that was not in the original run.
3. **skipped:** UNSUPPORTED; PASSED in a reuse run.

pytest gives the outcome "error" only for an exception in a fixture. The case fixture therefore raises the
exception for these statuses ([pytest_integration.md](pytest_integration.md), section 4).
