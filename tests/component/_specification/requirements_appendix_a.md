# Requirements and use cases of the component test harness

This document describes the use cases and the requirements for the xDiT component testing harness.

The document consists of 2 parts: the [requirements.md](requirements.md) file is the main part. This file contains Appendix A.

## Appendix A. Names of files, directories and test cases

This is the reference implementation of the names that TD75, TD80, RL10, PI20.2.2, PI50, FD10, FD20 and FD40 define.

```python
"""Names of the files, directories and pytest test cases of the component test harness.

Layout (FD20, TD75, RL10, PI20.2.2); F is the file encoding (FD40), the configuration name is
used without change (TD80):
    <xdit_data_dir>/<config_name>/<instance>_rank_<rank>.jsonl                  rank log
    <xdit_data_dir>/<config_name>/<instance>_<rank>_<F(call_id)>_<content_id>   data file
    <xdit_metrics_dir>/<config_name>/<F(case_id)>.jsonl                         metrics file

Test case ID (PI50); T is the test ID encoding:
    <config_name>/<T(case_id)>
"""

import os
import re
import string

NAME_MAX = 255
"""Maximum basename length in bytes on ext4, XFS, Btrfs and tmpfs."""

CONFIG_NAME_MAX = 200
"""TD80."""

_ALNUM = string.ascii_letters + string.digits


class _Codec:
    """Reversible encoding of any non-empty ``str`` into a restricted alphabet.

    Each input character becomes exactly one output form: itself if it is in ``literal`` (and, for
    the first character, not in ``not_first``); its single-character ``replacement``; or
    ``escape`` + two upper-case hex digits for each byte of its UTF-8 encoding ("surrogatepass"
    makes this total over all Python strings). Literal characters, replacement characters and the
    escape character are disjoint, and every escape character starts a 3-character token, so a
    left-to-right scan restores the input.

    Guarantees: ``decode(encode(x)) == x`` for every non-empty ``x``; ``decode`` raises
    ``ValueError`` for every string that ``encode`` cannot produce, so each input has exactly one
    encoded form.
    """

    def __init__(self, literal: str, replacement: dict[str, str], escape: str, not_first: str = ""):
        self._literal = frozenset(literal)
        self._replacement = dict(replacement)
        self._original = {v: k for k, v in replacement.items()}
        self._escape = escape
        self._not_first = frozenset(not_first)
        self._token = re.compile(rf"{re.escape(escape)}([0-9A-F]{{2}})|(.)", re.DOTALL)
        produced = [escape, *self._replacement.values()]
        if (
            len(escape) != 1
            or len(set(produced)) != len(produced)
            or not self._literal.isdisjoint(produced)
            or not self._literal.isdisjoint(self._replacement)
        ):
            raise ValueError("literal, replaced, replacement and escape characters must be disjoint")

    def encode(self, text: str) -> str:
        if not text:
            raise ValueError("an identifier must not be empty")
        out = []
        for i, ch in enumerate(text):
            if ch in self._literal and not (i == 0 and ch in self._not_first):
                out.append(ch)
            elif ch in self._replacement:
                out.append(self._replacement[ch])
            else:
                out.extend(f"{self._escape}{b:02X}" for b in ch.encode("utf-8", "surrogatepass"))
        return "".join(out)

    def decode(self, part: str) -> str:
        raw = bytearray()
        for hex_byte, ch in self._token.findall(part):
            if hex_byte:
                raw.append(int(hex_byte, 16))
            else:
                raw += self._original.get(ch, ch).encode("utf-8", "surrogatepass")
        try:
            text = raw.decode("utf-8", "surrogatepass")
        except UnicodeDecodeError:
            text = ""
        # The scan above accepts more than ``encode`` produces (foreign characters, an escape
        # character without two upper-case hex digits, escapes of literal characters). Encoding
        # again rejects all of them in one place, instead of a second copy of the rules.
        if not text or self.encode(text) != part:
            raise ValueError(f"{part!r} is not an encoded identifier")
        return text


# FD40. Output alphabet: [A-Za-z0-9_.-@{}%]; never "/", NUL, whitespace, ":" or a glob character;
# never starts with "." or "-". "{" and "}" are not glob characters; bash and zsh expand braces only
# if they contain "," or "..", and the codec never emits ",".
_FILE_CODEC = _Codec(_ALNUM + "_.-", {":": "@", "[": "{", "]": "}"}, "%", not_first=".-")

# PI50. Output alphabet: [A-Za-z0-9_.:[]-+], a subset of the characters of a ``-k`` identifier
# (``(\w|:|\+|-|\.|\[|\]|\\|/)+`` in pytest >= 7.0). It is printable ASCII without "\", so pytest's
# ``ascii_escaped`` does not change it, and it never contains "/", the separator after the
# configuration name. A case identifier that follows CP30 and CP50 does not change.
_TEST_ID_CODEC = _Codec(_ALNUM + "_.-:[]", {}, "+")


def path_id_to_file_part(identifier: str) -> str:
    """Encode a case or call identifier for a basename (FD40)."""
    return _FILE_CODEC.encode(identifier)


def path_id_from_file_part(part: str) -> str:
    return _FILE_CODEC.decode(part)


def check_basename_length(basename: str) -> str:
    """Return ``basename`` if it has at most ``NAME_MAX`` bytes; raise ``ValueError`` otherwise."""
    if len(basename.encode()) > NAME_MAX:
        raise ValueError(f"basename is longer than {NAME_MAX} bytes: {basename!r}")
    return basename


# ---------------------------------------------------------------------------------------------
# Configuration names (TD80).

_CONFIG_NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


def check_config_name(name: str) -> str:
    """Return ``name`` unchanged if it follows TD80; raise ``ValueError`` otherwise.

    A TD80 name is already a safe directory name and a safe ``-k`` identifier: ASCII only, no
    ``/``, never ``.`` or ``..``, no leading ``.`` or ``-``. So the configuration directory name and
    the first part of a test case ID are the configuration name itself; this function only checks
    the preconditions of that identity.
    """
    if not isinstance(name, str) or len(name) > CONFIG_NAME_MAX or not _CONFIG_NAME_RE.fullmatch(name):
        raise ValueError(f"{name!r} is not a valid configuration name (TD80)")
    return name


# ---------------------------------------------------------------------------------------------
# Test case IDs (PI50).

TEST_ID_SEP = "/"


def testcase_id(config_name: str, case_id: str) -> str:
    """``<config_name>/<T(case_id)>``, the only parametrization ID of the case's pytest item."""
    return f"{check_config_name(config_name)}{TEST_ID_SEP}{_TEST_ID_CODEC.encode(case_id)}"


def parse_testcase_id(test_id: str) -> tuple[str, str]:
    """Return ``(config_name, case_id)``. Neither part can contain "/", so the split is unique."""
    config_name, sep, part = test_id.partition(TEST_ID_SEP)
    if not sep:
        raise ValueError(f"{test_id!r} is not a test case ID")
    return check_config_name(config_name), _TEST_ID_CODEC.decode(part)


# ---------------------------------------------------------------------------------------------
# Paths.

INSTANCES = ("A", "B")
_RANK_RE = re.compile(r"0|[1-9][0-9]*")
# No "_" in a content id: the last "_" of a data file name ends the encoded call id (TD75).
_CONTENT_ID_RE = re.compile(r"[A-Za-z0-9.-]+")
_RANK_LOG_RE = re.compile(r"(?P<instance>[AB])_rank_(?P<rank>0|[1-9][0-9]*)\.jsonl")
_METRICS_SUFFIX = ".jsonl"


def _check_instance_rank(instance: str, rank: int) -> None:
    if instance not in INSTANCES or not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError(f"bad instance or rank: {instance!r}, {rank!r}")


def check_roots(data_dir: str, metrics_dir: str) -> tuple[str, str]:
    """FD10: return both roots resolved; raise if they are equal or one is inside the other.

    A fresh run deletes configuration directories under the data root (PI60.3), and TM60 forbids
    deleting a metrics file. A data root inside the metrics root can be the metrics subdirectory of
    a configuration (FD20).
    """
    data, metrics = os.path.realpath(data_dir), os.path.realpath(metrics_dir)
    if os.path.commonpath([data, metrics]) in (data, metrics):
        raise ValueError(f"--xdit_data_dir {data!r} and --xdit_metrics_dir {metrics!r} must not overlap")
    return data, metrics


def config_dir(root: str, config_name: str) -> str:
    """``<root>/<config_name>``, where ``root`` is a resolved data or metrics root (FD20)."""
    return os.path.join(root, check_config_name(config_name))


def rank_log_name(instance: str, rank: int) -> str:
    _check_instance_rank(instance, rank)
    return f"{instance}_rank_{rank}.jsonl"


def parse_rank_log_name(name: str) -> tuple[str, int]:
    """Also the launcher's filter for rank logs (RL10); it never matches a data file name."""
    m = _RANK_LOG_RE.fullmatch(name)
    if m is None:
        raise ValueError(f"{name!r} is not a rank log name")
    return m["instance"], int(m["rank"])


def data_file_name(instance: str, rank: int, call_id: str, content_id: str) -> str:
    """``<instance>_<rank>_<F(call_id)>_<content_id>``; ``content_id`` includes the extension.

    The result can be longer than ``NAME_MAX``: the rank logs it in ``files=`` (RL60, step 1), and
    a ``check_basename_length`` failure before the write makes the call UNSUPPORTED (RL60, step 2).
    """
    _check_instance_rank(instance, rank)
    if not _CONTENT_ID_RE.fullmatch(content_id):
        raise ValueError(f"bad content id: {content_id!r}")
    return f"{instance}_{rank}_{path_id_to_file_part(call_id)}_{content_id}"


def parse_data_file_name(name: str) -> tuple[str, int, str, str]:
    # Instance and rank contain no "_", so the first two "_" end them; the content id contains no
    # "_", so the last "_" starts it. The encoded call id is everything between and may contain "_".
    fields = name.split("_", 2)
    if len(fields) == 3 and "_" in fields[2]:
        instance, rank, rest = fields
        call_part, content_id = rest.rsplit("_", 1)
        if instance in INSTANCES and _RANK_RE.fullmatch(rank) and _CONTENT_ID_RE.fullmatch(content_id):
            return instance, int(rank), path_id_from_file_part(call_part), content_id
    raise ValueError(f"{name!r} is not a data file name")


def metrics_file_path(metrics_root: str, config_name: str, case_id: str) -> str:
    """``<metrics_root>/<config_name>/<F(case_id)>.jsonl`` (PI20.2.2).

    Raise ``ValueError`` for a basename longer than ``NAME_MAX``; the harness calls this during
    collection, so that is a collection error (FD40).
    """
    basename = check_basename_length(path_id_to_file_part(case_id) + _METRICS_SUFFIX)
    return os.path.join(config_dir(metrics_root, config_name), basename)


def parse_metrics_file_path(path: str) -> tuple[str, str]:
    """Return ``(config_name, case_id)`` from the last two components of ``path``."""
    parent, basename = os.path.split(path)
    if not basename.endswith(_METRICS_SUFFIX):
        raise ValueError(f"{path!r} is not a metrics file path")
    case_id = path_id_from_file_part(basename[: -len(_METRICS_SUFFIX)])
    return check_config_name(os.path.basename(parent)), case_id
```
