"""Every name a module's own code reads, against the ones it can see. Catches what py_compile
cannot: an annotation naming a type the module never imported."""

import ast
import builtins
import sys

for path in sys.argv[1:]:
    tree = ast.parse(open(path).read(), path)
    known = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            known.update((a.asname or a.name.split(".")[0]) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            known.update((a.asname or a.name) for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            known.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            known.add(node.id)
        elif isinstance(node, ast.arg):
            known.add(node.arg)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            known.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            known.add(node.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in known:
                print(f"{path}:{node.lineno}: undefined name {node.id!r}")
