"""V3 research policy: flexible neural research, fixed trusted evaluation shell."""
import ast
import math

# Research code is not a security sandbox. Keep execution isolated and review
# generated candidates. No file/network access or validation access in hooks.
PROTECTED = {"seed_everything", "locate_data_root", "SpeechPairs", "ensure_pesq",
             "evaluate", "write_csv", "run", "self_test"}
CONSTANTS = {"EXPERIMENT_CONFIG", "DATASET_SLUG", "SAMPLE_RATE", "WORK"}
HOOKS = {"build_optimizer", "train_batch", "finish_epoch", "augment_train"}
SAFE_IMPORTS = {"math", "torch", "numpy", "collections", "functools", "itertools"}
FORBIDDEN = {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr",
             "load", "save", "read_text", "write_text", "read_bytes", "write_bytes",
             "system", "popen", "run", "connect", "request", "urlopen", "download",
             "from_file", "memmap", "fromfile", "tofile", "load_state_dict_from_url",
             "globals", "locals", "vars", "get_ipython", "settrace"}
BAD_NAMES = {"WORK", "DATASET_SLUG", "SpeechPairs", "locate_data_root", "evaluate",
             "ensure_pesq", "val_loader", "val_set", "validation", "subprocess",
             "sys", "Path", "sf", "pd", "os", "socket", "importlib"}

def name_of(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None

def nodes(source):
    tree = ast.parse(source)
    named = {}
    for n in tree.body:
        name = name_of(n)
        if name:
            if name in named:
                raise ValueError("Duplicate top-level name: " + name)
            named[name] = n
    return tree, named

def dump(node):
    return ast.dump(node, include_attributes=False)

def validate(candidate, before, family_class):
    old, a = nodes(before)
    new, b = nodes(candidate)
    if dump(old) == dump(new):
        raise ValueError("No substantive research change")
    for name in PROTECTED | CONSTANTS:
        if name not in a or name not in b or dump(a[name]) != dump(b[name]):
            raise ValueError("Protected evaluation/data/runner modified: " + name)
    for required in (family_class, "enhancement_loss", *HOOKS):
        if required not in b:
            raise ValueError("Required research interface missing: " + required)
    # Keep original imports, main guard and other executable top-level statements.
    def fixed(tree):
        return [dump(n) for n in tree.body if name_of(n) is None and not isinstance(n, (ast.Import, ast.ImportFrom))]
    if fixed(old) != fixed(new):
        raise ValueError("Protected top-level execution modified")
    reserved = PROTECTED | CONSTANTS | {"torch", "np", "nn", "F", "DataLoader", "Dataset", "json", "csv", "random", "time", "math", "sys", "Path", "pd", "sf", "subprocess", "importlib", "zipfile"}
    for n in new.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for alias in n.names:
                if alias.name == "*" or ((alias.asname or alias.name.split(".")[0]) in reserved and dump(n) not in {dump(x) for x in old.body}):
                    raise ValueError("Import shadows protected name")
    if any(name in reserved and name not in a for name in b):
        raise ValueError("Research definition shadows trusted dependency")
    old_imports = {dump(n) for n in old.body if isinstance(n, (ast.Import, ast.ImportFrom))}
    new_imports = {dump(n) for n in new.body if isinstance(n, (ast.Import, ast.ImportFrom))}
    if not old_imports <= new_imports:
        raise ValueError("Original imports removed")
    for n in new.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)) and dump(n) not in old_imports:
            modules = [x.name for x in n.names] if isinstance(n, ast.Import) else [n.module or ""]
            if any(x.split(".")[0] not in SAFE_IMPORTS for x in modules):
                raise ValueError("Unapproved research import")
    edits = []
    for name, n in b.items():
        if name in PROTECTED | CONSTANTS:
            continue
        if name in a and dump(a[name]) == dump(n):
            continue
        edits.append(name)
        if isinstance(n, ast.Assign):
            try:
                ast.literal_eval(n.value)
            except (ValueError, TypeError):
                raise ValueError("Research constants must be literal values")
        elif not isinstance(n, (ast.ClassDef, ast.FunctionDef)):
            raise ValueError("Unsupported research definition")
        for item in ast.walk(n):
            if isinstance(item, ast.Attribute) and isinstance(item.ctx, (ast.Store, ast.Del)) and isinstance(item.value, ast.Name) and item.value.id != "self":
                raise ValueError("External attribute mutation prohibited")
            if isinstance(item, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
                raise ValueError("Nested import/global prohibited")
            if isinstance(item, ast.Name) and item.id in BAD_NAMES:
                raise ValueError("Protected resource access: " + item.id)
            if isinstance(item, ast.Attribute) and item.attr.startswith("__") and item.attr != "__init__":
                raise ValueError("Dunder access prohibited")
            if isinstance(item, ast.Call):
                fn = item.func
                call = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
                if call in FORBIDDEN:
                    raise ValueError("Forbidden research call: " + call)
    if not edits and not (set(a) - set(b)):
        raise ValueError("No research change")
    return edits

def upgrade(source, family_class):
    """Migrate a legacy trainer to protected runner + train-only research hooks."""
    if "def train_batch(" in source:
        return source
    start = source.index("    optimizer = ", source.index("def run():"))
    end = source.index("    best_score, best_epoch", start)
    opt = source[start:end]
    has_scheduler = "scheduler =" in opt
    opt = opt.replace('cfg["epochs"]', 'epochs')
    hooks = ("def build_optimizer(model, epochs):\n" + opt +
             ("    return optimizer, scheduler\n" if has_scheduler else "    return optimizer, None\n") +
             "\n\ndef augment_train(noisy, clean):\n    return noisy, clean\n"
             "\n\ndef train_batch(model, optimizer, noisy, clean, variant):\n"
             "    noisy, clean = augment_train(noisy, clean)\n"
             "    optimizer.zero_grad(set_to_none=True)\n"
             "    pred = model(noisy)\n"
             "    loss = enhancement_loss(pred, clean, variant)\n"
             "    if not torch.isfinite(loss):\n        raise RuntimeError('Non-finite loss')\n"
             "    loss.backward()\n    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)\n"
             "    optimizer.step()\n    return float(loss.detach())\n"
             "\n\ndef finish_epoch(scheduler):\n    if scheduler is not None:\n        scheduler.step()\n\n\n")
    source = source[:start] + '    optimizer, scheduler = build_optimizer(model, int(cfg["epochs"]))\n' + source[end:]
    start = source.index("            optimizer.zero_grad", source.index("def run():"))
    end = source.index("        val_pesq, _ = evaluate", start)
    source = source[:start] + "            losses.append(train_batch(model, optimizer, noisy, clean, variant))\n        finish_epoch(scheduler)\n" + source[end:]
    source = source.replace("def run():", hooks + "def run():", 1)
    # Replace architecture-specific smoke assertions with fixed general contract.
    start = source.index("def self_test():")
    end = source.index('if __name__ == "__main__":', start)
    source = source[:start] + f'''def self_test():
    torch.manual_seed(42)
    model = {family_class}()
    assert isinstance(model, nn.Module)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0
    noisy = torch.randn(1, 4096) * 0.1
    clean = noisy * 0.9
    out = model(noisy)
    assert out.shape == noisy.shape and torch.isfinite(out).all()
    loss = enhancement_loss(out, clean, EXPERIMENT_CONFIG["id"])
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer, scheduler = build_optimizer(model, 2)
    assert np.isfinite(train_batch(model, optimizer, noisy, clean, EXPERIMENT_CONFIG["id"]))
    finish_epoch(scheduler)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 48000) * 0.1)
    assert out.shape == (1, 48000) and torch.isfinite(out).all()
    print("SELF_TEST_PASSED: train hooks, noisy-only inference, finite gradients, sample count")


''' + source[end:]
    # Explicitly require the original complete validation split.
    source = source.replace('    val_set = SpeechPairs(root, "validation")', '    val_set = SpeechPairs(root, "validation")\n    if len(val_set) != 164:\n        raise RuntimeError("Expected all 164 validation clips")', 1)
    # Validation callback receives no reference; train hooks receive only train batches.
    ast.parse(source)
    return source
