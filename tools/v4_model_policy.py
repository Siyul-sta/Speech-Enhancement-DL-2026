"""Validate a generated model-only fragment before embedding in the fixed runner.
This is a conservative research policy, not a security sandbox.
"""
import ast
import hashlib

def validate_model(code):
    tree=ast.parse(code)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.ClassDef):
        raise ValueError("Exactly one model class required")
    cls=tree.body[0]
    if cls.name!="GeneratedEnhancer" or len(cls.bases)!=1 or ast.unparse(cls.bases[0])!="nn.Module" or cls.decorator_list or cls.keywords:
        raise ValueError("Expected GeneratedEnhancer(nn.Module)")
    if not {"__init__","forward"} <= {n.name for n in cls.body if isinstance(n,ast.FunctionDef)}:
        raise ValueError("Required model interface missing")
    forbidden={"open","eval","exec","compile","getattr","setattr","globals","locals","vars","__import__","load","save","from_file","fromfile","tofile","read_text","write_text","system","popen","request","urlopen","connect","download","register_forward_hook","register_forward_pre_hook"}
    allowed={"nn","torch","F","math","self","super","range","len","int","float","bool","list","tuple","zip","enumerate","min","max","abs","sum","isinstance","ValueError","RuntimeError","True","False","None"}
    locals_={n.id for n in ast.walk(cls) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
    locals_|={n.arg for n in ast.walk(cls) if isinstance(n,ast.arg)}
    for n in ast.walk(cls):
        if isinstance(n,(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,ast.Delete,ast.With,ast.AsyncFunctionDef,ast.Lambda)):
            raise ValueError("Forbidden generated syntax")
        if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id not in allowed|locals_:
            raise ValueError("Unexpected name "+n.id)
        if isinstance(n,ast.Attribute):
            if n.attr in forbidden or (n.attr.startswith("__") and n.attr!="__init__"):
                raise ValueError("Forbidden attribute "+n.attr)
            if isinstance(n.ctx,ast.Store) and not (isinstance(n.value,ast.Name) and n.value.id=="self"):
                raise ValueError("External mutation")
        if isinstance(n,ast.FunctionDef) and (n.decorator_list or n.args.defaults or n.args.kw_defaults and any(x is not None for x in n.args.kw_defaults)):
            raise ValueError("Unsupported method decoration/default")
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in forbidden:
            raise ValueError("Forbidden call")
    if any(not isinstance(n,ast.FunctionDef) for n in cls.body):
        raise ValueError("Class body must contain methods only")
    compile(tree,"generated_model.py","exec")
    return hashlib.sha256(code.encode("utf-8")).hexdigest()

def embed(source,code):
    validate_model(code)
    marker="class GeneratedEnhancer(nn.Module):\n    def __init__(self):\n        super().__init__()\n        raise RuntimeError('Generated model not supplied')\n"
    if source.count(marker)!=1:
        raise ValueError("Missing exact generated model slot")
    result=source.replace(marker,code.rstrip()+"\n",1)
    compile(result,"run.py","exec")
    return result
