"""Public-only metadata discovery and bounded experimental planning.

Unknown repositories are proposals, not executable code. Only the registered
adapters/corpora are automatically admitted; no private data enters queries.
"""
import json
import math
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

DATASETS = {
    "base": {"sources":["LibriSpeech SLR12","MUSAN SLR17"],"role":"train/dev/test split"},
    "rir":{"sources":["LibriSpeech SLR12","MUSAN SLR17","RIRS_NOISES SLR28"],"role":"train-only simulated room responses","license":"Apache-2.0"},
    "spanish": {"sources":["LibriSpeech SLR12","MUSAN SLR17","SLR73 Peruvian Spanish"],"role":"additional training speech only","license":"CC-BY-SA-4.0"},
}
MODELS = {
    "metricgan":"speechbrain/metricgan-plus-voicebank",
    "sepformer_dns":"speechbrain/sepformer-dns4-16k-enhancement",
    "sepformer_wham":"speechbrain/sepformer-wham16k-enhancement",
    "deepfilter":"DeepFilterNet",
}
LOSS_KEYS=("wave","spectral","sisdr")
def validate_loss(weights):
    if not isinstance(weights,dict) or set(weights)!=set(LOSS_KEYS):
        raise ValueError("Loss must use exactly the registered primitives")
    clean={}
    for key in LOSS_KEYS:
        value=weights[key]
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=2:
            raise ValueError("Invalid loss weight")
        clean[key]=float(value)
    if clean["wave"]+clean["spectral"]<=0:
        raise ValueError("Loss must retain a waveform or spectral anchor")
    return clean

def fetch(url,timeout=15):
    with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"Ralph-V4-public-research/1.0"}),timeout=timeout) as response:
        data=response.read(2000001)
        if len(data)>2000000: raise ValueError("Metadata response too large")
        return data

def discover(fetcher=fetch):
    result={"datasets":[],"models":[],"papers":[],"errors":{}}
    for kind,query in (("datasets","speech enhancement"),("models","speech enhancement")):
        url="https://huggingface.co/api/"+kind+"?"+urllib.parse.urlencode({"search":query,"limit":15,"sort":"downloads","direction":-1})
        try:
            rows=json.loads(fetcher(url))
            for row in rows:
                if not isinstance(row,dict) or row.get("private") or row.get("gated"): continue
                name=row.get("id")
                if not isinstance(name,str): continue
                tags=row.get("tags",[])
                result[kind].append({"id":name,"sha":row.get("sha"),"license":[t for t in tags if isinstance(t,str) and t.startswith("license:")],"admitted":name in MODELS.values() if kind=="models" else False})
        except Exception as exc: result["errors"][kind]=str(exc)
    try:
        url="https://export.arxiv.org/api/query?"+urllib.parse.urlencode({"search_query":"ti:speech enhancement OR abs:speech enhancement","sortBy":"submittedDate","sortOrder":"descending","max_results":8})
        root=ET.fromstring(fetcher(url))
        ns={"a":"http://www.w3.org/2005/Atom"}
        for node in root.findall("a:entry",ns):
            result["papers"].append({"id":node.findtext("a:id",default="",namespaces=ns),"title":" ".join(node.findtext("a:title",default="",namespaces=ns).split()),"abstract":" ".join(node.findtext("a:summary",default="",namespaces=ns).split())[:1000]})
    except Exception as exc: result["errors"]["papers"]=str(exc)
    return result
