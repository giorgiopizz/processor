import awkward as ak
import tracemalloc
import glob
from processor.nanoevents import nanoaod
from processor.mixins import mixins
from processor.processor import Processor
import gc

files = glob.glob("rootFiles/*.root")


def mask_eval(arr, arrToSave, mask, to_eval, default=-9999.0):
    # utility function to evaluate a function on a masked array
    # different from ak.mask because it won't ever use masked values
    out = ak.ones_like(arrToSave) * default
    _arr = arr[mask]
    out_real = to_eval(_arr)
    try:
        nj = ak.num(out)
        x = ak.flatten(out).to_numpy()
        x[ak.flatten(mask).to_numpy()] = out_real.to_numpy()
        return ak.unflatten(ak.from_numpy(x), nj)
    except Exception:
        x = out.to_numpy()
        x[mask.to_numpy()] = out_real.to_numpy()
        return ak.from_numpy(x)


def process(events):
    # real function that process events
    events = events[
        (ak.num(events.Jet) >= 2)
        & ((ak.num(events.Electron) >= 2) | (ak.num(events.Muon) >= 2))
    ]
    # events["mjj"] = mask_eval(
    #     events.Jet,
    #     events.run,
    #     ak.num(events.Jet) >= 2,
    #     lambda j: (j[:, 0] + j[:, 1]).mass,
    # )
    events["mjj"] = (events.Jet[:, 0] + events.Jet[:, 1]).mass

    return events


not_load = [
    "CaloMET",
    "ChsMET",
    "CorrT1METJet",
    "DeepMETRes*",
    "Fat*",
    "Flag",
    "FsrPhoton",
    # "Gen*",
    "HLT*",
    "HTXS",
    "IsoTrack",
    "L1*",
    "Low*",
    "OtherPV",
    # "PV",
    # "SV",
    "Soft*",
    "Sub*",
    "genTtbarId",
]

processor = Processor(
    files[:1],
    not_load,
    process,
    chuncksize=100_000,  # max_events=10_000_000,
    mixins=mixins,
    behavior=nanoaod.behavior,
)
tracemalloc.start()
a = tracemalloc.get_traced_memory()
print("Curr mem", a[0] / 1024**2, "MB")
processor()
del processor
gc.collect()
a = tracemalloc.get_traced_memory()
print("Curr mem", a[0] / 1024**2, "MB")
print("Max mem", a[1] / 1024**2, "MB")
