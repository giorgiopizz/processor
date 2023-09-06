import uproot
import awkward as ak
import fnmatch
import sys
import gc
from math import ceil


class Processor:
    def __init__(
        self,
        files,
        not_load,
        process,
        output_file="output.root",
        chuncksize=100_000,
        max_events=10_000_000,
        mixins={},
        behavior=None,
    ):
        self.files = files
        self.not_load = not_load
        self.process = process
        self.output_file = output_file
        self.chunksize = chuncksize
        self.max_events = max_events
        self.mixins = mixins
        self.behavior = behavior

    @staticmethod
    def read_array(tree, branch_name, start, stop):
        interp = tree[branch_name].interpretation
        interp._forth = True
        return tree[branch_name].array(
            interp,
            entry_start=start,
            entry_stop=stop,
            decompression_executor=uproot.source.futures.TrivialExecutor(),
            interpretation_executor=uproot.source.futures.TrivialExecutor(),
        )

    @staticmethod
    def should_not_load(branch_name, not_load):
        if branch_name in not_load:
            return True
        for nl in not_load:
            if fnmatch.fnmatch(branch_name, nl):
                return True
        return False

    @staticmethod
    def read_events(tree, start=0, stop=100, not_load=[], mixins={}, behavior=None):
        start = min(start, tree.num_entries)
        stop = min(stop, tree.num_entries)
        if start >= stop:
            return ak.Array([])
        branches = [k.name for k in tree.branches]
        base_branches = set(list(map(lambda k: k.split("_")[0], branches)))
        base_branches = list(filter(lambda k: not k.startswith("n"), base_branches))
        events = {}
        for coll in base_branches:
            if Processor.should_not_load(coll, not_load):
                continue
            d = {}
            coll_branches = list(
                map(
                    lambda k: "_".join(k.split("_")[1:]),
                    list(filter(lambda k: k.startswith(coll + "_"), branches)),
                )
            )

            if len(coll_branches) == 0:
                # print(f"Collection {coll} is not a collection but a signle branch")
                events[coll] = Processor.read_array(tree, coll, start, stop)
                continue

            for branch in coll_branches:
                if Processor.should_not_load(coll + "_" + branch, not_load):
                    continue
                d[branch] = Processor.read_array(tree, coll + "_" + branch, start, stop)

            if len(d.keys()) == 0:
                print("did not find anything for", coll, file=sys.stderr)
                continue

            if coll not in mixins.keys():
                # print("missing collection mixin", coll, file=sys.stderr)
                events[coll] = ak.zip(d)
                continue

            if behavior:
                events[coll] = ak.zip(d, with_name=mixins[coll], behavior=behavior)
            else:
                events[coll] = ak.zip(d, with_name=mixins[coll])
            del d

        # f.close()
        _events = ak.zip(events, depth_limit=1)
        del events
        return _events

    @staticmethod
    def unpack_events(events):
        d = {}
        for branch in events.fields:
            d[branch] = events[branch]
        return d

    def __call__(self):
        fout = uproot.recreate(self.output_file)

        for file in self.files:
            f = uproot.open(file)
            tree = f["Events"]
            max_events = min(tree.num_entries, self.max_events)
            nIterations = ceil(max_events / self.chunksize)

            for i in range(nIterations):
                events = Processor.read_events(
                    tree,
                    start=self.chunksize * i,
                    stop=self.chunksize * (i + 1),
                    not_load=self.not_load,
                    mixins=self.mixins,
                    behavior=self.behavior,
                )
                if len(events) == 0:
                    break
                events = self.process(events)
                if len(events) == 0:
                    continue
                d = Processor.unpack_events(events)
                if "Events" not in fout:
                    fout["Events"] = d
                else:
                    fout["Events"].extend(d)
                del d
                del events
                gc.collect()
            del tree
            f.close()

        fout.close()
        # gc.collect()
