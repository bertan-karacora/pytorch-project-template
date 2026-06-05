import jsonlines
import torch
import torch_geometric as torch_geo

from project.datasets.scannet import ScanNet


class NR3DReferral(ScanNet):
    def __init__(
        self,
        path,
        split="training",
        transform=None,
        transform_pre=None,
        transforms_points=None,
        use_log=False,
        use_reload=False,
        use_background=False,
        max_distance=2.0,
        num_neighbors=5,
        use_sr3d_plus=True,
    ):
        self.annotations = None
        self.names_scan = None
        self.use_sr3d_plus = use_sr3d_plus

        super().__init__(
            path,
            split,
            transform,
            transform_pre,
            transforms_points,
            use_log,
            use_reload,
            use_background,
            max_distance,
            num_neighbors,
        )

    def _init(self):
        super()._init()

        self._init_annotations()

    def _init_annotations(self):
        self.annotations = self.load_annotations()

        names_scan = []
        for annotation in self.annotations:
            names_scan.append(annotation["scan_id"])
        names_scan = sorted(names_scan)

        self.names_scan = names_scan

    def load_annotations(self):
        annotations = []

        path_annotations = self.path / "annotations" / "refer" / f"nr3d.jsonl"
        with jsonlines.open(path_annotations, "r") as stream:
            for item in stream:
                if item["scan_id"] in self.names_scan_split and item["instance_type"] not in ["wall", "floor", "ceiling"] and len(item["tokens"]) <= 24:
                    annotations.append(item)

        if self.use_sr3d_plus and self.split == "training":
            path_annotations = self.path / "annotations" / "refer" / f"sr3d+.jsonl"
            with jsonlines.open(path_annotations, "r") as stream:
                for item in stream:
                    if item["scan_id"] in self.names_scan_split and item["instance_type"] not in ["wall", "floor", "ceiling"] and len(item["tokens"]) <= 24:
                        annotations.append(item)

        return annotations

    def len(self):
        return len(self.names_scan)

    def get(self, idx):
        annotation = self.annotations[idx]

        name_scan = annotation["scan_id"]
        scan = self.scans[name_scan]
        scan = {k: v.clone() if isinstance(v, torch.Tensor) else [t.clone() for t in v] if isinstance(v, list) else copy.deepcopy(v) for k, v in self.scans[name_scan].items()}

        id_target = int(annotation["target_id"])
        idx_target = (scan["ids"] == id_target).nonzero(as_tuple=True)[0]

        label_target = scan["labels"][idx_target]
        label_target = torch.tensor([label_target], dtype=torch.long)

        if self.transforms_points is not None:
            for transform in self.transforms_points:
                scan = transform(scan)

        graph = self.create_graph(scan)
        item = torch_geo.utils.from_networkx(graph)

        item["idx_item"] = annotation["item_id"]
        item["sentence"] = annotation["utterance"]
        item["idx_target"] = idx_target
        item["label_target"] = label_target

        return item
