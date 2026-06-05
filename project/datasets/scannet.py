import copy
import json

import networkx as nx
import numpy as np
import scipy as sp
import torch
import torch_geometric as torch_geo
from tqdm import tqdm


class ScanNet(torch_geo.data.Dataset):
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
    ):
        self.class_to_id = None
        self.classes = None
        self.max_distance = max_distance
        self.names_scan_split = None
        self.num_neighbors = num_neighbors
        self.path = path
        self.scans = None
        self.split = split
        self.transforms_points = transforms_points
        self.use_background = use_background

        super().__init__(path, transform, transform_pre, log=use_log, force_reload=use_reload)

        self._init()

    def _init(self):
        self.classes = json.load(open(self.path / "annotations" / "meta_data" / "scannetv2_raw_categories.json", "r", encoding="utf-8"))
        self.class_to_id = {c: i for i, c in enumerate(self.classes)}
        self.embeddings_class = json.load(open(self.path / "annotations" / "meta_data" / "cat2glove42b.json", "r"))

        self._init_split()
        self._init_scans()

    def _init_split(self):
        split_to_filenames = dict(training="scannetv2_train.txt", validation="scannetv2_val.txt", test="scannetv2_test.txt")
        path_split = self.path / f"annotations" / "splits" / split_to_filenames[self.split]

        names_scan = {x.strip() for x in open(path_split, "r", encoding="utf-8")}
        names_scan = sorted(names_scan)

        self.names_scan_split = names_scan

    def _init_scans(self):
        self.scans = self.load_scans()

    def load_scans(self):
        scans = {}
        for name_scan in tqdm(self.names_scan_split):
            scan = self.load_scan(name_scan)

            if scan is None:
                continue

            scans[name_scan] = scan

        return scans

    def load_scan(self, name_scan):
        # Load points
        path_points = self.path / "scan_data" / "pcd_with_global_alignment" / f"{name_scan}.pth"
        points, colors, _, labels = torch.load(path_points, weights_only=False)
        colors = (colors / 255.0) * 2.0 - 1.0
        points_colored = torch.from_numpy(np.concatenate([points, colors], 1))
        points_colored = points_colored.float()

        # Load labels
        path_labels = self.path / "scan_data" / "instance_id_to_label" / f"{name_scan}.pth"
        ids_instance_to_label = torch.load(path_labels, weights_only=False)

        # Create instances
        points_colored_instance = []
        ids_instance = []
        labels_instance = []
        embeddings_class = []
        for id_instance, label in ids_instance_to_label.items():
            if label not in self.classes:
                continue

            mask_is_instance = labels == id_instance

            if np.sum(mask_is_instance) == 0:
                continue

            points_colored_instance.append(points_colored[mask_is_instance])
            ids_instance.append(id_instance)
            labels_instance.append(self.class_to_id[label])
            embeddings_class.append(self.embeddings_class[label])

        # Filter background
        if not self.use_background:
            idxs_instance_selected = []
            for i, label in enumerate(labels_instance):
                if self.classes[label] in ["wall", "floor", "ceiling"]:
                    continue

                idxs_instance_selected.append(i)

            if len(idxs_instance_selected) == 0:
                return None

            points_colored_instance = [points_colored_instance[idx] for idx in idxs_instance_selected]
            ids_instance = [ids_instance[idx] for idx in idxs_instance_selected]
            labels_instance = [labels_instance[idx] for idx in idxs_instance_selected]
            embeddings_class = [embeddings_class[idx] for idx in idxs_instance_selected]

        # Create scan
        ids_instance = torch.tensor(ids_instance, dtype=torch.long)
        labels_instance = torch.tensor(labels_instance, dtype=torch.long)
        embeddings_class = torch.tensor(embeddings_class, dtype=torch.float)
        scan = dict(
            points_colored_instance=points_colored_instance,
            ids=ids_instance,
            labels=labels_instance,
            embeddings_class=embeddings_class,
        )

        return scan

    def create_graph(self, scan):
        graph = nx.Graph()

        num_nodes = len(scan["centers"])
        for i in range(num_nodes):
            graph.add_node(
                i,
                pos=scan["centers"][i],
                size=scan["sizes"][i],
                features=scan["points_colored_instance"][i],
                label=scan["labels"][i],
                embeddings_class=scan["embeddings_class"][i],
                id_instance=scan["ids"][i],
            )

        # Gaussian weighting such that weight is 0.5 if distance is max_distance.
        # Could view this as probabilistic modeling of whether the node relations, such that a thresholding at 0.5 corresponds to thresholding at max_distance.
        # Decided not to use this for creating edges though, using k-NN for now.
        std = np.sqrt(-2.0 * np.log(0.5)) * self.max_distance
        distances = sp.spatial.distance.cdist(scan["centers"], scan["centers"], metric="euclidean")
        scores = np.exp(-0.5 * (distances / std) ** 2)

        k = min(self.num_neighbors + 1, scores.shape[1])
        for i in range(num_nodes):
            idxs_neighbors = np.argpartition(scores[i], -k)[-k:]
            for j in idxs_neighbors:
                # Prevent self-loops
                if j != i:
                    graph.add_edge(i, j, weight=scores[i, j], distance=distances[i, j])

        return graph

    def len(self):
        return len(self.names_scan_split)

    def get(self, idx):
        name_scan = self.names_scan_split[idx]
        scan = self.scans[name_scan]
        scan = {k: v.clone() if isinstance(v, torch.Tensor) else [t.clone() for t in v] if isinstance(v, list) else copy.deepcopy(v) for k, v in self.scans[name_scan].items()}

        if self.transforms_points is not None:
            for transform in self.transforms_points:
                scan = transform(scan)

        graph = self.create_graph(scan)
        item = torch_geo.utils.from_networkx(graph)

        return item
