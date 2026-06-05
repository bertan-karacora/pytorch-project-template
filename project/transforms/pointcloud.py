import numpy as np
import scipy as sp
import torch


def subsample_points(points, num_points):
    num_points_original = len(points)
    use_replace = num_points_original < num_points
    idxs_points = np.random.choice(num_points_original, size=num_points, replace=use_replace)
    points = points[idxs_points]

    return points


def points_to_tensor_scan_subsample(scan, num_points):
    for i in range(len(scan["points_colored_instance"])):
        scan["points_colored_instance"][i] = subsample_points(scan["points_colored_instance"][i], num_points)

    scan["points_colored_instance"] = torch.stack(scan["points_colored_instance"], dim=0)

    return scan


class PointsToTensorScanSubsample(torch.nn.Module):
    def __init__(self, num_points):
        super().__init__()

        self.num_points = num_points

    def forward(self, scan):
        scan_transformed = points_to_tensor_scan_subsample(scan, self.num_points)
        return scan_transformed


def sample_rotation(use_axis_z=False, use_axis_alignment=False):
    if not use_axis_alignment:
        if not use_axis_z:
            vec_q = np.random.normal(size=4)
            vec_q /= np.linalg.norm(vec_q)
            rotation = sp.spatial.transform.Rotation.from_quat(vec_q)
        else:
            angle = np.random.uniform(-np.pi, np.pi)
            rotation = sp.spatial.transform.Rotation.from_euler("z", angle, degrees=False)
    else:
        angles_axis_aligned = np.array([0.0, 0.25, 0.5, 0.75]) * 2.0 * np.pi
        if not use_axis_z:
            angle_x = np.random.choice(angles_axis_aligned)
            angle_y = np.random.choice(angles_axis_aligned)
            angle_z = np.random.choice(angles_axis_aligned)
            rotation = sp.spatial.transform.Rotation.from_euler("xyz", [angle_x, angle_y, angle_z], degrees=False)
        else:
            angle = np.random.choice(angles_axis_aligned)
            rotation = sp.spatial.transform.Rotation.from_euler("z", angle, degrees=False)

    mat_r = torch.from_numpy(rotation.as_matrix())

    return mat_r


def rotate_scan(scan, use_axis_z=False, use_axis_alignment=False):
    mat_r = sample_rotation(use_axis_z=use_axis_z, use_axis_alignment=use_axis_alignment)

    scan["points_colored_instance"][:, :, :3] = scan["points_colored_instance"][:, :, :3] @ mat_r.T.float()

    return scan


class RotationScan(torch.nn.Module):
    def __init__(self, prob=1.0, use_axis_z=False, use_axis_alignment=False):
        super().__init__()

        self.prob = prob
        self.use_axis_alignment = use_axis_alignment
        self.use_axis_z = use_axis_z

    def forward(self, scan):
        scan_transformed = scan
        if np.random.rand() < self.prob:
            scan_transformed = rotate_scan(scan_transformed, use_axis_z=self.use_axis_z, use_axis_alignment=self.use_axis_alignment)

        return scan_transformed


def rotate_object(scan, idx):
    mat_r = sample_rotation()
    scan["points_colored_instance"][idx, :, :3] = scan["points_colored_instance"][idx, :, :3] @ mat_r.T.float()

    return scan


class RotationObjects(torch.nn.Module):
    def __init__(self, prob=1.0):
        super().__init__()

        self.prob = prob

    def forward(self, scan):
        scan_transformed = scan
        for i in range(len(scan["points_colored_instance"])):
            if np.random.rand() < self.prob:
                scan_transformed = rotate_object(scan_transformed, idx=i)

        return scan_transformed


def translate_object(scan, scale, idx):
    scan["points_colored_instance"][idx, :, :3] += torch.randn(3) * scale

    return scan


class TranslationObjectsScan(torch.nn.Module):
    def __init__(self, prob=1.0, scale=0.2):
        super().__init__()

        self.scale = scale
        self.prob = prob

    def forward(self, scan):
        scan_transformed = scan
        for i in range(len(scan["points_colored_instance"])):
            if np.random.rand() < self.prob:
                scan_transformed = translate_object(scan, scale=self.scale, idx=i)

        return scan_transformed


def jitter_object(scan, scale, idx):
    scan["points_colored_instance"][idx, :, 3:] += (torch.randn(len(scan["points_colored_instance"][idx]), 3) - 0.5) * scale
    # TODO: Need clipping?

    return scan


class ColorJitterScan(torch.nn.Module):
    def __init__(self, prob=1.0, scale=0.1):
        super().__init__()

        self.scale = scale
        self.prob = prob

    def forward(self, scan):
        scan_transformed = scan
        for i in range(len(scan["points_colored_instance"])):
            if np.random.rand() < self.prob:
                scan_transformed = jitter_object(scan_transformed, scale=self.scale, idx=i)

        return scan_transformed


def create_boxes_scan_axis_aligned(scan):
    mins = torch.min(scan["points_colored_instance"][:, :, :3], dim=1).values
    maxs = torch.max(scan["points_colored_instance"][:, :, :3], dim=1).values

    scan["centers"] = (mins + maxs) / 2.0
    scan["sizes"] = maxs - mins

    return scan


class CreateBoxesScanAxisAligned(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scan):
        scan_transformed = create_boxes_scan_axis_aligned(scan)
        return scan_transformed


def center_points_scan(scan):
    scan["points_colored_instance"][:, :, :3].sub_(torch.mean(scan["points_colored_instance"][:, :, :3], dim=1, keepdim=True))

    return scan


class PointsCenterScan(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scan):
        scan_transformed = center_points_scan(scan)
        return scan_transformed


def normalize_scale_scan(scan):
    max_dist = torch.max(torch.sqrt(torch.sum((scan["points_colored_instance"][:, :, :3] ** 2), dim=2)), dim=1).values
    max_dist.clamp_(min=1e-6)

    scan["points_colored_instance"][:, :, :3].div_(max_dist[:, None, None])

    return scan


class PointsNormalizeScaleScan(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scan):
        scan_transformed = normalize_scale_scan(scan)
        return scan_transformed
