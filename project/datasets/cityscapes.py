import logging
from pathlib import Path

import torch

import self_supervised_learning_of_depth_and_motion.libs.utils_io as utils_io

_LOGGER = logging.getLogger(__name__)


class Cityscapes(torch.utils.data.Dataset):
    """The Cityscapes dataset.
    The folder structure of the Cityscapes dataset: {path}/{type_data}[_sequence]/{split}/{name_city}/{name_city}_{id_sequence:0>6}_{id_frame:0>6}_{type_data}.{suffix}
    Note: This does not hold for all city directories. In some cases id_sequence and id_frame cannot be directly identified using this scheme

    See: http://www.cityscapes-dataset.com/
    See: https://github.com/mcordts/cityscapesScripts
    """

    # id 19 corresponds to 20th frame
    id_frame_annotated = 19
    names_city_split = {
        "training": ["aachen", "bochum", "bremen", "cologne", "darmstadt", "dusseldorf", "erfurt", "hamburg", "hanover", "jena"]
        + ["krefeld", "monchengladbach", "strasbourg", "stuttgart", "tubingen", "ulm", "weimar", "zurich"],
        "validation": ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"],
        "test": ["frankfurt", "lindau", "munster"],
    }
    names_split = {
        "training": "train",
        "validation": "val",
        "test": "test",
    }
    num_frames_sequence = 30
    # Shape needed along camera parameters
    shape_image = (3, 1024, 2048)
    subpath_images_left_8bit = Path(".") / "leftImg8bit"
    subpath_images_sequence_left_8bit = Path(".") / "leftImg8bit_sequence"
    subpath_parameters_camera = Path(".") / "odometry" / "camera"

    def __init__(self, path, split, transform_features=None, transform_targets=None, use_download=False, use_intrinsics=False):
        self.descriptors_item = []
        self.path = Path(path)
        self.path_ids_items = None
        self.split = split
        self.transform_features = transform_features
        self.transform_targets = transform_targets
        self.use_download = use_download
        self.use_intrinsics = use_intrinsics

        self._init()

        _LOGGER.info(f"Initialized dataset:\n{self}")

    def _init(self):
        if self.use_download:
            self.download()

        self.path_ids_item = self.path / f"{self.names_split[self.split]}.txt"
        if not self.path_ids_item.exists():
            _LOGGER.warning(f"Split ids file {self.path_ids_item} not found.")

            self.create_file_split()

        self._init_item_descriptors()

    def create_file_split(self):
        path_dir_images_split = self.path / self.subpath_images_left_8bit / self.names_split[self.split]
        with open(self.path_ids_item, "w") as file_ids_item:
            for path_dir_images_city in sorted(path_dir_images_split.iterdir()):
                if path_dir_images_city.is_dir():
                    for path_image in sorted(path_dir_images_city.iterdir()):
                        name_city, id_sequence_str, id_frame_str, name_type_data = path_image.stem.split("_")
                        file_ids_item.write(f"{name_city}/{name_city}_{id_sequence_str}_{id_frame_str}\n")

        _LOGGER.info(f"Created split ids file: {self.path_ids_item}")

    def _init_item_descriptors(self):
        with open(self.path_ids_item) as file_ids_item:
            for id_item_str in file_ids_item:
                # Use this format since this is already provided
                name_city, id_sequence_str, id_frame_str = id_item_str.strip().split("/")[1].split("_")

                subpath_image = self.subpath_images_left_8bit / self.names_split[self.split] / name_city / f"{name_city}_{id_sequence_str}_{id_frame_str}_leftImg8bit.png"
                subpath_parameters_camera = self.subpath_parameters_camera / self.names_split[self.split] / name_city / f"{name_city}_{id_sequence_str}_{id_frame_str}_camera.json"
                # Save paths as strings for lower memory cost compared to path object
                self.descriptors_item += [(str(subpath_image), str(subpath_parameters_camera))]

    def __len__(self):
        length = len(self.descriptors_item)
        return length

    def __getitem__(self, index):
        descriptor_item = self.descriptors_item[index]
        image, intrinsics_camera = self.load_item(descriptor_item)

        if self.transform_features is not None:
            image = self.transform_features(image)

        item = {"image": image, "intrinsics_camera": intrinsics_camera}
        return item

    def __str__(self):
        s = f"""Dataset {self.__class__.__name__}
    Number of samples: {len(self)}
    Path: {self.path}
    Split: {self.split}
    Transform of features: {self.transform_features}
    Transform of targets: {self.transform_targets}"""
        return s

    def load_item(self, descriptor_item):
        subpath_image_str, subpath_parameters_camera_str = descriptor_item

        path_image = self.path / subpath_image_str
        image = utils_io.read_image(path_image)

        path_parameters_camera = self.path / subpath_parameters_camera_str
        parameters_camera = utils_io.read_json(path_parameters_camera)
        intrinsics_camera = parameters_camera["intrinsic"]

        return image, intrinsics_camera

    def download(self):
        raise NotImplementedError


class CityscapesSequence(Cityscapes):
    def __init__(self, path, split, transform_features=None, transform_targets=None, use_download=False):
        path = path.parent / "Cityscapes".lower()
        super().__init__(path, split, transform_features, transform_targets, use_download)

    def _init_item_descriptors(self):
        with open(self.path_ids_item) as file_ids_item:
            for id_item_str in file_ids_item:
                # Use this format since this is already provided
                name_city, id_sequence_str, id_frame_str = id_item_str.strip().split("/")[1].split("_")
                id_frame = int(id_frame_str)

                # Assume that camera parameters of annotated frame correspond to entire sequence.
                subpath_parameters_camera = self.subpath_parameters_camera / self.names_split[self.split] / name_city / f"{name_city}_{id_sequence_str}_{id_frame_str}_camera.json"

                for offset_id_frame in range(-self.id_frame_annotated, self.num_frames_sequence - self.id_frame_annotated):
                    subpath_image = self.subpath_images_sequence_left_8bit / self.names_split[self.split] / name_city / f"{name_city}_{id_sequence_str}_{id_frame+offset_id_frame:0>6}_leftImg8bit.png"
                    if self.split == "training" or offset_id_frame == 0:
                        # Save paths as strings for lower memory cost compared to path object
                        descriptor_item = (str(subpath_image), str(subpath_parameters_camera))
                        self.descriptors_item += [descriptor_item]


class CityscapesTriplet(Cityscapes):
    def __init__(self, path, split, transform_features=None, transform_targets=None, use_download=False, stride=1, dilation=1):
        self.dilation = dilation
        self.stride = stride

        path = path.parent / "Cityscapes".lower()
        super().__init__(path, split, transform_features, transform_targets, use_download)

    def _init_item_descriptors(self):
        with open(self.path_ids_item) as file_ids_item:
            for id_item_str in file_ids_item:
                # Use this format since this is already provided
                name_city, id_sequence_str, id_frame_str = id_item_str.strip().split("/")[1].split("_")
                id_frame = int(id_frame_str)

                # Assume that camera parameters of annotated frame correspond to entire sequence.
                subpath_parameters_camera = self.subpath_parameters_camera / self.names_split[self.split] / name_city / f"{name_city}_{id_sequence_str}_{id_frame_str}_camera.json"

                descriptors_item_single = []
                for offset_id_frame in range(-self.id_frame_annotated, self.num_frames_sequence - self.id_frame_annotated):
                    subpath_image = self.subpath_images_sequence_left_8bit / self.names_split[self.split] / name_city / f"{name_city}_{id_sequence_str}_{id_frame+offset_id_frame:0>6}_leftImg8bit.png"
                    if self.split == "training" or offset_id_frame == 0:
                        # Save paths as strings for lower memory cost compared to path object
                        descriptor_item_single = (str(subpath_image), str(subpath_parameters_camera))
                        descriptors_item_single += [descriptor_item_single]

                # Ignore frames at borders of sequences. For some cities the sequences are non-contiguous.
                for i in range(self.dilation, len(descriptors_item_single) - self.dilation, self.stride):
                    triplet = (descriptors_item_single[i - self.dilation], descriptors_item_single[i], descriptors_item_single[i + self.dilation])
                    self.descriptors_item += [triplet]

    def __getitem__(self, index):
        images = []
        intrinsics_cameras = []
        for descriptor_item in self.descriptors_item[index]:
            image, intrinsics_camera = self.load_item(descriptor_item)
            images += [image]
            intrinsics_cameras += [intrinsics_camera]

        if self.transform_features is not None:
            for i, image in enumerate(images):
                images[i] = self.transform_features(image)

        item = {
            "image_source_before": images[0],
            "image_target": images[1],
            "image_source_after": images[2],
            "intrinsics_camera_source_before": intrinsics_camera[0],
            "intrinsics_camera_target": intrinsics_camera[1],
            "intrinsics_camera_source_after": intrinsics_camera[2],
        }
        return item


#     intrinsics = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]])

#     image = utils_io.read_image(frame_path)
#     zoom_y = self.img_height / image.shape[0]
#     zoom_x = self.img_width / image.shape[1]

#     intrinsics[0] *= zoom_x
#     intrinsics[1] *= zoom_y
#     return intrinsics


# def load_speed(self, city, scene_id, frame_id):
#     vehicle_folder = self.dataset_dir / "vehicle_sequence" / self.split / city.basename()
#     vehicle_file = vehicle_folder / f"{city.basename()}_{frame_id}_{scene_id}_vehicle.json"
#     vehicle = utils_io.read_json(vehicle_file)
#     return vehicle["speed"]


# class CityscapesTriplets(Cityscapes):
#     def __init__(self, *args, **kwargs):
#         self.indices = None
#         self.targets = None

#         super().__init__(*args, **kwargs)

#     def _init(self):
#         super()._init()

#         self.targets = torch.as_tensor([target for _, target in self.dataset_tv])
#         self.indices = torch.arange(len(self))

#     def __getitem__(self, index):
#         features_anchor, target_anchor = self.dataset_tv[index]

#         indices_positive_all = self.indices[self.targets == target_anchor]
#         indices_positive = indices_positive_all[indices_positive_all != index] if self.use_remove_single_occurances else indices_positive_all
#         indices_negative = self.indices[self.targets != target_anchor]

#         index_positive = random.choice(indices_positive).item()
#         index_negative = random.choice(indices_negative).item()
#         features_positive, target_positive = self.dataset_tv[index_positive]
#         features_negative, target_negative = self.dataset_tv[index_negative]

#         features = dict(anchor=features_anchor, positive=features_positive, negative=features_negative)
#         targets = dict(anchor=target_anchor, positive=target_positive, negative=target_negative)
#         return features, targets

#     def collect_scenes(self, city):
#         img_files = sorted(city.files("*.png"))
#         scenes = {}
#         connex_scenes = {}
#         connex_scene_data_list = []
#         for f in img_files:
#             frame_id, scene_id = f.basename().split("_")[1:3]
#             if scene_id not in scenes.keys():
#                 scenes[scene_id] = []
#             scenes[scene_id].append(frame_id)

#         # divide scenes into connexe sequences
#         for scene_id in scenes.keys():
#             previous = None
#             connex_scenes[scene_id] = []
#             for id in scenes[scene_id]:
#                 if previous is None or int(id) - int(previous) > 1:
#                     current_list = []
#                     connex_scenes[scene_id].append(current_list)
#                 current_list.append(id)
#                 previous = id

#         # create scene data dicts, and subsample scene every two frames
#         for scene_id in connex_scenes.keys():
#             intrinsics = self.load_intrinsics(city, scene_id)
#             for subscene in connex_scenes[scene_id]:
#                 frame_speeds = [self.load_speed(city, scene_id, frame_id) for frame_id in subscene]
#                 connex_scene_data_list.append(
#                     {"city": city, "scene_id": scene_id, "rel_path": city.basename() + "_" + scene_id + "_" + subscene[0] + "_0", "intrinsics": intrinsics, "frame_ids": subscene[0::2], "speeds": frame_speeds[0::2]}
#                 )
#                 connex_scene_data_list.append(
#                     {"city": city, "scene_id": scene_id, "rel_path": city.basename() + "_" + scene_id + "_" + subscene[0] + "_1", "intrinsics": intrinsics, "frame_ids": subscene[1::2], "speeds": frame_speeds[1::2]}
#                 )
#         return connex_scene_data_list

#     def get_scene_imgs(self, scene_data):
#         cum_speed = np.zeros(3)
#         # print(scene_data['city'].basename(), scene_data['scene_id'], scene_data['frame_ids'])
#         for i, frame_id in enumerate(scene_data["frame_ids"]):
#             cum_speed += scene_data["speeds"][i]
#             speed_mag = np.linalg.norm(cum_speed)
#             if speed_mag > self.min_speed:
#                 yield {"img": self.load_image(scene_data["city"], scene_data["scene_id"], frame_id), "id": frame_id}
#                 cum_speed *= 0


# # Crop out the bottom 25% of the image to remove the car logo
# self.crop_bottom = crop_bottom
# # img_height=171, img_width=416):  # Get rid of the car logo
# self.min_speed = 2

# def crop_intrinsics(intrinsics, offset_height, offset_width, target_height, target_width):
#     """Crops camera intrinsics based on target image dimensions and offset.

#     Args:
#       intrinsics: 1-d array containing w, h, fx, fy, x0, y0.
#       offset_height: amount of offset in y direction.
#       offset_width: amount of offset in x direction.
#       target_height: target height of images.
#       target_width: target width of images.

#     Returns:
#       A 1-d tensor containing the adjusted camera intrinsics.
#     """
#     with tf.name_scope("crop_intrinsics"):
#         w, h, fx, fy, x0, y0 = tf.unstack(intrinsics)

#         x0 -= tf.cast(offset_width, tf.float32)
#         y0 -= tf.cast(offset_height, tf.float32)

#         w = tf.cast(target_width, tf.float32)
#         h = tf.cast(target_height, tf.float32)

#         return tf.stack((w, h, fx, fy, x0, y0))


# def resize_intrinsics(intrinsics, target_size):
#     """Transforms camera intrinsics when image is resized.

#     Args:
#       intrinsics: 1-d array containing w, h, fx, fy, x0, y0.
#       target_size: target size, a tuple of (height, width).

#     Returns:
#       A 1-d tensor containing the adjusted camera intrinsics.
#     """
#     with tf.name_scope("resize_intrinsics"):
#         w, h, fx, fy, x0, y0 = tf.unstack(intrinsics)

#         def float_div(a, b):
#             return tf.cast(a, tf.float32) / tf.cast(b, tf.float32)

#         xfactor = float_div(target_size[1], w)
#         yfactor = float_div(target_size[0], h)
#         fx *= xfactor
#         fy *= yfactor
#         x0 *= xfactor
#         y0 *= yfactor
#         w = target_size[1]
#         h = target_size[0]

#         return tf.stack((w, h, fx, fy, x0, y0))


# def flip_egomotion(egomotion):
#     """Transforms camera egomotion when the image is flipped horizontally.

#        The intrinsics matrix is ((fx, 0, x0), (0, fy, y0), (0, 0, 1)).
#        Given a pixel (px, py, 1), the x coordinate is x = px * fx + 1.
#        Now what if we flip the image along x? This maps px to w - 1 - px,
#        where w is the image width. Therefore for the flipped image,
#        we have x' = (w - px - 1) * fx + 1.
#        Therefore x' = -x + (w - 1 - 2 * x0) / fx,
#        if x0 = ((w - 1) / 2), that is, if the optical center is exactly at
#        the center of the image, then indeed x' = -x, so we can just flip x.
#        Otherwise there is a correction which is inrinsics-dependent:
#        we'd have to add a small translation component to flip_mat, but we ignore
#        this small correction for now.
#     Args:
#       egomotion: a 2-d transformation matrix.

#     Returns:
#       A 2-d transformation matrix.
#     """
#     with tf.name_scope("flip_egomotion"):
#         flip_mat = tf.constant([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=tf.float32)
#         egomotion = tf.matmul(tf.matmul(flip_mat, egomotion), flip_mat)
#         return egomotion


# def flip_intrinsics(intrinsics):
#     """Flips camera intrinsics when the image is flipped horizontally.

#     Args:
#       intrinsics: 1-d array containing w, h, fx, fy, x0, y0.

#     Returns:
#       A 1-d tensor containing the adjusted camera intrinsics.
#     """
#     with tf.name_scope("flip_intrinsics"):
#         w, h, fx, fy, x0, y0 = tf.unstack(intrinsics)
#         x0 = w - x0
#         y0 = h - y0

#         return tf.stack((w, h, fx, fy, x0, y0))
