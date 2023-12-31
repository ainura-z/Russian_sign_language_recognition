import cv2
import numpy as np
import torch


def get_frames(video_path, all_frames=True):
    """
    Function for retrieving frames from video

    Inputs:
        video_path - path to video file
        all_frames - whether to return all frames
          from video or only the first one

    Returns:
        imgs - list of video frames
    """
    cam = cv2.VideoCapture(f"{video_path}")
    frameno = 0
    imgs = []

    while True:
        ret, frame = cam.read()
        if ret:
            # convert color to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs.append(frame)
            frameno += 1
        else:
            break
        # break if only the first frame is retrieved
        if not all_frames:
            break
    cam.release()
    return imgs


def get_absolute_coord(xy_coord, H, W, clip=True):
    """
    Transform from relative coords to absolute coords

    Inputs:
        xy_coord - array of coords
        H, W - height and width of frame
        clip - whether to clip coords values between 0 and 1
    Output:
        xy_coord - absolute coords
    """
    if clip:
        xy_coord[..., 0] = np.clip(xy_coord[..., 0], a_min=0, a_max=1) * (W - 1)
        xy_coord[..., 1] = np.clip(xy_coord[..., 1], a_min=0, a_max=1) * (H - 1)
    else:
        xy_coord[..., 0] = xy_coord[..., 0] * W
        xy_coord[..., 1] = xy_coord[..., 1] * H
    xy_coord = xy_coord.astype("int")
    return xy_coord


def get_relative_coord(xy_coord, H, W):
    """
    Transform from absolute coords to relative coords

    Inputs:
        xy_coord - array of coords
        H, W - height and width of frame
    Output:
        xy_coord - relative coords
    """
    xy_coord[..., 0] = np.clip(xy_coord[..., 0] / W, a_min=0.0, a_max=1.0)
    xy_coord[..., 1] = np.clip(xy_coord[..., 1] / H, a_min=0.0, a_max=1.0)
    return xy_coord


def sample_video(frames, seq_len):
    """
    Sample video frames

    Inputs:
        frames - list of frames
        seq_len - number of frames to sample

    Returns:
        sampled_frames - list of seq_len num of sampled frames
    """
    idxs = np.round(np.linspace(0, len(frames) - 1, seq_len)).astype("int")
    indices = np.unique(idxs)
    sampled_frames = [frames[ind] for ind in indices]
    return sampled_frames


def get_maxmin_limits(points, impute_val=-10):
    """
    Min and max coordinates of nonnegative values

    Inputs:
        points - coordinates
        impute_val - value to be considered as threshold for negative values
    Output:
        minx, maxx, miny, maxy - limits
    """
    minx = points[:, :, 0][points[:, :, 0] > impute_val].min()
    maxx = points[:, :, 0][points[:, :, 0] > impute_val].max()
    miny = points[:, :, 1][points[:, :, 1] > impute_val].min()
    maxy = points[:, :, 1][points[:, :, 1] > impute_val].max()
    return minx, maxx, miny, maxy


def shift_transform(coords, left_limit, right_limit, top_limit, bottom_limit, H, W):
    """
    Shift transform

    Inputs:
        coords - array of keypoints coordinates
        left_limit, right_limit, top_limit, bottom_limit - num of pixels to each side of frame
        H, W - height and width of frame
    Output:
        coords - array of transformed coordinates
        shift_x, shift_y - amount of shifting for each side
    """
    # random shift values
    power = np.random.uniform(0.05, 0.3)
    left_limit, right_limit = int((1 - power) * left_limit), min(
        int((1 + power) * right_limit), W - 1
    )
    top_limit, bottom_limit = int((1 - power) * top_limit), min(
        int((1 + power) * bottom_limit), H - 1
    )

    shift_x = np.random.randint(-left_limit, W - right_limit, size=1)[0]
    shift_y = np.random.randint(-top_limit, H - bottom_limit, size=1)[0]

    # transform
    coords[..., 0] = coords[..., 0] + shift_x
    coords[..., 1] = coords[..., 1] + shift_y
    return coords, shift_x, shift_y


def crop_transform(coords, left_limit, right_limit, top_limit, bottom_limit, H, W):
    """
    Crop transform

    Inputs:
        coords - array of keypoints coordinates
        left_limit, right_limit, top_limit, bottom_limit - num of pixels to each side of frame
        H, W - height and width of frame
    Output:
        coords - array of transformed coordinates
        crops - amount of cropping for each side
        new_H, new_W - new frame size
    """
    # random cropping value
    power = np.random.uniform(0.1, 0.5)
    left_limit, right_limit = int((1 - power) * left_limit), min(
        int((1 + power) * right_limit), W - 1
    )
    top_limit, bottom_limit = int((1 - power) * top_limit), min(
        int((1 + power) * bottom_limit), H - 1
    )

    if left_limit > 0:
        crop_left = np.random.randint(0, left_limit, size=1)[0]
    else:
        crop_left = 0
    crop_right = np.random.randint(0, W - right_limit, size=1)[0]
    if top_limit > 0:
        crop_top = np.random.randint(0, top_limit, size=1)[0]
    else:
        crop_top = 0
    crop_bottom = np.random.randint(0, H - bottom_limit, size=1)[0]

    # transform coordinates after cropping
    coords[..., 0] = coords[..., 0] - crop_left
    coords[..., 1] = coords[..., 1] - crop_top

    crops = (crop_left, crop_right, crop_top, crop_bottom)
    new_H = H - crop_bottom - crop_top
    new_W = W - crop_left - crop_right
    return coords, crops, new_H, new_W


def pad_ifneeded(coords, H, W, min_size=700):
    """
    Padding transform

    Inputs:
        coords - array of keypoints coordinates
        H, W - height and width of frame
        min_size - length of min side of resulting frame
    Output:
        coords - array of transformed coordinates
        new_H, new_W - new frame size
    """
    new_H, new_W = H, W

    # if the min side is less than min_size
    if min(H, W) < min_size:
        padd = min_size - min(H, W)
        padd = padd + padd % 2

        # padd horizontally
        if W < H:
            coords[..., 0] = coords[..., 0] + (padd // 2)
            new_W = W + padd
        # padd vertically
        elif H < W:
            coords[..., 1] = coords[..., 1] + (padd // 2)
            new_H = H + padd
    return coords, new_H, new_W


class LandmarksDataset(torch.utils.data.Dataset):
    """Dataset of landmarks"""

    def __init__(self, landmarks, file_names, annot, label_enc, seq_len, mode="train"):
        """
        Inputs:
            landmarks - dictionary of type filename:landmarks
            file_names - list of file names of videos in dataset
            annot - annotation table
            label_enc - mapping of labels to integers
            seq_len - length of frames sequence for each video
            mode - mode of dataset: train or test
        """
        self.video_paths = file_names
        # sample videos to be of length seq_len (if len(video) > seq_len)
        self.landmarks = {
            file: sample_video(landmarks[file], seq_len) for file in self.video_paths
        }
        self.seq_len = seq_len
        self.mode = mode

        # labels and frame shape data
        self.labels = []
        self.sizes = []
        for filename in self.video_paths:
            ann = annot[annot["attachment_id"] == filename].iloc[0]
            self.sizes.append((ann["height"], ann["width"]))
            self.labels.append(label_enc[ann["text"]])

    def __getitem__(self, idx):
        # current data of idx video
        filename = self.video_paths[idx]
        landmarks = self.landmarks[filename]
        label = self.labels[idx]
        H, W = self.sizes[idx]

        # order of landmarks
        hands_order = ["right hand"]

        points = []
        for i, frame in enumerate(landmarks):
            curr_landm = []
            # for each hand
            for key in hands_order:
                if key in frame:
                    curr_frame = frame[key]
                    # frame keypoints
                    curr_coord = [
                        [point["x"], point["y"], point["z"]] for point in curr_frame
                    ]
                    curr_coord = np.array(curr_coord)
                    curr_landm.extend(curr_coord)
                else:
                    if self.mode == "train":
                        # replace missing landmarks with large negative value for processing later
                        curr_landm.extend(np.full((21, 3), -100000).astype(np.float32))
                    else:
                        # replace missing landmarks with 0
                        curr_landm.extend(np.zeros((21, 3)).astype(np.float32))
            points.append(curr_landm)
        # num_frames x seq_len x 3
        points = np.array(points)

        if self.mode == "train":
            # transform prob
            p_shift = 0.5
            p_crop = 0.5
            p_pad = 0.5

            # absolute coords for transform
            abs_points = get_absolute_coord(points.copy(), H, W, clip=True)
            coords = abs_points.copy()
            # maxmin coords
            minx, maxx, miny, maxy = get_maxmin_limits(abs_points)
            new_H, new_W = H, W

            if np.random.rand() > p_shift:
                coords, shx, shy = shift_transform(
                    coords.copy(), minx, maxx, miny, maxy, H, W
                )
                minx, maxx, miny, maxy = minx + shx, maxx + shx, miny + shy, maxy + shy
            if np.random.rand() > p_crop:
                coords, crops, new_H, new_W = crop_transform(
                    coords.copy(), minx, maxx, miny, maxy, H, W
                )
                minx, maxx, miny, maxy = get_maxmin_limits(coords)
            if np.random.rand() > p_pad:
                coords, new_H, new_W = pad_ifneeded(coords.copy(), new_H, new_W)
            # translate to relativa coords
            final_coords = get_relative_coord(coords.astype("float"), new_H, new_W)
            # impute z coord values with 0 for missing landmarks
            final_coords[..., 2] = np.where(points[..., 2] > -5, points[..., 2], 0.0)
            final_coords = final_coords.reshape(len(landmarks), -1)
        elif self.mode == "test":
            final_coords = points.reshape(len(landmarks), -1)
        # padding of sequences to be of length seq_len
        padding = self.seq_len - len(landmarks)
        # add padding
        if padding > 0:
            # padd_points = np.zeros((padding, final_coords.shape[-1]))
            # padd with the last frame keypoints
            padd_points = np.repeat(final_coords[-1][None, :], repeats=padding, axis=0)
            final_coords = np.concatenate((final_coords, padd_points))
        return torch.from_numpy(final_coords).float(), torch.tensor(label).long()

    def __len__(self):
        return len(self.video_paths)