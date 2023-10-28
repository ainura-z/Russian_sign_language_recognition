import cv2
import numpy as np
import torch


def mediapipe_detection(image, model):
    """
     Function for retrieving keypoints from frame

    Inputs:
        image - particular frame
        model - Holistic model

    Returns:
        image - returning initial image 
            for further drawing landmarks on it
        results - all detected keypoints
    """
    
    # COLOR CONVERSION BGR 2 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    # Image is no longer writeable
    image.flags.writeable = False         
    # Make prediction
    results = model.process(image) 
    # Image is now writeable
    image.flags.writeable = True        
    # COLOR COVERSION RGB 2 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    
    return image, results
    

def extract_keypoints(results):
     """
     Function for extracting keypoints from results

    Inputs:
        results - all detected keypoints after model

    Returns:
        concatenated vector of right hand and left hand keypoints
    """
    
    # extracting left hand keypoints
    lh = torch.tensor([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else torch.tensor([0]*21*3)
    # extracting right hand keypoints
    rh = torch.tensor([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else torch.tensor([0]*21*3)
    
    return torch.concatenate([lh, rh])
    
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
    cam = cv2.VideoCapture(f'{video_path}')
    frameno = 0
    imgs = []
    
    while(True):
        ret,frame = cam.read()
        if ret:
            # convert color to RGB
            frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs.append(frame)
            frameno += 1
        else:
            break
        # break if only the first frame is retrieved
        if not all_frames:
            break
    
    cam.release()
    return imgs


def sample_video(frames, seq_len):
    """
    Sample video frames

    Inputs:
        frames - list of frames
        seq_len - number of frames to sample

    Returns:
        sampled_frames - list of seq_len num of sampled frames
    """
    idxs = np.round(np.linspace(0, len(frames)-1, seq_len)).astype('int')
    indices = np.unique(idxs)
    sampled_frames = [frames[ind] for ind in indices]
    return sampled_frames


class LandmarksDataset(torch.utils.data.Dataset):
    """Dataset of landmarks"""
    def __init__(self, landmarks, file_names, annot, label_enc, seq_len):
        """
        Inputs:
            landmarks - dictionary of type filename:landmarks
            file_names - list of file names of videos in dataset
            annot - annotation table 
            label_enc - mapping of labels to integers
            seq_len - length of frames sequence for each video
        """
        self.video_paths = file_names
        # sample videos to be of length seq_len (if len(video) > seq_len)
        self.landmarks = {file:sample_video(landmarks[file], seq_len) for file in self.video_paths}
        self.seq_len = seq_len

        # label encoding of labels
        self.labels = []
        for filename in self.video_paths:
            ann = annot[annot['attachment_id']==filename].iloc[0]
            self.labels.append(label_enc[ann['text']])
        
    def __getitem__(self, idx):
        filename = self.video_paths[idx]
        landmarks = self.landmarks[filename]
        label = self.labels[idx]

        # from dict of landmarks to array of x, y, z points
        points = []
        for frame in landmarks:
            curr_landm = []
            for key in ['hand 1', 'hand 2']:
                if key in frame:
                    curr_frame = frame[key]
                    curr_landm.extend([[point['x'], point['y'], point['z']] for point in curr_frame])
                else:
                    # replace missing landmarks with 0
                    curr_landm.extend([[0., 0., 0.] for i in range(21)])
            points.append(curr_landm)
        points = np.array(points).reshape(len(landmarks), -1)

        # padding of sequences to be of length seq_len
        padding = (len(landmarks) - self.seq_len)
        # add padding
        if padding < 0:
            padd_points = np.zeros((-padding, points.shape[-1]))
            points = np.concatenate((points, padd_points))
        
        return torch.tensor(points), torch.tensor(label)

    def __len__(self):
        return len(self.video_paths)
    

class LandmarksTestDataset(torch.utils.data.Dataset):
    """Dataset of landmarks"""
    def __init__(self, landmarks, file_names, annot, label_enc, seq_len):
        self.video_paths = file_names
        # sample videos to be of length seq_len (if len(video) > seq_len)
        self.landmarks = {file:sample_video(landmarks[file], seq_len) for file in self.video_paths}
        self.seq_len = seq_len

        # label encoding of labels
        self.labels = []
        self.sizes = []
        for filename in self.video_paths:
            ann = annot[annot['attachment_id']==filename].iloc[0]
            self.sizes.append((ann['height'], ann['width']))
            self.labels.append(label_enc[ann['text']])
        
    def __getitem__(self, idx):
        filename = self.video_paths[idx]
        landmarks = self.landmarks[filename]
        label = self.labels[idx]
        # size of frame
        H, W = self.sizes[idx]
        hands_order = ['hand 1', 'hand 2']
        points = []
        for i, frame in enumerate(landmarks):
            curr_landm = []
            # for each hand
            for key in hands_order:
                if key in frame:
                    curr_frame = frame[key]
                    # frame keypoints
                    curr_coord = [[point['x'], point['y'], point['z']] for point in curr_frame]
                    curr_coord = np.array(curr_coord)
                    curr_landm.extend(curr_coord)
                else:
                    # replace missing landmarks with 0
                    curr_landm.extend(np.zeros((21, 3)).astype(np.float32)) 
            points.append(curr_landm)
        points = np.array(points).reshape(len(landmarks), -1)

        # padding of sequences to be of length seq_len
        padding = (self.seq_len - len(landmarks))
        # add padding
        if padding > 0:
            padd_points = np.zeros((padding, points.shape[-1]))
            points = np.concatenate((points, padd_points))
        
        return torch.from_numpy(points).float(), torch.tensor(label).long()

    def __len__(self):
        return len(self.video_paths)
    
def get_maxmin_limits(points, impute_val=-10):
    minx = points[:, :, 0][points[:, :, 0]>impute_val].min()
    maxx = points[:, :, 0][points[:, :, 0]>impute_val].max()
    miny = points[:, :, 1][points[:, :, 1]>impute_val].min()
    maxy = points[:, :, 1][points[:, :, 1]>impute_val].max()
    return minx, maxx, miny, maxy

def shift_transform(coords, left_limit, right_limit, top_limit, bottom_limit, H, W):
    power = np.random.uniform(0.05, 0.3)
    left_limit, right_limit = int((1-power)*left_limit), min(int((1+power)*right_limit), W-1)
    top_limit, bottom_limit = int((1-power)*top_limit), min(int((1+power)*bottom_limit), H-1)
    
    shift_x = np.random.randint(-left_limit, W - right_limit, size=1)[0]
    shift_y = np.random.randint(-top_limit, H - bottom_limit, size=1)[0]
    
    coords[..., 0] = coords[..., 0] + shift_x
    coords[..., 1] = coords[..., 1] + shift_y
    return coords, shift_x, shift_y

def crop_transform(coords, left_limit, right_limit, top_limit, bottom_limit, H, W):
    power = np.random.uniform(0.1, 0.5)
    left_limit, right_limit = int((1-power)*left_limit), min(int((1+power)*right_limit), W-1)
    top_limit, bottom_limit = int((1-power)*top_limit), min(int((1+power)*bottom_limit), H-1)
    
    if left_limit > 0:
        crop_left = np.random.randint(0, left_limit, size=1)[0]
    else:
        crop_left = 0
    crop_right = np.random.randint(0, W-right_limit, size=1)[0]
    if top_limit > 0:
        crop_top = np.random.randint(0, top_limit, size=1)[0]
    else:
        crop_top = 0
    crop_bottom = np.random.randint(0, H-bottom_limit, size=1)[0]

    coords[..., 0] = coords[..., 0] - crop_left
    coords[..., 1] = coords[..., 1] - crop_top
    
    crops = (crop_left, crop_right, crop_top, crop_bottom)
    new_H = H-crop_bottom-crop_top
    new_W = W-crop_left-crop_right
    return coords, crops, new_H, new_W

def pad_ifneeded(coords, H, W, min_size=700):
    new_H, new_W = H, W
    
    if min(H, W) < min_size:
        padd = min_size - min(H, W)
        padd = padd + padd % 2
        
        if W < H:
            coords[..., 0] = coords[..., 0] + (padd // 2)
            new_W = W + padd
        elif H < W:
            coords[..., 1] = coords[..., 1] + (padd // 2)
            new_H = H + padd
    return coords, new_H, new_W

class LandmarksDataset2(torch.utils.data.Dataset):
    """Dataset of landmarks"""
    def __init__(self, landmarks, file_names, annot, label_enc, seq_len):
        self.video_paths = file_names
        # sample videos to be of length seq_len (if len(video) > seq_len)
        self.landmarks = {file:sample_video(landmarks[file], seq_len) for file in self.video_paths}
        self.seq_len = seq_len

        # label encoding of labels
        self.labels = []
        self.sizes = []
        for filename in self.video_paths:
            ann = annot[annot['attachment_id']==filename].iloc[0]
            self.sizes.append((ann['height'], ann['width']))
            self.labels.append(label_enc[ann['text']])
        
    def __getitem__(self, idx):
        filename = self.video_paths[idx]
        landmarks = self.landmarks[filename]
        label = self.labels[idx]
        # size of frame
        H, W = self.sizes[idx]
        hands_order = ['hand 1', 'hand 2']
        
        points = []
        for i, frame in enumerate(landmarks):
            curr_landm = []
            # for each hand
            for key in hands_order:
                if key in frame:
                    curr_frame = frame[key]
                    # frame keypoints
                    curr_coord = [[point['x'], point['y'], point['z']] for point in curr_frame]
                    curr_coord = np.array(curr_coord)
                    curr_landm.extend(curr_coord)
                else:
                    # replace missing landmarks with 0
                    curr_landm.extend(np.full((21, 3), -100000).astype(np.float32)) 
            points.append(curr_landm)
        # num_frames x seq_len x 3
        points = np.array(points)#.reshape(len(landmarks), -1)

        p_shift = 0.5
        p_crop = 0.5
        p_pad = 0.5

        abs_points = get_absolute_coord(points.copy(), H, W, clip=False)
        minx, maxx, miny, maxy = get_maxmin_limits(abs_points)
        coords = abs_points.copy()
        new_H, new_W = H, W

        if np.random.rand() > p_shift:
            coords, shx, shy = shift_transform(coords.copy(), minx, maxx, miny, maxy, H, W)
            minx, maxx, miny, maxy = minx+shx, maxx+shx, miny+shy, maxy+shy 
        if np.random.rand() > p_crop:
            coords, crops, new_H, new_W = crop_transform(coords.copy(), minx, maxx, miny, maxy, H, W)
            minx, maxx, miny, maxy = get_maxmin_limits(coords)
        if np.random.rand() > p_pad:
            coords, new_H, new_W = pad_ifneeded(coords.copy(), new_H, new_W)

        final_coords = get_relative_coord(coords.astype('float'), new_H, new_W)
        final_coords[..., 2] = np.where(points[..., 2] > -5, points[..., 2], 0.0)
        final_coords = final_coords.reshape(len(landmarks), -1)
        
        # padding of sequences to be of length seq_len
        padding = (self.seq_len - len(landmarks))
        # add padding
        if padding > 0:
            padd_points = np.zeros((padding, final_coords.shape[-1]))
            final_coords = np.concatenate((final_coords, padd_points))
        
        return torch.from_numpy(final_coords).float(), torch.tensor(label).long()

    def __len__(self):
        return len(self.video_paths)