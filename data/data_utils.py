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
        if padding > 0:
            padd_points = np.zeros((padding, points.shape[-1]))
            points = np.concatenate(points, padd_points)
        
        return torch.tensor(points), torch.tensor(label)

    def __len__(self):
        return len(self.video_paths)