import numpy as np
import cv2,os,glob,tqdm

def pitchyaw_to_vector(pitchyaws):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.
    Args:
        pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.
    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
    """
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((3,n))
    out[0,:] = np.multiply(cos[:, 0], sin[:, 1])
    out[1,:] = sin[:, 0]
    out[2,:] = np.multiply(cos[:, 0], cos[:, 1])
    return out

def vector_to_pitchyaw(vectors):
    r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.
    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.
    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out

def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w / 2.0
    pos = (int(h / 2.0), int(w / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out

def unnormalizeData_face(normed_gaze, hr, ht):
    face_model_load = np.loadtxt('face_model.txt')
    landmark_use = [20, 23, 26, 29, 15, 19]
    face_model = face_model_load[landmark_use, :]

    cam = np.array([
        [224.,0.,112.],
        [0.,224.,112.],
        [0.,0.,1.],
    ])
    # cam = np.array([
    #     [1.32007218e+04,0.00000000e+00,1.12000000e+02],
    #     [0.00000000e+00,1.31925066e+04,1.12000000e+02],
    #     [0.00000000e+00,0.00000000e+00,1.00000000e+00],
    # ])


    ## normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 600  # normalized distance between eye and camera
    roiSize = (224, 224)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    # get the face center
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

    ## ---------- normalize image ----------
    distance = np.linalg.norm(face_center)  # actual distance between eye and original camera

    z_scale = distance_norm / distance
    cam_norm = np.array([  # camera intrinsic parameters of the virtual camera
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([  # scaling matrix
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    RI = np.linalg.inv(R)
    normed_gaze = np.array([[normed_gaze[0],normed_gaze[1]]])
    three_gaze = pitchyaw_to_vector(normed_gaze)
    out_gaze3 = np.dot(RI, three_gaze)
    out_gaze3 = np.array([[out_gaze3[0][0],out_gaze3[1][0],out_gaze3[2][0]]])
    out_gaze = vector_to_pitchyaw(out_gaze3)
    # print(out_gaze.shape)
    out_gaze = (out_gaze[0][0],out_gaze[0][1])



    return out_gaze

vis_imgs = glob.glob('../../data/dataset/Face/retinaface/widerface-all/train/widerface-faces-gazevis-05/*/*.jpg')
out_folder_addr = '/widerface-faces-gazevis-unnorm-05/'
orig_label_addr = '../../data/dataset/Face/retinaface/widerface-all/train/label.txt'
create_label_addr = '../../data/dataset/Face/retinaface/widerface-all/train/label_append_gaze_05.txt'

new_labels = []
for img_addr in tqdm.tqdm(vis_imgs):
    # print(img_addr)
    labels = os.path.basename(img_addr).split('_')
    line_num = int(labels[1])
    gaze0 = float(labels[4])
    gaze1 = float(labels[5])
    hr0 = float(labels[6])
    hr1 = float(labels[7])
    hr2 = float(labels[8])
    ht0 = float(labels[9])
    ht1 = float(labels[10])
    ht2 = float(labels[11][:-4])
    normed_gaze = [gaze0,gaze1]
    hr = [[hr0],[hr1],[hr2]]
    ht = [[ht0],[ht1],[ht2]]
    hr = np.array(hr)
    ht = np.array(ht)
    out_gaze = unnormalizeData_face(normed_gaze,hr,ht)


    orig_im = cv2.imread(img_addr)
    imout = draw_gaze(orig_im,out_gaze,color=(255,0,0))

    save_addr = img_addr.replace('/widerface-faces-gazevis-05/',out_folder_addr)
    if not os.path.exists(os.path.dirname(save_addr)):
        os.makedirs(os.path.dirname(save_addr))
    img_basename = 'line_%d_%f_%f.jpg'%(line_num,out_gaze[0],out_gaze[1])
    save_addr = '%s/%s'%(os.path.dirname(save_addr),img_basename)
    # cv2.imwrite(save_addr,imout)

    new_label = [line_num,out_gaze]
    new_labels.append(new_label)

    

    # break

fr = open(orig_label_addr,'r')
lines = fr.readlines()
for new_label in new_labels:
    linenum = new_label[0]
    gaze = new_label[1]
    newline = '%s %f %f\n'%(lines[linenum][:-1],gaze[0],gaze[1])
    lines[linenum] = newline
    # print(lines[linenum])

append_num = 0
all_num = 0
fw = open(create_label_addr,'w')
for line in lines:
    fw.write(line)
    if len(line.split(' ')) == 22:
        append_num += 1
        all_num += 1
    elif line[0] == '#':
        continue
    else:
        all_num += 1
        assert(len(line.split(' ')) == 20)
print(append_num,all_num)
    
