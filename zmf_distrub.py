import os
import cv2
import dlib
from imutils import face_utils
import numpy as np
import torch
from torchvision import transforms
from model import gaze_network
import glob,tqdm

from head_pose import HeadPoseEstimator

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True): # (6,1,2), (6,1,3), camera_matrix(f,u0,v0)
    # print(face_model.shape) # (6, 1, 3)
    # print(landmarks.shape) # (6, 1, 2)
    

    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    ## further optimize
    if iterate: ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec

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

def normalizeData_face(img, face_model, landmarks, hr, ht, cam): # (754, 1024, 3), (6,3), (6,1,2), (3,1), (3,1), (3,3)

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

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

    img_warped = cv2.warpPerspective(img, W, roiSize)  # warp the input image

    # head pose after normalization
    hR_norm = np.dot(R, hR)  # head pose rotation matrix in normalized space
    hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    # normalize the facial landmarks
    num_point = landmarks.shape[0]
    landmarks_warped = cv2.perspectiveTransform(landmarks, W)
    landmarks_warped = landmarks_warped.reshape(num_point, 2)

    return img_warped, landmarks_warped


# image_names = glob.glob('../../data/dataset/Face/retinaface/widerface-all/train/widerface-faces/*/line_1042_in_labeltxt.jpg')
image_names = glob.glob('../../data/dataset/Face/retinaface/widerface-all/test/distrub/input/*/*.jpg')
image_names.sort()

for imnum, image_name in enumerate(tqdm.tqdm(image_names)):
    # image_name = './example/input/cam00.JPG'
    # print('load input face image: ', image_name)
    image = cv2.imread(image_name)
    # print('image shape: ' + str(image.shape))

    rec = dlib.rectangle(0,0,image.shape[0],image.shape[1])

    # FACE LANDMARK

    predictor = dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks.dat')
    shape = predictor(image, rec) # predict for each face
    shape = face_utils.shape_to_np(shape) # (68,2)
    landmarks = []
    for (x, y) in shape: landmarks.append((x, y))
    landmarks = np.asarray(landmarks) # (68, 2)

    # LOAD CAMERA

    # load camera information
    cam_file_name = './example/input/cam00.xml'  # this is camera calibration information file obtained with OpenCV
    if not os.path.isfile(cam_file_name): exit('no camera calibration file is found.')
    fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
    camera_distortion = fs.getNode('Distortion_Coefficients').mat()
    camera_matrix[0][0] = 224. # 960.
    camera_matrix[1][1] = 224. # 960. camera_to_be_normed
    camera_matrix[0][2] = 112.
    camera_matrix[1][2] = 112.


    # HEAD POSE

    # print('estimate head pose')
    # load face model
    face_model_load = np.loadtxt('face_model.txt') # (50,3)  # Generic face model with 3D facial landmarks
    landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
    face_model = face_model_load[landmark_use, :] # (6,3)
    # estimate the head pose,
    ## the complex way to get head pose information, eos library is required,  probably more accurrated
    landmarks = landmarks.reshape(-1, 2)
    head_pose_estimator = HeadPoseEstimator()
    hr, ht, o_l, o_r, _ = head_pose_estimator(image, landmarks, camera_matrix)
    # ## the easy way to get head pose information, fast and simple
    # facePts = face_model.reshape(6, 1, 3) # (6,1,3)
    landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :] # (6,2)
    landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
    landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
    # hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion) # (3,1), (3,1)

    # DATA NORMALIZATION

    # data normalization method
    # print('data normalization, i.e. crop the face image')
    img_normalized, landmarks_normalized = normalizeData_face(image, face_model, landmarks_sub, hr, ht, camera_matrix)
    # print(img_normalized.shape) # (224, 224, 3)
    # print(landmarks_normalized.shape) # (6,2)
    # cv2.imwrite('./example/output/nor_face.jpg', img_normalized)
    img_normalized, landmarks_normalized = image, landmarks_sub





    # COMPUTE GAZE

    # print('load gaze estimator')
    model = gaze_network()
    model.cuda() # comment this line out if you are not using GPU
    pre_trained_model_path = '/home/zhangmf/Documents/data/checkpoints/Gaze/ETH-XGaze/epoch_24_ckpt.pth.tar'
    if not os.path.isfile(pre_trained_model_path):
        print('the pre-trained gaze estimation model does not exist.')
        exit()
    # else:
    #     print('load the pre-trained model: ', pre_trained_model_path)
    ckpt = torch.load(pre_trained_model_path)
    model.load_state_dict(ckpt['model_state'], strict=True)  # load the pre-trained model
    model.eval()  # change it to the evaluation mode
    input_var = img_normalized[:, :, [2, 1, 0]]  # from BGR to RGB
    input_var = trans(input_var)
    input_var = torch.autograd.Variable(input_var.float().cuda())
    input_var = input_var.view(1, input_var.size(0), input_var.size(1), input_var.size(2))  # the input must be 4-dimension
    pred_gaze = model(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
    pred_gaze = pred_gaze[0] # here we assume there is only one face inside the image, then the first one is the prediction
    pred_gaze_np = pred_gaze.cpu().data.numpy()  # convert the pytorch tensor to numpy array





    # DRAW GAZE

    # print('prepare the output')
    # draw the facial landmarks
    # landmarks_normalized = landmarks_normalized.astype(int) # landmarks after data normalization
    # for (x, y) in landmarks_normalized:
    #     cv2.circle(img_normalized, (x, y), 5, (0, 255, 0), -1)
    # print(pred_gaze_np,pred_gaze_np.shape, hr.shape, ht.shape)
    face_patch_gaze = draw_gaze(img_normalized, pred_gaze_np)  # draw gaze direction on the normalized face image
    output_path = image_name.replace('/input/','/output-eth/')
    # print(hr)
    # print(ht)
    # print(pred_gaze_np)
    # print(output_path)
    new_outpath = '%s__%f_%f_%f_%f_%f_%f_%f_%f.jpg'%(output_path[:-4],pred_gaze_np[0],pred_gaze_np[1],hr[0,0],hr[1,0],hr[2,0],ht[0,0],ht[1,0],ht[2,0])
    # print(new_outpath)
    if not os.path.exists(os.path.dirname(new_outpath)):
        os.makedirs(os.path.dirname(new_outpath))
    # print('save output image to: ', new_outpath)
    cv2.imwrite(new_outpath, face_patch_gaze)
    # print('over')
    # break