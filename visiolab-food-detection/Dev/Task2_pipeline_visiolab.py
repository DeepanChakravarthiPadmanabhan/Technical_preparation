import numpy as np
import cv2 as cv
import tensorflow as tf
import math
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

label_map = {1:'0',2:'1',3:'2',4:'3',5:'4',6:'5',7:'6'}

VISUALIZE = False

def show_frame(image, caption='Frame to see'):
    cv.imshow(caption, image)
    cv.waitKey(0)

def save_img(image, filename):
    cv.imwrite(filename, image)

def capture_frames(video_data):
    # To capture the stable frame, where the motion of objects has been stopped.
    # Implements frame differencing by consecutive frame subtraction.
    captured_frames = list()
    img_dict = dict()
    img_list = list()
    img_idx = list()
    frame_diff_list = list()

    cap = cv.VideoCapture(video_data)
    n_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv.CAP_PROP_FPS)

    count = 0
    while cap.isOpened():
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(fps / 2.0) == 0):
            b, g, r = cv.split(frame)
            img = cv.merge((b, g, r))
            gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img_dict[count] = (gray_image)
        count = count+1
    cap.release()
    if TEST:
        print("Frame: {}\nFrames per second: {}".format(n_frames, fps))

    k = img_dict.items()
    for n,i in enumerate(k):
        img_list.append(i[1])
        img_idx.append(i[0])

    # calculating the frame difference between consecutive frames
    for i in range(1, len(img_list)):
        frame_diff = cv.absdiff(img_list[i], img_list[i - 1])
        frame_diff = cv.GaussianBlur(frame_diff, (3, 3), 0)
        frame_diff = cv.threshold(frame_diff, 25, 255, cv.THRESH_BINARY)[1]
        frame_diff_list.append(cv.countNonZero(frame_diff))

    plt.plot(frame_diff_list)
    plt.xlabel('Frame pairs')
    plt.ylabel('Non-zero pixel count')
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1)
    plt.savefig('frame_diff_4.pdf', color="#6c3376", linewidth=3, dpi=600)
    plt.show()
    min_idx = np.argmin(frame_diff_list)
    if len(frame_diff_list) > 6 and min_idx > int(len(frame_diff_list) * 0.8):
        min_idx = np.argmin(frame_diff_list[:-2])

    # storing the frame with the least consecutive frame difference for further processing
    captured_frames.append(img_list[min_idx + 1])

    return captured_frames[0], img_idx[min_idx+1]

def run_inference(frozen_graph_path, stable_frame):

    # Read the graph.
    with tf.io.gfile.GFile(frozen_graph_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.compat.v1.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # Read and preprocess an image.
        img = stable_frame
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        class_label = list() # list of labelIDs
        bbox_list = list() # list of tuples - x,y, right, bottom
        score_list = list()

        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]

            if score > 0.3:
            # Have taken the best detection confidence because
            # no decoy class and definitely one class of trained classes per image is present in the test set.
                score_list.append(score)
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                tuple_data = (x, y, right, bottom)
                bbox_list.append(tuple_data)
                classId = classId-1
                class_label.append(classId)

                if VISUALIZE:
                    # For visualization of each frames bounding box
                    cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
            else:
                score_list.append(score)
        if class_label:
            max_id = np.argmax(score_list)
            class_selected = class_label[max_id]
            bbox_selected = bbox_list[max_id]
            confidence = score_list[max_id]
        else:
            max_id = np.argmax(score_list)
            confidence = 1 - score_list[max_id]
            return None, None, confidence

    return class_selected, bbox_selected, confidence

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visiolab- Kaggle Challenge')
    parser.add_argument("inp", type=str, help="input video path")
    parser.add_argument("out", type=str,nargs='?', default='Result/', help="Path to save the results.csv")

    args = parser.parse_args()

    data_path = args.inp
    output_path = args.out
    output_file_name = os.path.splitext(os.path.basename(data_path))[0]
    frozen_graph_path = './frozen_inference_graph.pb'
    best_class, best_bb, image_name_list, confidence_list, frame_list, best_class_ = \
        list(), list(), list(), list(), list(), list()

    video_data = data_path

    cap = cv.VideoCapture(video_data)
    n_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv.CAP_PROP_FPS)

    print("Total frames in the video:{}\nFrame rate:{}".format(n_frames, fps))

    count = 0
    while cap.isOpened():
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId!=0 and frameId % 5 == 0):
            # List of the images of every 5th frame in the video sequence
            # print('Frame attaching', frameId)
            image_name_list.append(frame)
            frame_list.append(frameId)
        count = count + 1
    cap.release()

    print('Number of frames captured:', len(image_name_list))

    for i,n in zip(image_name_list, frame_list):
        # plt.imshow(i)
        # plt.show()
        class_inferred, bb_inferred, confidence = run_inference(frozen_graph_path, i)
        if class_inferred:
            best_class.append(class_inferred)
            best_bb.append(bb_inferred)
            confidence_list.append(confidence)
        else:
            best_class.append('No object frame')
            best_bb.append(('NA','NA','NA','NA'))
            confidence_list.append(confidence)

    x = [i[0] for i in best_bb]
    y = [i[1] for i in best_bb]
    top_right = [i[2] for i in best_bb]
    bottom_left = [i[3] for i in best_bb]
    for i in best_class:
        if i!='No object frame':
            best_class_.append(label_map[i])
        else:
            best_class_.append('No object frame')
    df = {'Frame_number':frame_list, 'Class_label': best_class, 'x':x, 'y':y,
          'top_right': top_right, 'bottom_left': bottom_left, 'Confidence': confidence_list}
    result = pd.DataFrame(df)
    result = result.sort_values('Frame_number')
    print("FINAL RESULTS:\n",result)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    print('OUTPUT FILE PATH:', output_path+output_file_name+'.csv')
    result.to_csv(output_path+output_file_name, index=False)