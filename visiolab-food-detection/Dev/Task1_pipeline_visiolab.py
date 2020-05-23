import numpy as np
import cv2 as cv
import tensorflow as tf
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

label_map = {1:'0',2:'1',3:'2',4:'3',5:'4',6:'5',7:'6'}

TEST = True

def show_frame(image, caption='Frame to see'):
    cv.imshow(caption, image)
    cv.waitKey(0)

def save_img(image, filename):
    cv.imwrite(filename, image)

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


            # if score > 0.3:
            # Have taken the best detection confidence because
            # no decoy class and definitely one class of trained classes per image is present in the test set.
            score_list.append(score)
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            tuple_data = (x, y, right, bottom)
            bbox_list.append(tuple_data)
            class_label.append(classId)

            if TEST:
                cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

        max_id = np.argmax(score_list)
        class_selected = class_label[max_id]
        bbox_selected = bbox_list[max_id]
        confidence = score_list[max_id]

    return class_selected, bbox_selected, confidence

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visiolab- Kaggle Challenge')
    parser.add_argument("inp", type=str, help="input images path")
    parser.add_argument("out", type=str,nargs='?', default='Result/', help="output csv")

    args = parser.parse_args()

    data_path = args.inp
    output_path = args.out
    output_file_name = 'result.csv'
    frozen_graph_path = './frozen_inference_graph.pb'
    image_names = os.listdir(data_path)
    print("READER INPUT SHAPE:", len(image_names))
    best_class, best_bb, image_name_list, confidence_list, best_class_ = list(), list(), list(), list(), list()
    for i in image_names:
        image = plt.imread(os.path.join(data_path, i))
        class_inferred, bb_inferred, confidence = run_inference(frozen_graph_path, image)
        best_class.append(class_inferred)
        best_bb.append(bb_inferred)
        image_name_list.append(i)
        confidence_list.append(confidence)

    for i in best_class:
        best_class_.append(label_map[i])

    df = {'Test_image':image_name_list, 'Class_label': best_class_, 'Bounding_box': best_bb, 'Confidence': confidence_list}
    result = pd.DataFrame(df)
    result = result.sort_values('Test_image')
    print("FINAL RESULTS:\n",result)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    print('OUTPUT FILE PATH:', output_path+output_file_name)
    result.to_csv(output_path+output_file_name, index=False)





