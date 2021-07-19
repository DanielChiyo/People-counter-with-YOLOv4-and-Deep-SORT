import os
import csv
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    amount_by_now = 0
    total_entered = 0
    total_left = 0
    #possible states of the tracked objects
    Occluded = 0 #object was not detected in the frame
    Q1 = 1 # below the middle of the video
    Q2 = 2 # above the middle of the video

    time_in_seconds = 0


    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
        fps = vid.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    except:
        vid = cv2.VideoCapture(video_path)
        fps = vid.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fps = int(fps)
    print("size: " + str(size[1]))

    Y_max = int(size[1]) # change to real value
    Y_center =  int(Y_max/2)

    #         INSIDE THE STORE

    #------------------------------------#  video y = 0
    #                                    #
    #                 Q2                 #
    #                                    #
    #------------------------------------#  vertical center of the video
    #                                    #
    #                 Q1                 #
    #                                    #
    #------------------------------------#  video y = y_max

    #         OUTSIDE THE STORE

    N = 500 # limit of tracked objects that are entering or leaving
    frames_per_analysis = fps # size of the sequence to be analyzed
    # in reality will be used (frames_per_analysis + 1) 


    # valid state is any state that is not 0 (occluded object)
    first_valid_state = np.zeros(shape=(N,1)) # stores the first valid state of a sequence of frames
    last_valid_state = np.zeros(shape=(N,1)) # stores the last valid state of a sequence of frames
    # if the last valid state is 2 and the first is 1, then somebody entered
    # if the last valid state is 1 and the first is 2, then somebody leaved
    # otherwise nothing happens


    #initializing object_states matrix
    object_states = np.zeros(shape=(N,(frames_per_analysis + 1))) 

    # code EOF case (run one last time the matrix analysis)


    flag_first = 0
    flag_last = 0
    entered = 0
    left = 0
    result = 0
    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running

    with open('mycsv.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['TIME', 'ENTERED', 'LEFT'])

        while True:
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                video_current_time = vid.get(cv2.CAP_PROP_POS_MSEC)/1000
                print("TIME: {}".format(video_current_time))    
                print('Video has ended or failed, try a different video format!')
                break
            frame_num +=1
            #print('Frame #: ', frame_num)
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            # run detections on tflite if flag is set
            if FLAGS.framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                # run detections using yolov3 if flag is set
                if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )

            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())
            
            # custom allowed classes (uncomment line below to customize tracker for only people)
            allowed_classes = ['person']

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)
            if FLAGS.count:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                print("Objects being tracked: {}".format(count))
            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
                
            # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.circle(frame, (int((int(bbox[0])+int(bbox[2]))/2) ,int((int(bbox[1])+int(bbox[3]))/2)), radius=6, color=(0, 255, 0), thickness=-1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                ########################################################################
                cv2.line(frame, (0, Y_center - 100), (int(size[0]), Y_center -100), (255, 255, 0), thickness=2)


            # if enable info flag then print details about each track
                if FLAGS.info:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            


                #random position y (0 to y_max)
                
                #for object in detected_objects_id:
                
                #print("saving objects status to matrix")
                #setting random position for objects

                    

                aux = ( int(bbox[1]) + int(bbox[3]) ) / 2
                if(aux < Y_center - 100):
                    state = Q2
                else:   
                    state = Q1

                object_states[track.track_id][frame_num%frames_per_analysis + 1] = state
                if(track.track_id==6):
                    print("Frame: {}, ObjectID: {}, status: {}, written status: {}".format(frame_num, track.track_id, state, object_states[6][frame_num%frames_per_analysis + 1] ))


            #print("status objeto 2:", object_states[2][frame_num%frames_per_analysis + 1])
            if(frame_num%frames_per_analysis == 0):
                time_in_seconds = time_in_seconds + 1
                print('Frame #: {} - {} seconds elapsed'.format(frame_num, time_in_seconds))
                row = 0
                col = 0
                print(object_states[50])
                #print(object_states)
                while row < N:
                    col = 0
                    ant = 0
                    atual = 0
                    cont = 0
                    while col < frames_per_analysis + 1:
                        if(object_states[row][col] != 0):
                            ant = atual
                            atual = object_states[row][col]
                            if(ant == 1 and atual == 2):
                                cont = cont + 1
                            elif(ant == 2 and atual == 1):
                                cont = cont - 1
                        object_states[row][0] = 0
                        col = col + 1
                    if(cont == 1):
                        entered = entered + 1
                        print("Object {} entered".format(row))
                    if(cont == -1):
                        left = left + 1
                        print("Object {} left".format(row))


                    object_states[row][0] = atual
                    row = row + 1


                result = entered - left
                amount_by_now = amount_by_now + result
                print("Entered: {} \nLeft: {} \nResult: {} \n Total Result: {}".format(entered, left, result, amount_by_now))
                total_entered = total_entered + entered
                total_left = total_left + left
                print("Total entered: {} \n Total left: {}".format(total_entered, total_left))
                
                video_current_time = vid.get(cv2.CAP_PROP_POS_MSEC)/1000
                print("TIME: {}".format(video_current_time))           
                
                writer.writerow([video_current_time, entered, left])


                entered = 0
                left = 0
                #print("Entered: {} \nLeft: {}".format(entered, left))
                print("-----------------------")
                print("-----------------------")

            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            #print("FPS: %.2f" % fps)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if not FLAGS.dont_show:
                cv2.imshow("Output Video", result)
            
            # if output flag is set, save video file
            if FLAGS.output:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass