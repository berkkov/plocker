import cv2
import face_recognition
import numpy as np
from utils import postprocess, load_class_names, load_colors, load_net, postprocess2

# Fill these
NAME_TO_BLOCK = ''
VIDEO = ''
# Get graph, weight and class files from opencv lib.
GRAPH_PATH = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
WEIGHT_PATH = "frozen_inference_graph.pb"

net = load_net(WEIGHT_PATH, GRAPH_PATH)
classes = load_class_names(path="mscoco_labels.names")
colors = load_colors(path="colors.txt")
#model = mask.create_model()

cap = cv2.VideoCapture(VIDEO)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(length)
outputFile = VIDEO[:-4] + '_blocked.avi'

vid_writer = cv2.VideoWriter(outputFile,
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                             28,
                             (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Change these two lines to the person to be blocked
berk_image = face_recognition.load_image_file('known\\berk.jpg')
berk_encoding = face_recognition.face_encodings(berk_image)[0]

known_encodings = [berk_encoding]
known_encoding_names = ["berk"]
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
count = 0

while cv2.waitKey(1) < 0:
    count += 1
    print(count)
    # Get frame from the video
    hasFrame, frame = cap.read()

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv2.waitKey(3000)
        break

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

    # results = model.detect([frame])

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_matches_index = matches.index(True)
                name = known_encoding_names[first_matches_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):

        if name != NAME_TO_BLOCK:
            continue
        print(name)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        t2 = top
        r2 = right
        b2 = bottom
        l2 = left
        area = (bottom - top)*(right-left)

        #frame = postprocess2(frame, t2, r2, b2, l2, area, results, classes, colors)
        frame = postprocess(frame, t2, r2, b2, l2, area, boxes, masks, classes, colors)
    vid_writer.write(frame.astype(np.uint8))
