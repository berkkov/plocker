import cv2
import numpy as np

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
maskThreshold = 0.3  # Mask threshold


def load_net(modelWeights, textGraph):
    # Load the network
    net = cv2.dnn.readNetFromTensorflow(modelWeights, textGraph)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

    return net


def load_class_names(path):
    classes = None
    with open(path, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes


def load_colors(path):
    with open(path, 'rt') as f:
        colorsStr = f.read().rstrip('\n').split('\n')
    colors = []
    for i in range(len(colorsStr)):
        rgb = colorsStr[i].split(' ')
        color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
        colors.append(color)

    return colors


def postprocess(frame, t, r, b, l, area, boxes, masks, classes, colors):
    # Output size of masks is NxCxHxW where
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape
    numClasses = masks.shape[1]
    numDetections = boxes.shape[2]

    frameH = frame.shape[0]
    frameW = frame.shape[1]

    for i in range(numDetections):
        box = boxes[0, 0, i]
        mask = masks[i]
        score = box[2]
        if score > confThreshold:
            classId = int(box[1])
            if classId != 0:
                continue
            # Extract the bounding box
            left = int(frameW * box[3])
            top = int(frameH * box[4])
            right = int(frameW * box[5])
            bottom = int(frameH * box[6])

            left = max(0, min(left, frameW - 1))
            top = max(0, min(top, frameH - 1))
            right = max(0, min(right, frameW - 1))
            bottom = max(0, min(bottom, frameH - 1))

            i_left = max(l, left)
            i_right = min(r, right)
            i_top = max(t, top)
            i_bottom = min(b, bottom)
            i_area = (i_bottom - i_top) * (i_right - i_left)
            ratio = i_area / area
            if ratio < 0.10:
                print('skipped ', ratio)
                continue

            # Extract the mask for the object
            classMask = mask[classId]

            # Draw bounding box, colorize and show the mask on the image
            drawBox(frame, classId, score, left, top, right, bottom, classMask, classes, colors)

    return frame


def postprocess2(frame, t, r, b, l, area, result, classes, colors):
    # Output size of masks is NxCxHxW where
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape

    numDetections = len(result[0]['rois'])

    frameH = frame.shape[0]
    frameW = frame.shape[1]

    for i in range(numDetections):
        box = result[0]['rois'][i]
        mask = result[0]['masks'][i]
        score = result[0]['scores'][i]

        if score > confThreshold:
            classId = result[0]['class_ids'][i]
            if classId != 1:
                continue
            # Extract the bounding box
            left = int(frameW * box[0])
            top = int(frameH * box[1])
            right = int(frameW * box[2])
            bottom = int(frameH * box[3])

            left = max(0, min(left, frameW - 1))
            top = max(0, min(top, frameH - 1))
            right = max(0, min(right, frameW - 1))
            bottom = max(0, min(bottom, frameH - 1))

            i_left = max(l, left)
            i_right = min(r, right)
            i_top = max(t, top)
            i_bottom = min(b, bottom)
            i_area = (i_bottom - i_top) * (i_right - i_left)
            ratio = i_area / area
            if ratio < 0.50:
                continue

            drawBox(frame, classId, score, left, top, right, bottom, mask, classes, colors)

    return frame


# Draw the predicted bounding box, colorize and show the mask on the image
def drawBox(frame, classId, conf, left, top, right, bottom, classMask, classes, colors):
    # Print a label of class.
    label = '%.2f' % conf
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])

    # Resize the mask, threshold, color and apply it on the image
    classMask = cv2.resize(classMask, (right - left + 1, bottom - top + 1))
    mask = (classMask > maskThreshold)

    #color = colors[classId % len(colors)]
    color = (151, 160, 175)

    # Draw the contours on the image
    mask = mask.astype(np.uint8)
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame[top:bottom + 1, left:right + 1], contours, -1, color, -1, cv2.LINE_8, hierarchy, 100)
