# Github: https://github.com/ultralytics/yolov5
# Documentation main page: https://docs.ultralytics.com/
# Documentation Pytorch Hub: https://docs.ultralytics.com/tutorials/pytorch-hub/

import torch
import numpy as np
import cv2 as cv
import time

record_video = True
show_video = False

fps = 25 # Frames per second

# Video
capture = cv.VideoCapture('video.mp4')

width_cap = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
height_cap = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))

# Write video
# WINDOWS -- *'DIVX'
# MAC OR LINUX -- *'XVID'
writer = cv.VideoWriter('video_result.mp4',cv.VideoWriter_fourcc(*'XVID'),fps,(width_cap,height_cap))

if capture.isOpened() == False:
  print('ERROR FILE NOT FOUND OR WRONG CODEC USED!')

# Model loading
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')  # or yolov5n - yolov5x6, custom

# Model parameters
model.conf = 0.25  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1)
model.classes = [2, 3, 5, 7]  # (optional list) filter by class, [2=car, 3=motorcycle, 5=bus, 7=truck]. None is all classess.

# Generates colors for each class
np.random.seed(0)
color_matrix = np.random.randint(0, 255, size=(len(model.names),3))

print(model.names)

vehicles_frame_data = []

frame_count = 0

while capture.isOpened():

  ret, frame = capture.read()

  if ret == True:

    (height, width) = (frame.shape[0], frame.shape[1])
    
    region_points = np.array([
      [width * 0.4, height * 0.45],
      [ 0, height * 0.80],
      [0, height],
      [width * 0.43, height],
      [width * 0.49, height * 0.45]
    ], dtype=np.int32)

    mask = np.zeros(frame.shape[0:2], dtype=np.uint8)

    cv.drawContours(mask, [region_points], -1, (255, 255, 255), -1, cv.LINE_AA)

    cropped = cv.bitwise_and(frame,frame,mask = mask)

    # Inference
    pred = model(cropped).pandas().xyxy[0]
    # pred_centroid = model(frame).pandas().xywh[0] # centroid and bounding box

    # Class data from each frame
    unique, counts = np.unique(pred['name'], return_counts=True)
    class_freq = dict(zip(unique, counts))
    vehicles_frame_data.append(class_freq)
    # Display YOLO tracking

    for idx in range(0, len(pred)):

      xmin = int(round(pred['xmin'][idx]))
      ymin = int(round(pred['ymin'][idx]))
      xmax = int(round(pred['xmax'][idx]))
      ymax = int(round(pred['ymax'][idx]))

      class_label = pred['name'][idx]

      class_idx = pred['class'][idx]

      confidence_text = "{:.2f}".format(pred['confidence'][idx])

      full_text = f"{class_label} {confidence_text}"

      text_count = len(full_text)

      box_color = color_matrix[class_idx,:].tolist() 
      text_color = (255,255,255)

      cv.rectangle(frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=box_color, thickness=3)
      cv.rectangle(frame, pt1=(xmin, ymin), pt2=(xmin + (14 * text_count), ymin - 25), color=box_color, thickness=-1)
      cv.putText(frame, text=full_text, org=( xmin, ymin ),fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=text_color,thickness=2, lineType=cv.LINE_AA)
      
    cv.putText(frame, text=f"Frame: {frame_count}", org=( 10, 50 ),fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 0, 0),thickness=2, lineType=cv.LINE_AA)
    cv.putText(frame, text="Time(sec): {:.2f}".format(frame_count/fps), org=( 10, 100 ),fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 0, 0),thickness=2, lineType=cv.LINE_AA)

    frame_count += 1

    zero_mask = np.zeros_like(frame)

    # Display the coloured ROI
    cv.fillPoly(zero_mask, [region_points], color=(255,0,0))

    blend_image = cv.addWeighted(frame, 1.0, zero_mask, 0.5, 0.0)
    # WRITER Delay
    # time.sleep(1/50)

    # Write video
    if record_video:
       writer.write(blend_image)

    if show_video:
      cv.imshow('frame', blend_image)
      
      # Press "q" to quit on the window
      if cv.waitKey(10) & 0xFF == ord('q'):
          break
  else:
      break

capture.release()
cv.destroyAllWindows()


if record_video:
  print("Video saved!")
print("Complete!")
print("Vehicle data: ", vehicles_frame_data)