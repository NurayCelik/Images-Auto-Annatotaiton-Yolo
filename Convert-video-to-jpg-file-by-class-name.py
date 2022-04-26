
import cv2
import numpy as np
import os, shutil
from pathlib import Path
import datetime
     
file_name = "D61_1_2-9.mp4"
cap = cv2.VideoCapture("videolar/"+file_name)

count = 0
try:
    while True:
        
        success,frame = cap.read()
        at = str(datetime.datetime.now()).replace(" ", '').replace(":", '').replace("-", '')
        sep = '.'
        dt = at.split(sep, 1)[0]
        
        print('Read a new frame: ', success)
        
        #frame = cv2.flip(frame,1)
        #frame = cv2.resize(frame,(608,608),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
       
        frame_blob = cv2.dnn.blobFromImage(frame, 0.00392, (608,608),(0, 0, 0),swapRB=True, crop=False)
    
        labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                    "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
                    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
                    "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
                    "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
                    "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
                    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
                    "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
                    "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]
    
           
        colors = ["0,0,255","0,0,255","255,0,0","255,255,0","0,255,0"]
        colors = [np.array(color.split(",")).astype("int") for color in colors]
        colors = np.array(colors)
        colors = np.tile(colors,(18,1))
    
    
        model = cv2.dnn.readNetFromDarknet("model//yolo-obj.cfg","model//yolov4.weights")
    
        layers = model.getLayerNames()
        output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]
        
        model.setInput(frame_blob)
        
    
        
        detection_layers = model.forward(output_layer)
    
    
        ############## NON-MAXIMUM SUPPRESSION - OPERATION 1 ###################
        
        ids_list = []
        boxes_list = []
        confidences_list = []
        bounding_box=[]
        
        ############################ END OF OPERATION 1 ########################
        
        for detection_layer in detection_layers:
            for object_detection in detection_layer:
                
                scores = object_detection[5:]
                predicted_id = np.argmax(scores)
                confidence = scores[predicted_id]
                
                if confidence > 0.5:
                    
                    label = labels[predicted_id]
                    bounding_box = object_detection[0:4] * np.array([frame_width,frame_height,frame_width,frame_height])
                    (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
                    
                     
                    start_x = int(box_center_x - (box_width/2))
                    start_y = int(box_center_y - (box_height/2))
                    
                   
                    
                    ############## NON-MAXIMUM SUPPRESSION - OPERATION 2 ###################
                    
                    ids_list.append(predicted_id)
                    confidences_list.append(float(confidence))
                    boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
                    
                    ############################ END OF OPERATION 2 ########################
                                     
         
                    
        ############## NON-MAXIMUM SUPPRESSION - OPERATION 3 ###################
                    
        max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.6)
        
        for max_id in max_ids:
         
            max_class_id = max_id[0]
            box = boxes_list[max_class_id]
            
            start_x = box[0] 
            start_y = box[1] 
            box_width = box[2] 
            box_height = box[3] 
            predicted_id = ids_list[max_class_id]
            label = labels[predicted_id]
            confidence = confidences_list[max_class_id]
            
                     
        ############################ END OF OPERATION 3 ########################
                    
            end_x = start_x + box_width
            end_y = start_y + box_height
            
            print(end_x)
            print(end_y)
            print(start_x)
            print(start_y)
            print(frame.shape)
             
            box_color = colors[predicted_id]
            box_color = [int(each) for each in box_color]
            
            Path("images/"+file_name).mkdir(parents=True, exist_ok=True)
            if label == "person": 
                cv2.imwrite("images/"+file_name+"/D61_1_2-9.mp4_%s_%#06d.jpg" %(dt,count), frame,[int(cv2.IMWRITE_JPEG_QUALITY), 100])   # save frame as JPEG file   
                print('Read a new frame: ', success)
                print('dt = ' + str(dt))
                
                '''
                label = "{}: {:.2f}%".format(label, confidence*100)
                print("predicted object {}".format(label))
                 
                print(label)
                        
                cv2.rectangle(frame, (start_x,start_y),(end_x,end_y),box_color,2)
                cv2.rectangle(frame, (start_x-1,start_y),(end_x+1,start_y-30),box_color,-1)
                cv2.putText(frame,label,(start_x,start_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                '''
                
                
                count += 1
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        cv2.imshow("Detector",frame)
        
except Exception as e:
    print(str(e))

cap.release()
cv2.destroyAllWindows()
