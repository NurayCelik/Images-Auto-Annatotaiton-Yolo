
import cv2
import numpy as np
import os, shutil

# =============================================================================
# def removeFolder(folder):
#     for filename in os.listdir(folder):
#         file_path = os.path.join(folder, filename)
#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)
#         except Exception as e:
#             print('Failed to delete %s. Reason: %s' % (file_path, e))
#             
# removeFolder("obj_train_data")
# =============================================================================


open('train.txt', 'w').close()
    
def imageLabelling(readfile):  
    img = cv2.imread("obj_train_data/"+readfile)
    #img = cv2.resize(img,(608,608),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    
    with open("train.txt", 'a') as f:
        f.write('data/obj_train_data/%s' % readfile)
        print('data/obj_train_data/%s' % readfile)
        f.write('\n')
        f.close()
       
    #frame = cv2.flip(frame,1) #ters çevirir görüntüyü sağ sol olur.
    img = cv2.resize(img,(608,608),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    
    img_width = img.shape[1]
    img_height = img.shape[0]
   
    img_blob = cv2.dnn.blobFromImage(img, 0.00392, (608,608),(0, 0, 0),swapRB=True, crop=False)

    labels = ["person","car", "gozluk"]

         
    colors = ["0,0,255","0,0,255","255,0,0","255,255,0","0,255,0"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors,(18,1))


    model = cv2.dnn.readNetFromDarknet("model//yolo-obj.cfg","model//yolo-obj_final.weights")

    layers = model.getLayerNames()
    output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]
    
    model.setInput(img_blob)
    

    
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
           
            if confidence > 0.3:
                
                label = labels[predicted_id]
                print(label)
         
                bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
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
                
        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]
                              
        label = "{}: {:.2f}%".format(label, confidence*100)
        #print("predicted object {}".format(label))
         
               
        cv2.rectangle(img, (start_x,start_y),(end_x,end_y),box_color,2)
        #cv2.rectangle(frame, (start_x-1,start_y),(end_x+1,start_y-30),box_color,-1)
        cv2.putText(img,label,(start_x,start_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    lines  = [] 
    for index in range(len(boxes_list)):
        if index in max_ids:
            lines  = [int(ids_list[index]), float(box_center_x/img_width), float(box_center_y/img_height),float(end_x - start_x)/img_width, float(end_y - start_y)/img_height]  
            #lines  = [int(0), float(box_center_x/frame_width), float(box_center_y/frame_height),float(end_x - start_x)/frame_width, float(end_y - start_y)/frame_height]  
    content = readfile.split(".jpg")
    txtfile = content[0]+".txt"
    with open("obj_train_data/"+ txtfile,'w') as f:
        if len(boxes_list) == 0:
            f.write(" ")
        else:
          for line in lines:
             f.write(str(line))
             f.write(' ')
          f.write('\n')
          f.close()
    
                         
    cv2.imshow("Detection Window", img)  
    if cv2.waitKey(1) & 0xFF == ord("q"):
       return  
   


for readfile in os.listdir("obj_train_data/"):
    if readfile.endswith(".jpg"):
        print("obj_train_data/"+readfile)
        imageLabelling(readfile) 
        
