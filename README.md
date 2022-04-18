# Images-Auto-Annotation-Yolo
Images Auto Annotation Cvat - Yolo Format - OpenCV


video_spliy.py - > We divide the videos into the minutes we want (300 sec = 5 minutes by specifying in the times.txt file) to work more easily.


Convert-video-to-jpg-file-by-class-name.py -> we convert the shorter videos we split into jpg files according to the model. While doing this, we do not add the objects other than those objects to the image folder, we do not convert them into jpg files.
We add these new jpg files as a task on the CVAT site. We download this task to our computer as an export task dataset in Yolo 1.1 format.


Directory_file_extension_remove.py -> We delete the empty txt files in the obj_train_data folder in the downloaded dataset.
We retag the remaining jpg files in the image_make_task.py -> obj_train_data folder. We print their descriptions in txt format to the obj_train_data folder.


We add the new obj_train_data folder to the downloaded dataset and convert it to zip format. We upload this zip file to our CVAT task in upload annotation - yolo 1.1 format.
