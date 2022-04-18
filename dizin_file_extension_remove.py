
'''
jpg dosyları arasındaki txt dosyalarını siliyoruz
'''

import os


dir_name = "/home/eventgates/Desktop/python/opencv/task_olusturma_image/obj_train_data/"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".txt"):
        os.remove(os.path.join(dir_name, item))
