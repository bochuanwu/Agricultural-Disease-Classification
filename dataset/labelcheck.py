# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 21:49:30 2018

@author: 16703
"""

import json
import numpy as np
import matplotlib.pyplot as plt
submit_file='./AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations11.json'
user_result_list = json.load(open(submit_file))
x=np.zeros(61,)
for each_item in user_result_list:
    label_id = each_item['disease_class']
    name_id = each_item["image_id"]
    x[label_id]=int(x[label_id])+1
    if  label_id==53:
        fr = open(submit_file, 'a')
        name_emb={"image_id":str(name_id),"disease_class":int(label_id)}
        model=json.dumps(name_emb)
        fr.write(model)  
        fr.close()
    n=[]
for i in x:
    n.append(i)

print(n[5],n[22],n[52],n[53])

rects=plt.bar(range(len(n)), n, color='rgby')
plt.savefig('./test.png')
plt.show()