import numpy as np
from bs4 import BeautifulSoup
import os

'''
Reads XML file associated to each image and records
'count' data in count_annotations and 'classification'
data in class_annotations
'''

# Create folders
os.makedirs('./wild/class_annotations', exist_ok=True)
os.makedirs('./wild/count_annotations', exist_ok=True)

REDUCED_CLASSES = ['giraffe_reticulated','zebra_grevys',
                   'turtle_sea','zebra_plains',
                   'giraffe_masai','whale_fluke']
CLASS2IDX = {REDUCED_CLASSES[idx] : idx 
                  for idx in range(6)}

for i in range(1, 5784 + 1):
    str_i = ('00000' + str(i))[-6:]
    xml_file = f'./wild/Annotations/2018_{str_i}.xml'
    with open(xml_file, 'r') as f:
        file = f.read()
    soup = BeautifulSoup(file, 'xml')

    class_array = np.zeros(6)
    count_array = np.zeros(6)
    
    names = soup.find_all('name')
    for name in names:
        animal = name.text
        if animal in REDUCED_CLASSES:
            class_array[CLASS2IDX[animal]] = 1
            count_array[CLASS2IDX[animal]] += 1

    fn = f'2018_{str_i}.npy'
    np.save(f'./wild/class_annotations/{fn}', class_array)
    np.save(f'./wild/count_annotations/{fn}', count_array)
