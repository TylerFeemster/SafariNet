import numpy as np
from bs4 import BeautifulSoup
import os

for i in range(1, 5784+1):
    str_i = ('00000' + str(i))[-6:]
    original =  f'./wild/Annotations/2018_{str_i}.xml'
    revised = f'./wild/AnnCopy/2018_{str_i}.xml'
    os.rename(original, revised)