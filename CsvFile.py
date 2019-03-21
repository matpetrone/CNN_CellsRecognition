import csv
import numpy as np
from skimage import io
import os
import pandas as pd

def editFileName(string):
    new = list(string)
    new[3] = 'c'
    new[4] = 'e'
    new[5] = 'l'
    new[6] = 'l'
    return ''.join(new)

def editFileName_n(string):
    new = list(string)
    new[4] = 'c'
    new[5] = 'e'
    new[6] = 'l'
    new[7] = 'l'
    return ''.join(new)
#method that create a csvfile from landmarks in pics
landmarks_files = ['001dots.png','002dots.png','003dots.png','004dots.png','005dots.png','006dots.png','007dots.png','008dots.png','009dots.png','010dots.png','011dots.png','012dots.png','013dots.png','014dots.png','015dots.png','016dots.png','017dots.png','018dots.png','019dots.png','020dots.png','021dots.png','022dots.png','023dots.png','024dots.png','025dots.png','026dots.png','027dots.png','028dots.png','029dots.png','030dots.png','031dots.png','032dots.png','033dots.png','034dots.png','035dots.png','036dots.png','037dots.png','038dots.png','039dots.png','040dots.png','041dots.png','042dots.png','043dots.png','044dots.png','045dots.png','046dots.png','047dots.png','048dots.png','049dots.png','050dots.png','051dots.png','052dots.png','053dots.png','054dots.png','055dots.png','056dots.png','057dots.png','058dots.png','059dots.png','060dots.png','061dots.png','062dots.png','063dots.png','064dots.png','065dots.png','066dots.png','067dots.png','068dots.png','069dots.png','070dots.png','071dots.png','072dots.png','073dots.png','074dots.png','075dots.png','076dots.png','077dots.png','078dots.png','079dots.png','080dots.png','081dots.png','082dots.png','083dots.png','084dots.png','085dots.png','086dots.png','087dots.png','088dots.png','089dots.png','090dots.png','091dots.png','092dots.png','093dots.png','094dots.png','095dots.png','096dots.png','097dots.png','098dots.png','099dots.png','100dots.png','101dots.png','102dots.png','103dots.png','104dots.png','105dots.png','106dots.png','107dots.png','108dots.png','109dots.png','110dots.png','111dots.png','112dots.png','113dots.png','114dots.png','115dots.png','116dots.png','117dots.png','118dots.png','119dots.png','120dots.png','121dots.png','122dots.png','123dots.png','124dots.png','125dots.png','126dots.png','127dots.png','128dots.png','129dots.png','130dots.png','131dots.png','132dots.png','133dots.png','134dots.png','135dots.png','136dots.png','137dots.png','138dots.png','139dots.png','140dots.png','141dots.png','142dots.png','143dots.png','144dots.png','145dots.png','146dots.png','147dots.png','148dots.png','149dots.png','150dots.png','151dots.png','152dots.png','153dots.png','154dots.png','155dots.png','156dots.png','157dots.png','158dots.png','159dots.png','160dots.png', '161dots.png','162dots.png','163dots.png','164dots.png','165dots.png','166dots.png','167dots.png','168dots.png','169dots.png','170dots.png','171dots.png', '172dots.png','173dots.png','174dots.png','175dots.png','176dots.png','177dots.png','178dots.png','179dots.png','180dots.png','181dots.png','182dots.png', '183dots.png','184dots.png','185dots.png','186dots.png','187dots.png','188dots.png','189dots.png','190dots.png','191dots.png','192dots.png','193dots.png', '194dots.png','195dots.png','196dots.png','197dots.png','198dots.png','199dots.png', '200dots.png']

def create_csv(landmarks_list):
    with open('cells_landmarks.csv', mode='w') as landmarks_file:
        landmarks_writer = csv.writer(landmarks_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['image_name']
        for i in range(320):
            header += ['part_{}_x'.format(i), 'part_{}_y'.format(i)]
        landmarks_writer.writerow(header)
        for k in range(len(landmarks_list)):
            coordinates = np.array([])
            img = io.imread(os.path.abspath('CellsDataset/{}'.format(landmarks_list[k])))
            coordinates = np.append(coordinates, editFileName(landmarks_list[k]))
            for i in range(img.shape[1]):
                for j in range(img.shape[0]):
                    if (list(img[i, j]) != [0, 0, 0]):
                        coordinates = np.append(coordinates, [j, i])
            landmarks_writer.writerow(coordinates)

def create_csv_n(landmarks_list):
    with open('cells_n_landmarks.csv', mode='w') as landmarks_file:
        landmarks_writer = csv.writer(landmarks_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['image_name']
        for i in range(320):
            header += ['part_{}_x'.format(i), 'part_{}_y'.format(i)]
        landmarks_writer.writerow(header)
        for k in range(len(landmarks_list)):
            coordinates = np.array([])
            img = io.imread(os.path.abspath('CellsDataset_n/{}'.format(landmarks_list[k])))
            coordinates = np.append(coordinates, editFileName_n(landmarks_list[k]))
            for i in range(img.shape[1]):
                for j in range(img.shape[0]):
                    if (list(img[i, j]) != [0, 0, 0]):
                        coordinates = np.append(coordinates, [j, i])
            landmarks_writer.writerow(coordinates)



#create_csv()


def shuffler(filename):
  df = pd.read_csv(filename, header=0)
  # return the pandas dataframe
  return df.reindex(np.random.permutation(df.index))

def create_shuffle_csv(csv_file, outputfilename):
  shuffler(csv_file).to_csv(outputfilename, quoting=csv.QUOTE_MINIMAL , index=False, encoding='utf-8', na_rep = ' ')

def createList_2000():
    list =[]
    for i in range(2000):
        j=i+1
        if j<10:
            str = '000'+ j.__str__() +'dots_n.png'
        elif j<100:
            str = '00' + j.__str__() + 'dots_n.png'
        elif j<1000:
            str = '0' + j.__str__() + 'dots_n.png'
        elif j>=1000:
            str = j.__str__() + 'dots_n.png'
        list.append(str)
    return list