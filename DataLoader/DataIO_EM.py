import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scipy.io as scio
import copy
from skimage.morphology import square, binary_closing
import cv2
from PIL import Image
import pathlib

addUnlab_path_dict = {'Crack500':os.path.abspath('../Dataset/Crack500/CRACK500-20200128T063606Z-001/CRACK500/Cutomized')}

class DataIO():
    def __init__(self,batch_size, seed_split=0, seed_label=80, label_percent=0.05,
                 data_path=os.path.abspath(os.path.join(pathlib.Path(__file__).parent.resolve(),'../Dataset/EM')),
                 add_unlab = 'None', crop_size=128):
        self.crop_size = crop_size
        self.data_path = os.path.join(data_path,'Cropped{}'.format(self.crop_size))
        self.img_path = os.path.join(self.data_path,'img')
        self.gt_path = os.path.join(self.data_path,'gt')
        self.num_all_data = -1
        self.resize_image_size = (self.crop_size, self.crop_size)
        self.original_image_size = (self.crop_size, self.crop_size)

        self.label_percent = label_percent
        self.seed_split = seed_split
        self.seed_label = seed_label
        self.batch_size = batch_size
        self.add_unlab = add_unlab

        self.train_index = None
        self.val_index = None
        self.test_index = None
        self.num_train = None
        self.num_val = None
        self.num_test = None

        self.InitPointer()



    def LoadDataset(self):
        data_idx = 0
        self.all_data = {}
        self.num_train = 0
        self.num_val = 0
        self.num_test = 0

        self.train_names = []
        self.val_names = []
        self.test_names = []

        with open(os.path.join(self.data_path, 'train.txt'), 'r') as fid:
            file_list = fid.readlines()
            train_img_list = []

            for f_i in file_list:
                train_img_list.append(f_i.rstrip())

                img_name = train_img_list[-1]
                img = cv2.resize(plt.imread(os.path.join(self.img_path, img_name)),self.resize_image_size)
                seg = cv2.resize(plt.imread(os.path.join(self.gt_path, img_name)),self.resize_image_size)

                self.all_data.update(
                    {img_name:
                         {'img': (255*img).astype(int),
                          'gt': seg,
                          'gt_org': None,
                          'name': img_name,
                          'index': data_idx}})
                data_idx += 1
                self.num_train += 1
                self.train_names.append(img_name)
        np.sort(self.train_names)

        with open(os.path.join(self.data_path, 'test.txt'), 'r') as fid:
            file_list = fid.readlines()
            val_img_list = []

            for f_i in file_list:
                val_img_list.append(f_i.rstrip())
                img_name = val_img_list[-1]
                img = cv2.resize(plt.imread(os.path.join(self.img_path, img_name)),self.resize_image_size)
                seg = cv2.resize(plt.imread(os.path.join(self.gt_path, img_name)),self.resize_image_size)

                self.all_data.update(
                    {img_name:
                         {'img': img,
                          'gt': seg,
                          'gt_org': None,
                          'name': img_name,
                          'index': data_idx}})

                data_idx += 1
                self.num_val += 1
                self.val_names.append(img_name)

        self.num_test = self.num_val
        self.num_all_data = self.num_train + self.num_val

        self.all_add_data = {}
        self.add_train_names = []
        if self.add_unlab != 'None':
            addUnlab_filepath = os.path.join(addUnlab_path_dict[self.add_unlab], 'train4Crack500.mat')
            tmp = scio.loadmat(addUnlab_filepath)
            allImgNames = tmp['allImgNames'][0].split(' ')
            data_idx = 0
            for img, gt, img_name in zip(tmp['allImgs'], tmp['allGTs'], allImgNames):
                self.all_data.update(
                    {img_name: {'img': img, 'gt': gt, 'gt_thin': None, 'name': img_name, 'index': data_idx}})
                data_idx += 1
                self.add_train_names.append(img_name)


    def InitDataset(self, split_filepath=None):
        self.LoadDataset()
        self.GetDatasetMeanVar()



    def InitDataset_EqLabUnlab(self, lab_ratio=1., seed=0):
        self.LoadDataset()
        self.GenerateSplit_EqLabUnlab(lab_ratio,seed)
        self.GetDatasetMeanVar()


    def GenerateSplit_EqLabUnlab(self, lab_ratio, seed=0):
        self.num_train_labeled = int(np.ceil(self.num_train*self.label_percent))
        self.num_train_unlabeled = self.num_train - self.num_train_labeled + len(self.add_train_names)

        np.random.seed(self.seed_label)
        np.random.shuffle(self.train_names)
        self.train_labeled_names = self.train_names[0:self.num_train_labeled]
        self.train_labeled_names_active = self.train_labeled_names.copy()
        self.train_unlabeled_names = self.train_names[self.num_train_labeled::]
        self.train_unlabeled_names_active = self.train_unlabeled_names.copy()

        self.batch_size_train_labeled = int(np.ceil(self.batch_size*lab_ratio))
        self.batch_size_train_unlabeled = self.batch_size - self.batch_size_train_labeled

        self.lab_ratio = lab_ratio

        distribution_temp = []
        for name in self.train_labeled_names:
            distribution_temp.append(np.mean(self.all_data[name]['gt']))
        self.distribution = [round(np.mean(distribution_temp), 9), round(1 - np.mean(distribution_temp), 9)]


    def GetDatasetMeanVar(self):
        tmp_allpix = []  # tmp container for all pixels
        tmp_allmask = []  # tmp container for all mask pixels

        for img_name in self.train_names:
            tmp_allpix.append(self.all_data[img_name]['img'].reshape(-1))
            if self.all_data[img_name]['gt'] is not None:
                tmp_allmask.append(self.all_data[img_name]['gt'].reshape(-1))
        tmp_allpix = np.concatenate(tmp_allpix, axis=0)
        self.mean = np.tile(np.mean(tmp_allpix, axis=0),[3])
        self.stddev = np.tile(np.std(tmp_allpix, axis=0),[3])
        self.stddev[self.stddev == 0] = 1e-6

        tmp_allmask = np.concatenate(tmp_allmask, axis=0)
        self.mean_pos = np.mean(tmp_allmask)



    def InitPointer(self):
        self.train_labeled_ptr = 0
        self.train_unlabeled_ptr = 0
        self.val_ptr = 0
        self.test_ptr = 0



    def ShuffleTrainSet(self):
        np.random.shuffle(self.train_labeled_names_active)
        np.random.shuffle(self.train_unlabeled_names_active)



    def ShuffleValSet(self):
        '''
        Shuffle validation set
        :return:
        '''
        np.random.shuffle(self.val_names)



    def InitNewEpoch(self):
        self.ShuffleTrainSet()
        self.InitPointer()


    def NextTrainBatch(self,epoch):
        if self.lab_ratio == 1.:
            FinishEpoch, train_data = self.NextTrainBatch_FullSup()
        else:
            FinishEpoch, train_data = self.NextTrainBatch_SemiSup()
        return FinishEpoch, train_data



    def NextTrainBatch_FullSup(self, epoch):
            '''
            return the next batch training labeled samples only
            :return:
            '''

            ## Initialize All return variables
            train_data = {'labeled': {'data': None, 'gt': None, 'gt_thin': None, 'name': None},
                          'unlabeled': {'data': None, 'gt': None, 'gt_thin': None, 'name': None}}
            FinishEpoch = False

            ## Check reaching the end of dataset
            start_labeled_train = self.train_labeled_ptr  # start index for labeled train set
            end_labeled_train = self.train_labeled_ptr + self.batch_size_train_labeled  # end index for labeled train set
            if start_labeled_train >= self.num_train_labeled:
                # stop when reaching the end of training set
                FinishEpoch = True
                return FinishEpoch, train_data

            if end_labeled_train > self.num_train_labeled:
                # reached the end of dataset
                index_labeled_train = np.arange(start_labeled_train, self.num_train_labeled)
                index_labeled_train = np.concatenate([index_labeled_train,
                                                      np.random.choice(index_labeled_train,
                                                                       end_labeled_train - self.num_train_labeled)],
                                                     axis=0)
            else:
                # continue train with new batch of data
                index_labeled_train = np.arange(start_labeled_train, end_labeled_train)

            ## Slice labeled/unlabeled train sample
            # slice labeled train set
            train_labeled_data = np.stack(
                [self.all_data[self.train_labeled_names_active[i]]['img'] for i in index_labeled_train])
            train_labeled_gt = np.stack(
                [self.all_data[self.train_labeled_names_active[i]]['gt'] for i in index_labeled_train])
            # train_labeled_gt_thin = np.stack(
            #     [self.all_data[self.train_labeled_names_active[i]]['gt_thin'] for i in index_labeled_train])
            train_labeled_name = np.stack(
                [self.all_data[self.train_labeled_names_active[i]]['name'] for i in index_labeled_train])

            # train_labeled_gt = self.all_data['gt'][self.train_labeled_index_active[index_labeled_train]]
            # train_labeled_name = self.all_data['name'][self.train_labeled_index_active[index_labeled_train]]
            # slice unlabeled train set
            if self.batch_size_train_unlabeled > 0:
                index_unlabeled_train = np.arange(self.train_unlabeled_ptr,
                                                  self.train_unlabeled_ptr + self.batch_size_train_unlabeled)
                index_unlabeled_train = np.mod(index_unlabeled_train,
                                               self.num_train_unlabeled)  # recylce the unlabeled data if reached the end of unlabeled data
                train_unlabeled_data = np.stack(
                    [self.all_data[self.train_unlabeled_names_active[i]]['img'] for i in index_unlabeled_train])
                train_unlabeled_gt = np.stack(
                    [self.all_data[self.train_unlabeled_names_active[i]]['gt'] for i in index_unlabeled_train])
                # train_unlabeled_gt_thin = np.stack(
                #     [self.all_data[self.train_unlabeled_names_active[i]]['gt_thin'] for i in index_unlabeled_train])
                train_unlabeled_name = np.stack(
                    [self.all_data[self.train_unlabeled_names_active[i]]['name'] for i in index_unlabeled_train])
            else:
                train_unlabeled_data = None
                train_unlabeled_gt = None
                train_unlabeled_gt_thin = None
                train_unlabeled_name = None

            ## Update data sample pointer
            self.train_labeled_ptr += self.batch_size_train_labeled
            self.train_unlabeled_ptr += self.batch_size_train_unlabeled

            train_data['labeled']['data'] = np.tile(train_labeled_data[...,np.newaxis],[1,1,1,3])
            train_data['labeled']['gt'] = train_labeled_gt
            # train_data['labeled']['gt_thin'] = train_labeled_gt_thin
            train_data['labeled']['name'] = train_labeled_name
            train_data['unlabeled']['data'] = train_unlabeled_data
            train_data['unlabeled']['gt'] = train_unlabeled_gt
            # train_data['unlabeled']['gt_thin'] = train_unlabeled_gt_thin
            train_data['unlabeled']['name'] = train_unlabeled_name

            return FinishEpoch, train_data



    def NextTrainBatch_SemiSup(self):
        train_data = {'labeled':{'data':None,'gt':None,'gt_thin':None,'name':None},
                      'unlabeled':{'data':None,'gt':None,'gt_thin':None,'name':None}}
        FinishEpoch = False

        start_labeled_train = self.train_labeled_ptr
        end_labeled_train = self.train_labeled_ptr + self.batch_size_train_labeled
        index_labeled_train = np.arange(start_labeled_train, end_labeled_train)
        index_labeled_train = np.mod(index_labeled_train, self.num_train_labeled)

        start_unlabeled_train = self.train_unlabeled_ptr
        end_unlabeled_train = self.train_unlabeled_ptr + self.batch_size_train_unlabeled
        index_unlabeled_train = np.arange(start_unlabeled_train, end_unlabeled_train)
        index_unlabeled_train = np.mod(index_unlabeled_train, self.num_train_unlabeled)

        if start_labeled_train >= self.num_train_labeled and start_unlabeled_train >= self.num_train_unlabeled:
            FinishEpoch = True
            return FinishEpoch, train_data

        train_labeled_data = np.stack([self.all_data[self.train_labeled_names_active[i]]['img'] for i in index_labeled_train])
        train_labeled_gt = np.stack([self.all_data[self.train_labeled_names_active[i]]['gt'] for i in index_labeled_train])
        train_labeled_name = np.stack([self.all_data[self.train_labeled_names_active[i]]['name'] for i in index_labeled_train])

        if self.batch_size_train_unlabeled>0:
            train_unlabeled_data = np.stack([self.all_data[self.train_unlabeled_names_active[i]]['img'] for i in index_unlabeled_train])
            train_unlabeled_gt = np.stack([self.all_data[self.train_unlabeled_names_active[i]]['gt'] for i in index_unlabeled_train])
            train_unlabeled_name = np.stack([self.all_data[self.train_unlabeled_names_active[i]]['name'] for i in index_unlabeled_train])
        else:
            train_unlabeled_data = None
            train_unlabeled_gt = None
            train_unlabeled_name = None

        self.train_labeled_ptr += self.batch_size_train_labeled
        self.train_unlabeled_ptr += self.batch_size_train_unlabeled

        train_data['labeled']['data'] = np.tile(train_labeled_data[...,np.newaxis],[1,1,1,3])
        train_data['labeled']['gt'] = train_labeled_gt

        train_data['labeled']['name'] = train_labeled_name
        if train_unlabeled_data is not None:
            train_data['unlabeled']['data'] = np.tile(train_unlabeled_data[...,np.newaxis],[1,1,1,3])
        else:
            train_data['unlabeled']['data'] = None
        train_data['unlabeled']['gt'] = train_unlabeled_gt
        train_data['unlabeled']['name'] = train_unlabeled_name

        return FinishEpoch, train_data



    def NextValBatch(self,epoch):
        val_data = {'data': None, 'gt': None}
        data = None
        gt = None
        FinishEpoch = False

        ## Check reaching the end of dataset
        start_val = self.val_ptr  # start index for labeled train set
        end_val = self.val_ptr + self.batch_size  # end index for labeled train set
        if start_val >= self.num_val:
            # stop when reaching the end of training set
            FinishEpoch = True
            return FinishEpoch, val_data

        if end_val > self.num_val:
            # reached the end of dataset
            index_val = np.arange(start_val, self.num_val)
        else:
            # continue val with new batch of data
            index_val = np.arange(start_val, end_val)

        ## Slice val sample
        data = np.stack([self.all_data[self.val_names[i]]['img'] for i in index_val])
        # avoid float dtype
        if data.dtype == np.float32 and data.max() <= 1 and np.sum((data < 0.9)*(data>0.1)) > 0:
            data *= 255
        gt = np.stack([self.all_data[self.val_names[i]]['gt'] for i in index_val])
        name = np.stack([self.all_data[self.val_names[i]]['name'] for i in index_val])

        ## Update data sample pointer
        self.val_ptr += self.batch_size

        val_data['data'] = np.tile(data[...,np.newaxis],[1,1,1,3])
        val_data['gt'] = gt
        val_data['name'] = name

        return FinishEpoch, val_data



    def NextTestBatch(self):
        ## Initialize All return variables
        test_data = {'data': None, 'gt': None}
        data = None
        gt = None
        FinishEpoch = False

        ## Check reaching the end of dataset
        start_te = self.test_ptr  # start index for labeled train set
        end_te = self.test_ptr + self.batch_size  # end index for labeled train set
        if start_te >= self.num_test:
            # stop when reaching the end of training set
            FinishEpoch = True
            return FinishEpoch, test_data

        if end_te > self.num_test:
            # reached the end of dataset
            index_te = np.arange(start_te, self.num_test)

        else:
            # continue val with new batch of data
            index_te = np.arange(start_te, end_te)

        ## Slice val sample
        data = np.stack([self.all_data[self.val_names[i]]['img'] for i in index_te])
        # avoid float dtype
        if data.dtype == np.float32 and data.max() <= 1 and np.sum((data < 0.9) * (data > 0.1)) > 0:
            data *= 255
        gt = np.stack([self.all_data[self.val_names[i]]['gt'] for i in index_te])
        name = np.stack([self.all_data[self.val_names[i]]['name'] for i in index_te])

        # data = self.all_data['imgs'][self.val_index[index_val]]
        # gt = self.all_data['gt'][self.val_index[index_val]]
        # name = self.all_data['name'][self.val_index[index_val]]

        ## Update data sample pointer
        self.test_ptr += self.batch_size

        test_data['data'] = np.tile(data[...,np.newaxis],[1,1,1,3])
        test_data['gt_org'] = None
        test_data['gt'] = gt
        test_data['name'] = name

        return FinishEpoch, test_data
