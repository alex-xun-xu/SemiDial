## define trainer class and methods
import sys
import pandas as pd
import tqdm
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import datetime
from torch.utils.tensorboard import SummaryWriter
import socket
import pathlib
import parse
import shutil

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '../Network'))

import torch
import unet_model as model
from torch import optim
import loss as Loss
import evaluator as eval
from Augmentation import Augmentor
from GeoTform import GeometricTransform, MockTransform
# from PascalVOC.stream_metrics import StreamSegMetrics

class Trainer():

    def __init__(self, args, Loader, device, n_classes=1):

        self.lr = args.LearningRate    # learning rate
        self.epochs = args.Epoch    # number of training epochs
        self.MaxKeepCkpt = args.MaxKeepCkpt
        self.ValFreq = args.ValFreq   # frequency of validation
        self.target_data = args.TargetData    # target dataset
        self.swap_label = args.SwapLabel    # swap label 1 with 0。
        self.Location = args.Location    # add location as additional feature 1 with 0
        self.SinPeriod = args.SinPeriod    # sinusoid spatial encoding period
        self.Augment = args.Augment    # image augmentation type
        # self.ElasticAlpha = args.ElasticAlpha    # elastic deformation alpha
        # self.ElasticSigma = args.ElasticSigma    # elastic deformation sigma
        self.n_classes= n_classes

        self.Loader = Loader    # Data IO Loader
        self.device = device    # device to use
        self.global_step = 0

        if self.target_data == 'Crack500':
            self.evaluator = eval.evaluate_list()    # evaluator for Crack500 dataset
        else:
            self.evaluator = eval.evaluate()    # evaluator

        if self.target_data == 'PascalVOC':
            self.val_evaluator = StreamSegMetrics(self.n_classes)

        ## Initialize evaluation metrics
        self.val_micro_IoU = -1.
        self.te_micro_IoU = -1.

        self.val_macro_IoU = -1.
        self.te_macro_IoU = -1.

        self.val_Acc = -1.
        self.te_Acc = -1.

        self.val_micro_Dice = -1.
        self.te_micro_Dice = -1.

        self.val_macro_Dice = -1.
        self.te_macro_Dice = -1.

        self.best_te_macro_Dice = -1.
        self.best_te_micro_Dice = -1.


        self.te_macro_F1 = -1.
        self.te_micro_F1 = -1.

        self.best_val_Dice = -1


        self.epoch_cnt = 0

        ## initialize geometric transform
        self.augmentation = args.Augment
        self.AffinePara = {}
        self.ElasticPara = {}
        self.gt_model = self.get_transform()

        # loss_tr、loss_val, macro_dice_val, micro_dice_val, perpix_acc_val, macro_iou_val, micro_iou_val
        self.epoch_result = {
            "epoch":list(range(1,self.epochs+1)),
            "loss_tr":[],
            "loss_val":[],
            "macro_dice_val":[],
            "micro_dice_val":[],
            "perpix_acc_val":[],
            "macro_iou_val":[],
            "micro_iou_val":[]
        }


    def PrintEpochResult(self):
        print("epoch_result",self.epoch_result)


    def SaveEpochResult(self):
        path = os.path.join(self.result_path, 'epoch_result.csv')
        result = pd.DataFrame(data=self.epoch_result)
        result.to_csv(path,index = False)


    def get_transform(self,flag=True):
        if flag == True:
            gt = GeometricTransform()
            if 'Affine' in self.Augment:
                print('【trainer_unet】 Apply Affine Transform')
                gt.add_fliplr(0.5)
                gt.add_flipud(0.5)
                gt.add_translate_x(np.arange(-0.15, 0.15, 0.01))    # in percentage
                gt.add_translate_y(np.arange(-0.15, 0.15, 0.01))
                gt.add_rotate(np.arange(0, 360,1))
                # gt.add_shearx(np.arange(-0.1,0.1,0.01))
                # gt.add_sheary(np.arange(-0.1,0.1,0.01))
            if 'Elastic' in self.Augment:
                print('Apply Elastic Transform')
                self.ElasticPara['alpha'] = self.ElasticAlpha
                self.ElasticPara['sigma'] = self.ElasticSigma
                gt.add_elastic({'alpha':self.ElasticPara['alpha'],'sigma':self.ElasticPara['sigma']})

        else:
            gt = MockTransform()
        return gt



    def DefineNetwork(self, net_name, loss_name):
        if net_name == 'Unet':
            from NewModels.Unet.Unet import U_Net
            if self.Location:
                self.net = U_Net(img_ch=3+2, output_ch=1)
            else:
                self.net = U_Net(img_ch=3, output_ch=1)

        elif net_name == 'SegNet':
            from NewModels.SegNet.segnet import SegNet
            self.net = SegNet(3, 1)

        elif net_name == 'FCN':
            from NewModels.FCN.FCN8s import FCN
            self.net = FCN(3, 1)

        elif net_name == 'ResUnet':
            if self.Location:
                self.net = model.ResUNet(n_channels=3 + 2, n_classes=1, bilinear=False)
            else:
                self.net = model.ResUNet(n_channels=3, n_classes=1, bilinear=False)

        elif net_name == 'BCDUNet':
            from NewModels.BCDUnet.BCDUNet import BCDUNet
            if self.Location:
                self.net = BCDUNet(input_dim=3 + 2, output_dim=1)
            else:
                self.net = BCDUNet(input_dim=3, output_dim=1)

        elif net_name == 'Attention_Unet':
            from model.Attention_Unet import AttU_Net
            self.net = AttU_Net(3, 1)

        elif net_name == 'Unetplus':
            from model.Unetplus import unetpluses
            self.net = unetpluses(3, 1)

        elif net_name == 'HED':
            if self.Location:
                self.net = model.HED(n_channels=3 + 2)
            else:
                self.net = model.HED(n_channels=3)
        elif net_name == 'RCF':
            if self.Location:
                self.net = model.RCF(n_channels=3 + 2)
            else:
                self.net = model.RCF(n_channels=3)

        elif net_name == 'DeepLabV3Plus':
            if self.Location:
                self.net = model.DeepLabV3Plus_Exception(n_channels=3 + 2)
            else:
                self.net = model.DeepLabV3Plus_Exception(n_channels=3, n_classes=self.n_classes)

        elif net_name == 'LK34':
            if self.Location:
                self.net = model.LinkNet34(in_channels=3+2, num_classes=1)
            else:
                self.net = model.LinkNet34(in_channels=3, num_classes=1)

        elif net_name == 'LK34MTL':
            if self.Location:
                self.net = model.LinkNet34MTL(task1_classes=1, task2_classes=37)
            else:
                self.net = model.LinkNet34MTL(task1_classes=1, task2_classes=37)

        self.net.to(device=self.device)


        if loss_name == 'BCEloss':
            self.criterion = Loss.BCELoss()
        elif loss_name == 'CrossEntloss':
            self.criterion = Loss.CrossEntLoss()
        elif loss_name == 'WeightedBCEloss':
            self.criterion = Loss.WeightedBCELoss()
        elif loss_name == 'Diceloss':
            self.criterion = Loss.DiceLoss(logit=True)
        elif loss_name == 'Diceloss2':
            self.criterion = Loss.DiceLoss2(logit=True)
        elif loss_name == 'IoUloss':
            self.criterion = Loss.IoULoss(logit=True)
        elif loss_name == 'Dice+SupTopo_loss':
            self.criterion = Loss.DiceSupTopoLoss(logit=True, imgsize=self.masksize)
        elif loss_name == 'Dice+BCE_loss':
            self.criterion = Loss.DiceBCELoss(logit=True)
        elif loss_name == 'Dice+ConsistMSE_loss':
            self.criterion = Loss.Dice_ConsistMSELoss(logit=True)
        elif loss_name == 'WeightedBCEloss+ConsistMSE_loss':
            self.criterion = Loss.WBCE_ConsistMSELoss(logit=True)
        pass


    def DefineOptimizer(self):
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=0)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min' if self.net.n_classes > 1 else 'max', patience=2)

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=.95)
        pass



    def DefineAugmentation(self):
        self.augmentor = Augmentor(config=3)



    def Initialize(self):
        '''
        Dummy function to initialize from existing models
        :return:
        '''
        pass


    def ApplyNormalization(self, data):
        '''
        Apply normalization to batch data
        :param data:
        :return:
        '''
        return (data - self.Loader.mean) / self.Loader.stddev



    def ApplyAugmentation(self, data, gt):
        data, gt = self.augmentor.augment(images=data.astype(np.float32), masks=gt.astype(np.float32))

        # geometric transformation augmentation
        self.gt_model.construct_random_transform(data)
        data = self.gt_model.transform_images(data, extrapolation='reflect')
        gt = self.gt_model.transform_images(gt[..., np.newaxis], extrapolation='reflect')[..., 0]

        ## Apply Normalization
        data = (data - self.Loader.mean)/self.Loader.stddev

        ## new dimension order
        train_labeled_data = np.transpose(data, [0, 3, 1, 2])
        train_labeled_gt = np.transpose(gt, [0, 1, 2])

        return train_labeled_data, train_labeled_gt



    def TrainOneEpoch(self, max_itr=np.inf,epoch=0):
        start_time = datetime.datetime.now()

        epoch_loss = 0.

        train_labeled_pred_all = []
        train_labeled_gt_all = []

        itr = 0

        self.net.train()

        while True:
            FinishEpoch, data = self.Loader.NextTrainBatch_FullSup(epoch)
            if FinishEpoch or itr > max_itr:
                break

            train_labeled_data = data['labeled']['data']
            train_labeled_gt = data['labeled']['gt']

            train_labeled_data, train_labeled_gt = self.ApplyAugmentation(train_labeled_data, train_labeled_gt)

            train_labeled_data = torch.tensor(train_labeled_data,device=self.device, dtype=torch.float32)
            train_labeled_gt = torch.tensor(train_labeled_gt,device=self.device, dtype=torch.float32)

            if self.Location:
                x, y = np.meshgrid(np.arange(0, train_labeled_data.shape[3]),np.arange(0, train_labeled_data.shape[2]))
                x = x[np.newaxis, np.newaxis, ...].astype(np.float32)/x.max()
                y = y[np.newaxis, np.newaxis, ...].astype(np.float32)/y.max()
                x = torch.tensor(np.tile(x, [train_labeled_data.shape[0], 1, 1, 1]), device=train_labeled_data.device)
                y = torch.tensor(np.tile(y, [train_labeled_data.shape[0], 1, 1, 1]), device=train_labeled_data.device)
                train_labeled_data = torch.cat([train_labeled_data, x, y], dim=1)


            train_labeled_pred = self.net(train_labeled_data)

            if train_labeled_pred.shape[1] > 1:
                loss = self.criterion(pred=train_labeled_pred, gt=train_labeled_gt)
            else:
                loss = self.criterion(pred=train_labeled_pred[:,0,...], gt=train_labeled_gt)
            epoch_loss = epoch_loss * itr/(itr+1) + float(loss.item())/(itr+1)

            train_labeled_pred_all.append(train_labeled_pred.detach().cpu().numpy())
            train_labeled_gt_all.append(train_labeled_gt.detach().cpu().numpy())

            del train_labeled_pred, train_labeled_gt
            torch.cuda.empty_cache()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(self.net.parameters(), 0.1)
            self.optimizer.step()

            if self.global_step % 10 == 0:
                self.lr_scheduler.step()

            self.global_step += 1
            itr += 1

        self.tr_loss = epoch_loss
        self.epoch_result["loss_tr"].append(epoch_loss)

        # epoch计数器加一
        self.epoch_cnt += 1

        return epoch_loss



    def EvalTrainSet(self, max_itr=np.inf):
        '''
        Evaluate performance on train set only
        :param max_itr:
        :return:
        '''
        # with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        ## Shuffle Train Set
        # self.Loader.ShuffleTrainSet()
        # self.Loader.InitPointer()
        epoch_loss = 0.

        train_labeled_pred_all = []
        train_labeled_gt_all = []

        itr = 0

        # enable network for evaluation
        self.net.eval()

        while True:
            ## Get next batch train samples
            FinishEpoch, data = \
                self.Loader.NextTrainBatch_FullSup()
            if FinishEpoch:
                break

            train_labeled_data = data['labeled']['data']
            train_labeled_gt = data['labeled']['gt']
            train_labeled_name = data['labeled']['name']
            train_unlabeled_data = data['unlabeled']['data']
            train_unlabeled_gt = data['unlabeled']['gt']
            train_unlabeled_name = data['unlabeled']['name']

            ## Apply Augmentation
            # train_labeled_data, train_labeled_gt = self.ApplyAugmentation(train_labeled_data, train_labeled_gt)
            train_labeled_data = train_labeled_data.transpose([0,3,1,2])

            ## train one iteration
            train_labeled_data = torch.tensor(train_labeled_data,device=self.device, dtype=torch.float32)
            train_labeled_gt = torch.tensor(train_labeled_gt,device=self.device, dtype=torch.float32)

            # Augment with location
            if self.Location:
                x, y = np.meshgrid(np.arange(0, train_labeled_data.shape[3]),
                                   np.arange(0, train_labeled_data.shape[2]))
                x = x[np.newaxis, np.newaxis, ...].astype(np.float32)/x.max()
                y = y[np.newaxis, np.newaxis, ...].astype(np.float32)/y.max()
                x = torch.tensor(np.tile(x, [train_labeled_data.shape[0], 1, 1, 1]), device=train_labeled_data.device)
                y = torch.tensor(np.tile(y, [train_labeled_data.shape[0], 1, 1, 1]), device=train_labeled_data.device)
                train_labeled_data = torch.cat([train_labeled_data, x, y], dim=1)

            # forward pass
            train_labeled_pred = self.net(train_labeled_data)   # forward pass prediction of train labeled set
            loss = self.criterion(pred=train_labeled_pred[:,0,...], gt=train_labeled_gt)
            epoch_loss = epoch_loss * itr/(itr+1) + float(loss.item())/(itr+1)

            # accumulate predictions and ground-truths
            train_labeled_pred_all.append(train_labeled_pred.detach().cpu().numpy())
            train_labeled_gt_all.append(train_labeled_gt.detach().cpu().numpy())

            # empty cached memory
            del train_labeled_pred, train_labeled_gt
            torch.cuda.empty_cache()

            # update global and local train steps
            self.global_step += 1
            itr += 1

        ## Evaluate current epoch performance
        train_labeled_pred_all = np.concatenate(train_labeled_pred_all)
        train_labeled_pred_all = np.squeeze(train_labeled_pred_all>0,axis=1).astype(float)  # binarize predictions
        train_labeled_gt_all = np.concatenate(train_labeled_gt_all).astype(float)
        perpix_acc = self.evaluator.perpixel_acc(train_labeled_pred_all, train_labeled_gt_all)  # evaluate per pixel accuracy
        persamp_iou = self.evaluator.persamp_iou(train_labeled_pred_all, train_labeled_gt_all)  # evaluate per sample iou
        micro_iou = self.evaluator.micro_iou(train_labeled_pred_all, train_labeled_gt_all)  # evaluate per sample iou

        self.tr_loss = epoch_loss
        self.tr_Acc = perpix_acc
        self.tr_macro_IoU = persamp_iou
        self.tr_micro_IoU = micro_iou

        self.epoch_cnt += 1 # increase internal epoch counter by 1

        return epoch_loss, perpix_acc, persamp_iou, micro_iou



    def ValOneEpoch_CurveLinear(self,epoch=0):
        epoch_loss = 0.

        val_pred_all = []
        val_gt_all = []

        itr = 0

        # enable network for evaluation
        self.net.eval()

        while True:
            ## Get next batch val samples
            FinishEpoch, data = self.Loader.NextValBatch(epoch)
            if FinishEpoch:
                break

            ## Apply Normalization
            if data['data'].dtype == np.float32 and data['data'].max()<=1.0:
                val_data = self.ApplyNormalization(255*data['data'])
            else:   # <-
                val_data = self.ApplyNormalization(data['data'])

            val_data = np.transpose(val_data, [0, 3, 1, 2])
            val_gt = np.transpose(data['gt'], [0, 1, 2])

            ## val current batch
            val_data = torch.tensor(val_data, device=self.device, dtype=torch.float32)
            val_gt = torch.tensor(val_gt, device=self.device, dtype=torch.float32)

            # Augment with location
            if self.Location:
                x, y = np.meshgrid(np.arange(0, val_data.shape[3]),np.arange(0, val_data.shape[2]))
                x = x[np.newaxis, np.newaxis, ...].astype(np.float32)/x.max()
                y = y[np.newaxis, np.newaxis, ...].astype(np.float32)/y.max()
                x = torch.tensor(np.tile(x, [val_data.shape[0], 1, 1, 1]), device=val_data.device)
                y = torch.tensor(np.tile(y, [val_data.shape[0], 1, 1, 1]), device=val_data.device)
                val_data = torch.cat([val_data, x, y], dim=1)

            # forward pass
            val_pred = self.net(val_data)  # forward pass prediction of train labeled set
            if val_pred.shape[1]>1:
                loss = self.criterion(pred=val_pred, gt=val_gt)
            else:
                loss = self.criterion(pred=val_pred[:,0,...], gt=val_gt)

            epoch_loss = epoch_loss*itr/(itr+1) + loss.item()/(itr+1)

            # accumulate predictions and ground-truths
            val_pred_all.append(val_pred.detach().cpu().numpy())
            val_gt_all.append(val_gt.detach().cpu().numpy())

            del val_pred, val_gt
            torch.cuda.empty_cache()

            itr += 1

        ## Evaluate val performance
        val_pred_all = np.concatenate(val_pred_all)

        if self.n_classes == 1:   # ←
            val_pred_all = np.squeeze(val_pred_all > 0, axis=1).astype(float)  # binarize predictions
            val_gt_all = np.concatenate(val_gt_all).astype(float)

            persamp_dice = self.evaluator.persamp_dice(val_pred_all, val_gt_all)   # persamp_dice
            micro_dice = self.evaluator.micro_dice(val_pred_all, val_gt_all)   # micro_dice
            perpix_acc = self.evaluator.perpixel_acc(val_pred_all, val_gt_all)  # evaluate per pixel accuracy
            persamp_iou = self.evaluator.persamp_iou(val_pred_all, val_gt_all)  # evaluate per sample iou
            micro_iou = self.evaluator.micro_iou(val_pred_all, val_gt_all)  # evaluate per sample iou
        else:
            val_pred_all = np.argmax(val_pred_all, axis=1)
            val_gt_all = np.concatenate(val_gt_all).astype(float)
            self.val_evaluator.reset()
            self.val_evaluator.update(val_gt_all, val_pred_all)
            score = self.val_evaluator.get_results()
            perpix_acc = score['Mean Acc']
            micro_iou = score['Mean IoU']
            persamp_iou = np.array([score['Class IoU'][key] for key in score['Class IoU']])

        self.val_macro_Dice = persamp_dice
        self.val_micro_Dice = micro_dice

        self.val_loss = epoch_loss
        self.val_Acc = perpix_acc
        self.val_macro_IoU = persamp_iou
        self.val_micro_IoU = micro_iou

        self.epoch_result["loss_val"].append(epoch_loss)
        self.epoch_result["macro_dice_val"].append(100 * np.mean(persamp_dice))
        self.epoch_result["micro_dice_val"].append(100 * micro_dice)
        self.epoch_result["perpix_acc_val"].append(100 * perpix_acc)
        self.epoch_result["macro_iou_val"].append(100 * np.mean(persamp_iou))
        self.epoch_result["micro_iou_val"].append(100 * micro_iou)

        return epoch_loss, persamp_dice, micro_dice, perpix_acc, persamp_iou, micro_iou



    def ValOneEpoch_PascalVOC(self):
        '''
        Evaluate one epoch on validation set
        :return:
        '''

        epoch_loss = 0.

        val_pred_all = []
        val_gt_all = []

        itr = 0

        # enable network for evaluation
        self.net.eval()
        self.val_evaluator.reset()

        while True:
            ## Get next batch val samples
            FinishEpoch, data = \
                self.Loader.NextValBatch()
            if FinishEpoch:
                break

            ## Apply Normalization
            # val_data = self.ApplyNormalization(data['data'])
            val_data = data['data']

            # val_data = np.transpose(val_data, [0, 3, 1, 2])
            val_gt = np.transpose(data['gt'], [0, 1, 2])

            ## val current batch
            val_data = torch.tensor(val_data, device=self.device, dtype=torch.float32)
            val_gt = torch.tensor(val_gt, device=self.device, dtype=torch.float32)

            # Augment with location
            if self.Location:
                x, y = np.meshgrid(np.arange(0, val_data.shape[3]),
                                   np.arange(0, val_data.shape[2]))
                x = x[np.newaxis, np.newaxis, ...].astype(np.float32)/x.max()
                y = y[np.newaxis, np.newaxis, ...].astype(np.float32)/y.max()
                x = torch.tensor(np.tile(x, [val_data.shape[0], 1, 1, 1]), device=val_data.device)
                y = torch.tensor(np.tile(y, [val_data.shape[0], 1, 1, 1]), device=val_data.device)
                val_data = torch.cat([val_data, x, y], dim=1)

            # forward pass
            val_pred = self.net(val_data)  # forward pass prediction of train labeled set
            if val_pred.shape[1]>1:
                loss = self.criterion(pred=val_pred, gt=val_gt)
            else:
                loss = self.criterion(pred=val_pred[:,0,...], gt=val_gt)
            epoch_loss = epoch_loss*itr/(itr+1) + loss.item()/(itr+1)

            # accumulate predictions and ground-truths
            # val_pred_all.append(val_pred.detach().cpu().numpy())
            # val_gt_all.append(val_gt.detach().cpu().numpy())

            self.val_evaluator.update(val_gt.detach().cpu().numpy(), val_pred.detach().cpu().numpy().argmax(axis=1))


            del val_pred, val_gt
            torch.cuda.empty_cache()


            itr += 1


        ## Evaluate val performance
        # val_pred_all = np.concatenate(val_pred_all)

        score = self.val_evaluator.get_results()
        perpix_acc = score['Mean Acc']
        micro_iou = score['Mean IoU']
        persamp_iou = [score['Class IoU'][key] for key in score['Class IoU']]

        self.val_loss = epoch_loss
        self.val_Acc = perpix_acc
        self.val_macro_IoU = persamp_iou
        self.val_micro_IoU = micro_iou

        return epoch_loss, perpix_acc, persamp_iou, micro_iou



    def ValOneEpoch(self,epoch=0):
        if self.target_data == 'PascalVOC':
            return self.ValOneEpoch_PascalVOC()
        else:
            return self.ValOneEpoch_CurveLinear(epoch)



    def TestAll_SavePred(self, exp_fig=False, best_ckpt_filepath=None):

        epoch_loss = 0.
        itr = 0

        test_pred_all = []
        test_gt_all = []
        test_gt_org_all = []

        if best_ckpt_filepath is not None:
            self.net.load_state_dict(torch.load(best_ckpt_filepath,map_location=self.device))

        self.net.eval()

        while True:
            FinishEpoch, data = self.Loader.NextTestBatch()
            if FinishEpoch:
                break

            test_data = data['data']
            test_gt = data['gt']
            test_gt_org = data['gt_org']
            test_names = data['name']

            test_data = self.ApplyNormalization(test_data)

            test_data = np.transpose(test_data, [0, 3, 1, 2])
            test_gt = np.transpose(test_gt, [0, 1, 2])

            test_data = torch.tensor(test_data, device=self.device, dtype=torch.float32)
            test_gt = torch.tensor(test_gt, device=self.device, dtype=torch.float32)

            if self.Location:
                x, y = np.meshgrid(np.arange(0, test_data.shape[3]),np.arange(0, test_data.shape[2]))
                x = x[np.newaxis, np.newaxis, ...].astype(np.float32)/x.max()
                y = y[np.newaxis, np.newaxis, ...].astype(np.float32)/y.max()
                x = torch.tensor(np.tile(x, [test_data.shape[0], 1, 1, 1]), device=test_data.device)
                y = torch.tensor(np.tile(y, [test_data.shape[0], 1, 1, 1]), device=test_data.device)
                test_data = torch.cat([test_data, x, y], dim=1)

            test_pred = self.net(test_data)

            loss = self.criterion(pred=test_pred[:, 0, ...], gt=test_gt)
            epoch_loss = epoch_loss*itr/(itr+1) + loss.item()/(itr+1)

            test_pred = torch.sigmoid(test_pred)
            for pred_i in test_pred.detach().cpu().numpy():
                test_pred_all.append(pred_i[0])
            for gt_i in test_gt.detach().cpu().numpy():
                test_gt_all.append(gt_i)

            if test_gt_org is not None:
                for gt_i in test_gt_org:
                    test_gt_org_all.append(gt_i)

            if exp_fig:
                for img_i, gt_i, pred_i in zip(test_names, test_gt, test_pred):
                    pred_path = os.path.join(self.export_path, 'pred')
                    if not os.path.exists(pred_path):
                        os.makedirs(pred_path)
                    exp_pred_filepath = os.path.join(pred_path, '{}_pred.png'.format(img_i))
                    img = Image.fromarray(255 * pred_i.detach().cpu().numpy()[0]).convert('RGB')
                    img.save(exp_pred_filepath)
                    exp_pred_filepath = os.path.join(pred_path,'{}_pred_bin.png'.format(img_i))
                    img = Image.fromarray(255.*(pred_i.detach().cpu().numpy()[0] > 0.5)).convert('RGB')
                    img.save(exp_pred_filepath)

            # increase local iteration
            itr += 1

        if self.target_data != 'Crack500':

            test_pred_all = np.array(test_pred_all)
            test_pred_bin_all = (test_pred_all > 0.5).astype(float)
            test_gt_all = np.array(test_gt_all).astype(float)

            persamp_dice = self.evaluator.persamp_dice(test_pred_bin_all,test_gt_all)
            micro_dice = self.evaluator.micro_dice(test_pred_bin_all, test_gt_all)
            perpix_acc = self.evaluator.perpixel_acc(test_pred_bin_all,test_gt_all)  # etestuate per pixel accuracy
            persamp_iou = self.evaluator.persamp_iou(test_pred_bin_all,test_gt_all)  # evaluate per sample iou
            micro_iou = self.evaluator.micro_iou(test_pred_bin_all, test_gt_all)  # evaluate micro-average iou
            AIU = self.evaluator.AIU(test_pred_all,test_gt_all)   # evaluate AIU

            # micro_F1, macro_F1 = self.evaluator.F1(test_pred_bin_all, test_gt_all)
            micro_F1 = 0
            macro_F1 = 0

        elif self.target_data == 'Crack500':
            perpix_acc, persamp_iou, micro_iou, AIU, micro_F1, macro_F1 = \
                self.CalMetric_Crack500(test_pred_all, test_gt_org_all)

        self.best_te_loss = epoch_loss
        self.best_te_macro_Dice = persamp_dice
        self.best_te_micro_Dice = micro_dice
        self.best_te_Acc = perpix_acc
        self.best_te_macro_IoU = persamp_iou
        self.best_te_micro_IoU = micro_iou
        self.best_te_AIU = AIU
        self.best_te_micro_F1 = micro_F1
        self.best_te_macro_F1 = macro_F1

        return epoch_loss, persamp_dice, micro_dice, perpix_acc, persamp_iou, micro_iou, AIU



    def CalMetric_Crack500(self, pred, gt):
        '''
        Test all samples in the test set for Crack500 dataset only
        :return:
        '''

        ## Evaluate test performance
        # test_pred_all = np.squeeze(np.concatenate(test_pred_all),axis=1)
        # test_pred_bin_all = (test_pred_all > 0.5).astype(float)  # binarize predictions
        # test_gt_all = np.concatenate(test_gt_all).astype(float)
        # test_gt_thin_all = np.concatenate(test_gt_thin_all).astype(float)
        perpix_acc = self.evaluator.perpixel_acc(pred,
                                                 gt)  # etestuate per pixel accuracy
        persamp_iou = self.evaluator.persamp_iou(pred,
                                                 gt)  # evaluate per sample iou
        micro_iou = self.evaluator.micro_iou(pred, gt)  # evaluate micro-average iou
        # ods = self.evaluator.ODS(test_pred_all,test_gt_thin_all)   # evaluate ODS
        # AIU = self.evaluator.AIU(pred,gt)   # evaluate AIU
        AIU=-1
        micro_F1, macro_F1 = self.evaluator.F1(pred, gt)

        # self.best_te_Acc = perpix_acc
        # self.best_te_macro_IoU = persamp_iou
        # self.best_te_micro_IoU = micro_iou
        # self.best_te_AIU = AIU

        return perpix_acc, persamp_iou, micro_iou, AIU, micro_F1, macro_F1



    def UpdateBest(self):
        self.best_val_Acc = self.val_Acc
        self.best_val_macro_IoU = self.val_macro_IoU
        self.best_val_micro_IoU = self.val_micro_IoU
        self.best_val_macro_Dice = self.val_macro_Dice
        self.best_val_micro_Dice = self.val_micro_Dice




    def ExportResults(self,result_filepath=None):
        if result_filepath is not None:
            fid = open(result_filepath, 'w')
        else:
            fid = open(os.path.join(self.result_path,'results.txt'),'w')
        fid.write('Val:\n')
        fid.write('Macro Average  Dice: {:.2f}%\n'.format(100 * np.mean(self.best_val_macro_Dice)))
        fid.write('Micro Average  Dice: {:.2f}%\n'.format(100 * np.mean(self.best_val_micro_Dice)))
        fid.write('Acc: {:.2f}%\n'.format(100*np.mean(self.best_val_Acc)))
        fid.write('Macro Average  IoU: {:.2f}%\n'.format(100 * np.mean(self.best_val_macro_IoU)))
        fid.write('Micro Average  IoU: {:.2f}%\n'.format(100 * np.mean(self.best_val_micro_IoU)))
        fid.write('Test:\n')
        fid.write('Macro Average  Dice: {:.2f}%\n'.format(100 * np.mean(self.best_te_macro_Dice)))
        fid.write('Micro Average  Dice: {:.2f}%\n'.format(100 * np.mean(self.best_te_micro_Dice)))
        fid.write('Acc: {:.2f}%\n'.format(100*np.mean(self.best_te_Acc)))
        fid.write('Macro Average  IoU: {:.2f}%\n'.format(100*np.mean(self.best_te_macro_IoU)))
        fid.write('Micro Average  IoU: {:.2f}%\n'.format(100*np.mean(self.best_te_micro_IoU)))
        fid.write('AIU: {:.2f}%\n'.format(100*self.best_te_AIU))
        fid.write('Macro Average  F1: {:.2f}%\n'.format(100*np.mean(self.best_te_macro_F1)))
        fid.write('Micro Average  F1: {:.2f}%\n'.format(100*np.mean(self.best_te_micro_F1)))
        fid.close()



    def PrintTrValInfo(self):
        print("【train_unet】 ======================================================")
        print('【train_unet】 Epoch: {}'.format(self.epoch_cnt))
        print('【train_unet】 loss/train: {:.2f}   val: {:.2f}'.format(self.tr_loss, self.val_loss))
        print('【train_unet】 macro average Dice/val:{:.2f}%'.format(100 * np.mean(self.val_macro_Dice)))
        print('【train_unet】 micro average Dice/val:{:.2f}%'.format(100 * self.val_micro_Dice))
        print('【train_unet】 perpix_accuracy/val: {:.2f}%'.format(100 * self.val_Acc))
        print('【train_unet】 macro average IoU/val:{:.2f}%'.format(100 * np.mean(self.val_macro_IoU)))
        print('【train_unet】 micro average IoU/val:{:.2f}%'.format(100 * self.val_micro_IoU))
        print("【train_unet】 ======================================================")


    def PrintTrInfo(self):
        print("【train_unet】 ======================================================")
        print('【train_unet】 Epoch: {}'.format(self.epoch_cnt))
        print('【train_unet】 loss/train: {:.2f}'.format(self.tr_loss))
        print("【train_unet】 ======================================================")


    def PrintTeInfo(self):
        print("【train_unet】 ======================================================")
        print('【train_unet】 Inference')
        print('【train_unet】 loss/test: {:.2f}'.format(self.best_te_loss))
        print('【train_unet】 Macro Average Dice/test: {:.2f}%'.format(100 * np.mean(self.best_te_macro_Dice)))
        print('【train_unet】 Micro Average Dice/test: {:.2f}%'.format(100 * self.best_te_micro_Dice))
        print('【train_unet】 perpix_accuracy/test: {:.2f}%'.format(100 * self.best_te_Acc))
        print('【train_unet】 Macro Average IoU/test: {:.2f}%'.format(100 * np.mean(self.best_te_macro_IoU)))
        print('【train_unet】 Micro Average IoU/test: {:.2f}%'.format(100 * self.best_te_micro_IoU))
        print("【train_unet】 ======================================================")


    def ExportTensorboard(self):
        self.writer.add_scalar('loss/train', self.tr_loss, self.epoch_cnt)
        self.writer.add_scalar('loss/val', self.val_loss, self.epoch_cnt)
        self.writer.add_scalar('macro average Dice/val', np.mean(self.val_macro_Dice), self.epoch_cnt)
        self.writer.add_scalar('micro average Dice/val', np.mean(self.val_micro_Dice), self.epoch_cnt)
        self.writer.add_scalar('perpix_accuracy/val', 100 * self.val_Acc, self.epoch_cnt)
        self.writer.add_scalar('macro average IoU/val', np.mean(self.val_macro_IoU), self.epoch_cnt)
        self.writer.add_scalar('micro average IoU/val', np.mean(self.val_micro_IoU), self.epoch_cnt)



    def PrepareSaveResults(self, base_path, args):
        result_base_path = os.path.join(base_path, 'Results', args.TargetData, args.ssl)
        export_path = None
        best_ckpt = None
        writer = None

        if not os.path.exists(result_base_path):
            os.makedirs(result_base_path)

        if args.SaveRslt:
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.result_path = os.path.join(result_base_path, '{}_{}_{}_ep-{}_m-{}_{}'.
                                       format(args.ssl, args.net, args.loss, args.Epoch, args.labelpercent,time))
            os.makedirs(self.result_path)

            summary_path = os.path.join(self.result_path, 'summary')
            os.makedirs(summary_path)
            self.writer = SummaryWriter(log_dir=summary_path)

            self.export_path = os.path.join(self.result_path, 'figures')
            os.makedirs(self.export_path)

            if args.TargetData == 'Crack500':
                os.makedirs(os.path.join(self.export_path, 'gt', 'testcrop'))
                os.makedirs(os.path.join(self.export_path, 'pred', 'testcrop'))

            self.ckpt_path = os.path.join(self.result_path, 'ckpt')
            os.makedirs(self.ckpt_path)



    def SetRsltPath(self, rslt_path):
        '''
        Set the result path
        :param base_path:
        :param args:
        :return:
        '''
        result_base_path = rslt_path
        export_path = None
        best_ckpt = None

        self.result_path = rslt_path
        # tensorboard summary file
        summary_path = os.path.join(self.result_path, 'summary')
        # prediction figures
        self.export_path = os.path.join(self.result_path, 'figures')
        self.ckpt_path = os.path.join(self.result_path, 'ckpt')
        self.aug_path = os.path.join(self.result_path, 'augment')

        ## Copy augmentation files
        shutil.copy('../Trainer/trainer_unet.py',self.aug_path)
        shutil.copy('../Network/Augmentation.py', self.aug_path)
        shutil.copy('../Network/GeoTform.py', self.aug_path)



    def RestoreModelByPath(self, ckpt_path=None, model_epoch='best'):
        if ckpt_path is None:
            load_path = os.path.join(self.ckpt_path, 'model_epoch-{}.pt'.format(model_epoch))
        else:
            load_path = os.path.join(ckpt_path, 'model_epoch-{}.pt'.format(model_epoch))

        state_dict = torch.load(load_path, map_location=str(self.device))
        net = getattr(self, 'net')
        net.load_state_dict(state_dict)



    def SaveAllSettings(self, args):
        if args.SaveRslt:
            self.settings_filepath = os.path.join(self.result_path,'settings.txt')

            with open(self.settings_filepath,'w') as fid:
                fid.write('Host:{}\n'.format(socket.gethostname()))
                fid.write('GPU:{}\n'.format(args.GPU))
                fid.write('Network:{}\n'.format(args.net))
                fid.write('LearningRate:{}\n'.format(args.LearningRate))
                fid.write('Epoch:{}\n'.format(args.Epoch))
                fid.write('batchsize:{}\n'.format(args.batchsize))
                fid.write('labelpercent:{}\n'.format(args.labelpercent))
                fid.write('ssl:{}\n'.format(args.ssl))
                fid.write('Gamma:{}\n'.format(args.Gamma))
                fid.write('RampupEpoch:{}\n'.format(args.RampupEpoch))
                fid.write('Target Dataset:{}\n'.format(args.TargetData))
                fid.write('Aux Dataset:{}\n'.format(args.AddUnlab))
                fid.write('AddLocation:{}\n'.format(args.Location))
                fid.write('Augment:{}\n'.format(args.Augment))
                if 'Elastic' in self.Augment:
                    fid.write('Elastic Alpha:{}\n'.format(self.ElasticPara['alpha']))
                    fid.write('Elastic Sigma:{}\n'.format(self.ElasticPara['sigma']))



    def UpdateLatestModel(self):
        todelete_ckpt = os.path.join(self.ckpt_path, 'model_epoch-{}.pt'.format(self.epoch_cnt - self.ValFreq * self.MaxKeepCkpt))
        if os.path.exists(todelete_ckpt):
            os.remove(todelete_ckpt)

        current_ckpt = os.path.join(self.ckpt_path, 'model_epoch-{}.pt'.format(self.epoch_cnt))
        torch.save(self.net.state_dict(), current_ckpt)

        if self.best_val_Dice < self.val_micro_Dice:
            self.best_val_Dice = self.val_micro_Dice
            best_ckpt = os.path.join(self.ckpt_path, 'model_epoch-{}.pt'.format('best'))
            os.system('cp {} {}'.format(current_ckpt, best_ckpt))
            torch.save(self.net.state_dict(), best_ckpt)

            self.UpdateBest()
            print('【train_unet】 saved current epoch as the best up-to-date model!')
