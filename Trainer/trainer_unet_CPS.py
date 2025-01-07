## define trainer class and methods
import sys
import tqdm
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import copy
import pathlib
import time
from time import *
from time import time
from PIL import Image

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(),'../Network'))

import torch
import unet_model as model
from torch import optim
import loss as Loss
import evaluator as eval
from Augmentation import Augmentor
from geometric_transform import GeometricTransform, MockTransform
from trainer_unet import Trainer as UnetTrainer
import datetime
import loss as Loss



class Trainer(UnetTrainer):

    def __init__(self, args, Loader, device):

        super(Trainer, self).__init__(args, Loader, device)
        self.rampup_epoch = args.RampupEpoch    # rampup epoch
        self.rampup_type = args.RampupType  # rampup type
        self.alpha = args.Alpha  # ema moving average weight
        self.W = self.WeightRampup(self.rampup_type, self.rampup_epoch)  # rampup weight

        self.criterion_bce = nn.BCELoss()

        self.val_micro_Dice = -1.
        self.te_micro_Dice = -1.

        self.val_macro_Dice = -1.
        self.te_macro_Dice = -1.

        self.best_val_Dice = -1

        self.best_te_macro_Dice = -1.
        self.best_te_micro_Dice = -1.

        # loss_tr、loss_val, macro_dice_val, micro_dice_val, perpix_acc_val, macro_iou_val, micro_iou_val
        self.epoch_result = {
            "epoch": list(range(1, self.epochs + 1)),
            "loss_tr": [],
            "loss_val": [],
            "macro_dice_val": [],
            "micro_dice_val": [],
            "perpix_acc_val": [],
            "macro_iou_val": [],
            "micro_iou_val": []
        }


    def DefineNetwork(self, net_name, loss_name):
        from NewModels.BCDUnet.BCDUNet import BCDUNet
        from NewModels.BCDUnet.BCDUNet2 import BCDUNet2
        self.net = BCDUNet(3, 1)
        self.net_r = BCDUNet2(3, 1)

        self.net.to(device=self.device)
        self.net_r.to(device=self.device)


    def DefineOptimizer(self):
        self.optimizer_l = torch.optim.SGD(self.net.parameters(),lr=0.01,momentum=0.9)
        self.optimizer_r = torch.optim.SGD(self.net_r.parameters(),lr=0.01,momentum=0.9)


    def TrainOneEpoch(self, writer=None, max_itr=np.inf, epoch=0):
        epoch_loss = 0.
        epoch_loss_cps = 0.
        epoch_loss_sup_l = 0.
        epoch_loss_sup_r = 0.

        self.net.train()
        self.net_r.train()

        itr = 0

        while True:
            FinishEpoch, data = self.Loader.NextTrainBatch(epoch=epoch)

            if FinishEpoch or itr > max_itr:
                break

            self.optimizer_l.zero_grad()
            self.optimizer_r.zero_grad()

            train_labeled_data = data['labeled']['data']
            train_labeled_gt = data['labeled']['gt']
            train_unlabeled_data = data['unlabeled']['data']
            train_unlabeled_gt = data['unlabeled']['gt']

            ## Update weight for consistency loss
            w = self.W(epoch=float(self.epoch_cnt))
            self.weight = torch.tensor(w, device=self.device)

            train_labeled_data_view1, train_labeled_gt_view1, train_labeled_mask_view1, train_labeled_Tform_view1 = self.ApplyAugmentation_Mask(train_labeled_data, train_labeled_gt)

            train_unlabeled_data_view1, train_unlabeled_gt_view1, train_unlabeled_mask_view1, train_unlabeled_Tform_view1 = self.ApplyAugmentation_Mask(train_unlabeled_data, train_unlabeled_gt)
            train_unlabeled_data_view2, train_unlabeled_gt_view2, train_unlabeled_mask_view2, train_unlabeled_Tform_view2 = self.ApplyAugmentation_Mask(train_unlabeled_data, train_unlabeled_gt)

            train_labeled_data_view1 = torch.tensor(train_labeled_data_view1, device=self.device,dtype=torch.float32)
            train_labeled_gt_view1 = torch.tensor(train_labeled_gt_view1, device=self.device,dtype=torch.float32)
            train_unlabeled_data_view1 = torch.tensor(train_unlabeled_data_view1, device=self.device,dtype=torch.float32)

            pred_sup_l = self.net(train_labeled_data_view1)
            pred_sup_r = self.net_r(train_labeled_data_view1)
            pred_sup_l = torch.squeeze(pred_sup_l, dim=1).view(-1)
            pred_sup_r = torch.squeeze(pred_sup_r, dim=1).view(-1)
            gts = train_labeled_gt_view1.view(-1)

            train_unlabeled_pred_view1_l = self.net(train_unlabeled_data_view1)
            train_unlabeled_pred_view1_r = self.net_r(train_unlabeled_data_view1)

            train_unlabeled_pred_view1_l_aligned = self.gt_model.invtransform_image_tensor(train_unlabeled_pred_view1_l, train_unlabeled_Tform_view1)
            train_unlabeled_pred_view1_r_aligned = self.gt_model.invtransform_image_tensor(train_unlabeled_pred_view1_r, train_unlabeled_Tform_view1)
            unlabeled_mask = torch.tensor(train_unlabeled_mask_view1 * train_unlabeled_mask_view2,dtype=torch.uint8,device=self.device)

            pred_unsup_l = torch.masked_select(train_unlabeled_pred_view1_l_aligned, unlabeled_mask)
            pred_unsup_r = torch.masked_select(train_unlabeled_pred_view1_r_aligned, unlabeled_mask)

            pred_sup_l = torch.sigmoid(pred_sup_l)
            pred_sup_r = torch.sigmoid(pred_sup_r)
            pred_unsup_l = torch.sigmoid(pred_unsup_l)
            pred_unsup_r = torch.sigmoid(pred_unsup_r)

            pred_l = torch.cat([pred_sup_l, pred_unsup_l])
            pred_r = torch.cat([pred_sup_r, pred_unsup_r])

            max_l = torch.zeros_like(pred_l)
            max_l[pred_l > 0.5] = 1
            max_r = torch.zeros_like(pred_r)
            max_r[pred_r > 0.5] = 1

            loss_cps = self.criterion_bce(pred_l, max_r) + self.criterion_bce(pred_r, max_l)
            loss_sup_l = self.criterion_bce(pred_sup_l, gts)
            loss_sup_r = self.criterion_bce(pred_sup_r, gts)
            loss = loss_cps * self.weight + loss_sup_l + loss_sup_r

            epoch_loss = epoch_loss * itr / (itr + 1) + float(loss.item()) / (itr + 1)
            epoch_loss_cps = epoch_loss_cps * itr / (itr + 1) + float(loss_cps) / (itr + 1)
            epoch_loss_sup_l = epoch_loss_sup_l * itr / (itr + 1) + float(loss_sup_l) / (itr + 1)
            epoch_loss_sup_r = epoch_loss_sup_r * itr / (itr + 1) + float(loss_sup_r) / (itr + 1)

            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

            loss.backward()
            self.optimizer_l.step()
            self.optimizer_r.step()

            self.global_step += 1
            itr += 1

        self.tr_loss = epoch_loss
        self.epoch_result["loss_tr"].append(epoch_loss)

        self.epoch_cnt += 1

        return epoch_loss


    def ValOneEpoch(self, epoch=0):
        epoch_loss = 0.

        val_pred_all = []
        val_gt_all = []

        # enable network for evaluation
        self.net.eval()

        itr = 0

        while True:
            ## Get next batch val samples
            FinishEpoch, data = \
                self.Loader.NextValBatch(epoch)
            if FinishEpoch:
                break

            ## Apply Normalization
            val_data = self.ApplyNormalization(data['data'])

            val_data = np.transpose(val_data, [0, 3, 1, 2])
            val_gt = np.transpose(data['gt'], [0, 1, 2])

            ## val current batch
            val_data = torch.tensor(val_data, device=self.device, dtype=torch.float32)
            val_gt = torch.tensor(val_gt, device=self.device, dtype=torch.float32)

            # forward pass
            val_pred = self.net(val_data)  # forward pass prediction of train labeled set
            loss = self.criterion_bce(torch.sigmoid(val_pred[:, 0, ...]), val_gt)
            epoch_loss = epoch_loss*itr/(itr+1) + loss.item()/(itr+1)

            # accumulate predictions and ground-truths
            val_pred_all.append(val_pred.detach().cpu().numpy())
            val_gt_all.append(val_gt.detach().cpu().numpy())

            itr += 1

        ## Evaluate val performance
        val_pred_all = np.concatenate(val_pred_all)
        val_pred_all = np.squeeze(val_pred_all > 0.5, axis=1).astype(float)  # binarize predictions
        val_gt_all = np.concatenate(val_gt_all).astype(float)

        perpix_acc = self.evaluator.perpixel_acc(val_pred_all, val_gt_all)  # evaluate per pixel accuracy
        persamp_iou = self.evaluator.persamp_iou(val_pred_all, val_gt_all)  # evaluate per sample iou
        micro_iou = self.evaluator.micro_iou(val_pred_all, val_gt_all)  # evaluate micro-average iou
        persamp_dice = self.evaluator.persamp_dice(val_pred_all, val_gt_all)  # persamp_dice
        micro_dice = self.evaluator.micro_dice(val_pred_all, val_gt_all)  # micro_dice

        self.val_loss = epoch_loss
        self.val_Acc = perpix_acc
        self.val_macro_IoU = persamp_iou
        self.val_micro_IoU = micro_iou
        self.val_macro_Dice = persamp_dice
        self.val_micro_Dice = micro_dice

        self.epoch_result["loss_val"].append(epoch_loss)
        self.epoch_result["macro_dice_val"].append(100 * np.mean(persamp_dice))
        self.epoch_result["micro_dice_val"].append(100 * micro_dice)
        self.epoch_result["perpix_acc_val"].append(100 * perpix_acc)
        self.epoch_result["macro_iou_val"].append(100 * np.mean(persamp_iou))
        self.epoch_result["micro_iou_val"].append(100 * micro_iou)

        return epoch_loss, persamp_dice, micro_dice, perpix_acc, persamp_iou, micro_iou



    def TestAll_SavePred(self,exp_fig=False,best_ckpt_filepath=None):
        epoch_loss = 0.

        test_pred_all = []
        test_gt_all = []
        test_gt_org_all = []

        ## Restore best ckpt
        if best_ckpt_filepath is not None:
            self.net.load_state_dict(torch.load(best_ckpt_filepath,map_location=self.device))

        self.net.eval()

        itr = 0

        while True:
            ## Get next batch test samples
            FinishEpoch, data = \
                self.Loader.NextTestBatch()
            if FinishEpoch:
                break

            test_data = data['data']
            test_gt = data['gt']
            test_gt_org = data['gt_org']
            test_names = data['name']

            ## Apply Normalization
            test_data = self.ApplyNormalization(test_data)

            test_data = np.transpose(test_data, [0, 3, 1, 2])
            test_gt = np.transpose(test_gt, [0, 1, 2])

            ## test current batch
            test_data = torch.tensor(test_data, device=self.device, dtype=torch.float32)
            test_gt = torch.tensor(test_gt, device=self.device, dtype=torch.float32)

            test_pred = self.net(test_data)
            loss_class = self.criterion_bce(torch.sigmoid(test_pred[:, 0, ...]),test_gt)
            epoch_loss = epoch_loss*itr/(itr+1) + loss_class.detach().cpu().numpy()/(itr+1)

            # accumulate predictions and ground-truths
            test_pred = torch.sigmoid(test_pred)
            for pred_i in test_pred.detach().cpu().numpy():
                test_pred_all.append(pred_i[0])
            for gt_i in test_gt.detach().cpu().numpy():
                test_gt_all.append(gt_i)

            if test_gt_org is not None:
                for gt_i in test_gt_org:
                    test_gt_org_all.append(gt_i)

            # export prediction and gt figures
            if exp_fig:
                for img_i, gt_i, pred_i in zip(test_names, test_gt, test_pred):
                    exp_gt_filepath = os.path.join(self.export_path, 'gt', '{}_gt.png'.format(img_i))
                    pred_path = os.path.join(self.export_path, 'pred')
                    if not os.path.exists(pred_path):
                        os.makedirs(pred_path)
                    ## Save Posterior
                    exp_pred_filepath = os.path.join(pred_path, '{}_pred.png'.format(img_i))
                    img = Image.fromarray(255 * pred_i.detach().cpu().numpy()[0]).convert('RGB')
                    img.save(exp_pred_filepath)
                    ## Save Binarized
                    exp_pred_filepath = os.path.join(pred_path, '{}_pred_bin.png'.format(img_i))
                    img = Image.fromarray(255. * (pred_i.detach().cpu().numpy()[0] > 0.5)).convert('RGB')
                    img.save(exp_pred_filepath)

            # increase local iteration
            itr += 1

        ## Evaluate test performance
        test_pred_all = np.array(test_pred_all)
        test_pred_bin_all = (test_pred_all > 0.5).astype(float)  # binarize predictions
        test_gt_all = np.array(test_gt_all).astype(float)

        perpix_acc = self.evaluator.perpixel_acc(test_pred_bin_all, test_gt_all)  # etestuate per pixel accuracy
        persamp_iou = self.evaluator.persamp_iou(test_pred_bin_all, test_gt_all)  # evaluate per sample iou
        micro_iou = self.evaluator.micro_iou(test_pred_bin_all, test_gt_all)  # evaluate micro-average iou
        AIU = self.evaluator.AIU(test_pred_all, test_gt_all)  # evaluate AIU
        micro_F1, macro_F1 = self.evaluator.F1(test_pred_bin_all, test_gt_all)
        persamp_dice = self.evaluator.persamp_dice(test_pred_bin_all, test_gt_all)
        micro_dice = self.evaluator.micro_dice(test_pred_bin_all, test_gt_all)

        self.best_te_loss = epoch_loss
        self.best_te_Acc = perpix_acc
        self.best_te_macro_IoU = persamp_iou
        self.best_te_micro_IoU = micro_iou
        self.best_te_AIU = AIU
        self.best_te_micro_F1 = micro_F1
        self.best_te_macro_F1 = macro_F1

        self.best_te_macro_Dice = persamp_dice
        self.best_te_micro_Dice = micro_dice
        return epoch_loss, persamp_dice, micro_dice, perpix_acc, persamp_iou, micro_iou, AIU


    def ApplyAugmentation_Mask(self, data, gt):
        data, gt = self.augmentor.augment(images=data.astype(np.float32), masks=gt.astype(np.float32))

        self.gt_model.construct_random_transform(data)
        Tform = self.gt_model.Tform
        data = self.gt_model.transform_images(data, extrapolation='reflect')  # transformed image data
        gt = self.gt_model.transform_images(gt[..., np.newaxis], extrapolation='reflect')[
            ..., 0]  # transformed ground-truth masks
        mask = np.ones_like(gt)
        mask = self.gt_model.transform_images(mask[..., np.newaxis], extrapolation='constant')[..., 0]
        mask = self.gt_model.invtransform_images(mask[..., np.newaxis], extrapolation='constant')[
            ..., 0]  # valid mask in original frame

        # Apply Normalization
        data = (data - self.Loader.mean) / self.Loader.stddev

        ## new dimension order
        data = np.transpose(data, [0, 3, 1, 2])
        gt = np.transpose(gt, [0, 1, 2])
        mask = np.transpose(mask, [0, 1, 2])

        return data, gt, mask, Tform



    def UpdateLatestModel(self):
        todelete_ckpt = os.path.join(self.ckpt_path,'model_epoch-{}.pt'.format(self.epoch_cnt - self.ValFreq * self.MaxKeepCkpt))
        if os.path.exists(todelete_ckpt):
            os.remove(todelete_ckpt)

        current_ckpt = os.path.join(self.ckpt_path, 'model_epoch-{}.pt'.format(self.epoch_cnt))
        torch.save(self.net.state_dict(), current_ckpt)

        if self.best_val_Dice < self.val_micro_Dice:
            self.best_val_Dice = self.val_micro_Dice
            best_ckpt = os.path.join(self.ckpt_path, 'model_epoch-{}.pt'.format('best'))
            os.system('cp {} {}'.format(current_ckpt, best_ckpt))

            self.UpdateBest()
            print('【train_unet_CPS】 saved current epoch as the best up-to-date model')



    class WeightRampup(nn.Module):
        def __init__(self,RampupType='Exp',RampupEpoch=50):
            super().__init__()
            self.RampupType = RampupType
            self.RampupEpoch = RampupEpoch

        def forward(self, **kwargs):

            epoch = kwargs['epoch']

            if self.RampupType == 'Exp':
                return np.exp(-10.*(1-np.clip(epoch, 0.0, self.RampupEpoch)/self.RampupEpoch)**2)
            elif self.RampupType == 'Step':
                return 1.0 if epoch >= self.RampupEpoch else 0.0



    def ExportTensorboard(self):
        self.writer.add_scalar('loss/train', self.tr_loss, self.epoch_cnt)
        self.writer.add_scalar('loss/val', self.val_loss, self.epoch_cnt)

        # 自己添加。
        self.writer.add_scalar('macro average Dice/val', np.mean(self.val_macro_Dice), self.epoch_cnt)
        self.writer.add_scalar('micro average Dice/val', np.mean(self.val_micro_Dice), self.epoch_cnt)
        self.writer.add_scalar('perpix_accuracy/val', 100 * self.val_Acc, self.epoch_cnt)
        self.writer.add_scalar('macro average IoU/val', np.mean(self.val_macro_IoU), self.epoch_cnt)
        self.writer.add_scalar('micro average IoU/val', np.mean(self.val_micro_IoU), self.epoch_cnt)


    def PrintTrValInfo(self):
        print("【train_unet_CPS】 ======================================================")
        print('Epoch: {}'.format(self.epoch_cnt))
        print('【train_unet_CPS】 macro average Dice/val:{:.2f}%'.format(100 * np.mean(self.val_macro_Dice)))
        print('【train_unet_CPS】 micro average Dice/val:{:.2f}%'.format(100 * self.val_micro_Dice))
        print('perpix_accuracy/val: {:.2f}%'.format(100 * self.val_Acc))
        print('macro average IoU/val:{:.2f}%'.format(100 * np.mean(self.val_macro_IoU)))
        print('micro average IoU/val:{:.2f}%'.format(100 * self.val_micro_IoU))
        print("【train_unet_CPS】 ======================================================")



    def SaveAllSettings(self, args):
        if args.SaveRslt:
            self.settings_filepath = os.path.join(self.result_path,'settings.txt')
            with open(self.settings_filepath,'w') as fid:
                fid.write('GPU:{}\n'.format(args.GPU))
                fid.write('Network:{}\n'.format(args.net))
                fid.write('LearningRate:{}\n'.format(args.LearningRate))
                fid.write('Epoch:{}\n'.format(args.Epoch))
                fid.write('batchsize:{}\n'.format(args.batchsize))
                fid.write('labelpercent:{}\n'.format(args.labelpercent))
                fid.write('loss:{}\n'.format(args.loss))
                fid.write('HingeC:{}\n'.format(args.HingeC))
                fid.write('Temperature:{}\n'.format(args.Temperature))
                fid.write('lp:{}\n'.format(args.lp))
                fid.write('ssl:{}\n'.format(args.ssl))
                fid.write('EmaAlpha:{}\n'.format(self.alpha))
                fid.write('Gamma:{}\n'.format(args.Gamma))
                fid.write('RampupEpoch:{}\n'.format(args.RampupEpoch))
                fid.write('RampupType:{}\n'.format(args.RampupType))
                fid.write('Target Dataset:{}\n'.format(args.TargetData))
                fid.write('Aux Dataset:{}\n'.format(args.AddUnlab))
                fid.write('AddLocation:{}\n'.format(args.Location))

