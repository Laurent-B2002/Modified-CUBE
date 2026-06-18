import argparse, os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.optim import AdamW, Adam, SGD
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import json
import pytorch_lightning as pl
import numpy as np
import torch.nn as nn
from collections import Counter
from scipy.stats import norm
from scipy import stats
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

##import user lib
from base.data_eeg import load_eeg_data
from base.data_meg import load_meg_data
from base.utils import update_config, ClipLoss, instantiate_from_config, get_device


def load_model(config, train_loader):
    model = {}
    for k, v in config['models'].items():
        print(f"init {k}")
        model[k] = instantiate_from_config(v)

    pl_model = PLModel(model, config, train_loader)
    return pl_model


def load_dataset(config):
    return load_eeg_data(config) if config['dataset'] == 'eeg' else load_meg_data(config)


def get_z_dim(config):
    features_filename = f"{config['db_dir']}/test.pt"
    f_tmp = torch.load(features_filename, weights_only=False)
    if 'low' in f_tmp['img_features']:
        return f_tmp['img_features']['low'][list(f_tmp['img_features']['low'].keys())[0]].shape[0]
    else:
        return f_tmp['img_features'][list(f_tmp['img_features'].keys())[0]].shape[0]


def micro_f1_score(gt, pred_probs, gt_thresh=0.33, pred_thresh=0.33, eps=1e-8):
    # Binarise ground truth and predictions
    gt_binary = (gt >= gt_thresh).astype(int)
    pred_binary = (pred_probs >= pred_thresh).astype(int)

    # True positives, false positives, false negatives
    TP = np.logical_and(pred_binary == 1, gt_binary == 1).sum()
    FP = np.logical_and(pred_binary == 1, gt_binary == 0).sum()
    FN = np.logical_and(pred_binary == 0, gt_binary == 1).sum()

    # Micro precision, recall, and F1
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return f1


def compute_colour_metrics(outputs, targets, scale=True):
    pearson = []
    spearman = []
    f1s = []
    outputs = 1 / (1 + torch.exp(-outputs))
    for p, t in zip(outputs, targets):
        gt = t.cpu().numpy()
        pred_raw = p.detach().cpu().numpy()
        pred = pred_raw * (np.sum(gt) / np.sum(pred_raw)) if scale else pred_raw
        corr = stats.pearsonr(pred, gt)[0]
        pearson.append(corr)
        corr = stats.spearmanr(pred, gt)[0]
        spearman.append(corr)
        f1s.append(micro_f1_score(gt, pred, 0.33, 0.33))
    return {
        'pearson': np.mean(pearson),
        'spearman': np.mean(spearman),
        'f1': np.mean(f1s)
    }


class PLModel(pl.LightningModule):
    def __init__(self, model, config, train_loader):
        super().__init__()

        self.config = config
        for key, value in model.items():
            setattr(self, f"{key}", value)
        self.criterion = ClipLoss()
        self.colour_criterion = nn.BCEWithLogitsLoss()

        self.all_predicted_classes = []
        self.all_true_labels = []

        self.z_dim = self.config['z_dim']

        self.sim = np.ones(len(train_loader.dataset))
        self.match_label = np.ones(len(train_loader.dataset), dtype=int)
        self.alpha = 0.05
        self.gamma = 0.3

        self.mAP_total = 0
        self.match_similarities = []

        self.test_colour_preds = []
        self.test_colour_gts = []

    def forward(self, batch, sample_posterior=False):

        idx = batch['idx'].cpu().detach().numpy()
        eeg = batch['eeg']
        colour_gt = batch['colour_gt']

        img_z = batch['img_features']

        eeg_z, colour_z = self.brain(eeg)

        if self.training:
            print("eeg std:", eeg.std().item())
            print("eeg_z std:", eeg_z.std().item())
            print("colour_z std:", colour_z.std().item())

        img_z = img_z / img_z.norm(dim=-1, keepdim=True)

        logit_scale = self.brain.logit_scale
        logit_scale = self.brain.softplus(logit_scale)

        eeg_loss, img_loss, logits_per_image = self.criterion(eeg_z, img_z, logit_scale)
        total_loss = (eeg_loss.mean() + img_loss.mean()) / 2

        if self.config['data']['uncertainty_aware']:
            diagonal_elements = torch.diagonal(logits_per_image).cpu().detach().numpy()
            gamma = self.gamma

            batch_sim = gamma * diagonal_elements + (1 - gamma) * self.sim[idx]

            mean_sim = np.mean(batch_sim)
            std_sim = np.std(batch_sim, ddof=1)
            match_label = np.ones_like(batch_sim)
            z_alpha_2 = norm.ppf(1 - self.alpha / 2)

            lower_bound = mean_sim - z_alpha_2 * std_sim
            upper_bound = mean_sim + z_alpha_2 * std_sim

            match_label[diagonal_elements > upper_bound] = 0
            match_label[diagonal_elements < lower_bound] = 2

            self.sim[idx] = batch_sim
            self.match_label[idx] = match_label

            loss = total_loss
        else:
            loss = total_loss
        #check things eeg
        colour_loss = self.colour_criterion(colour_z, colour_gt)
        colour_pred = torch.softmax(colour_z, dim=1)
        colour_loss = self.colour_criterion(colour_pred, colour_gt)
        colour_weight = 50.0
        loss = loss + colour_weight * colour_loss # colour_loss 
        return eeg_z, img_z, loss, colour_z, total_loss, colour_loss

    def training_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss, colour_z, clip_loss, colour_loss = self(batch, sample_posterior=True)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True, batch_size=batch_size)

        self.log(
            "clip_loss",
            clip_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        self.log(
            "colour_loss",
            colour_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)

        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        if batch_idx == self.trainer.num_training_batches - 1:
            all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
            all_true_labels = np.array(self.all_true_labels)
            top_1_predictions = all_predicted_classes[:, 0]
            top_1_correct = top_1_predictions == all_true_labels
            top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
            top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
            top_k_accuracy = sum(top_k_correct) / len(top_k_correct)
            f1_colour = compute_colour_metrics(colour_z, batch['colour_gt'])['f1']
            self.log('train_top1_acc', top_1_accuracy, on_step=False, on_epoch=True, prog_bar=True,
                     logger=True, sync_dist=True)
            self.log('train_top5_acc', top_k_accuracy, on_step=False, on_epoch=True, prog_bar=True,
                     logger=True, sync_dist=True)
            self.log('train_f1', f1_colour, on_step=False, on_epoch=True, prog_bar=True,
                     logger=True, sync_dist=True)
            self.all_predicted_classes = []
            self.all_true_labels = []

            counter = Counter(self.match_label)
            count_dict = dict(counter)
            key_mapping = {0: 'low', 1: 'medium', 2: 'high'}
            count_dict_mapped = {key_mapping[k]: v for k, v in count_dict.items()}
            self.log_dict(count_dict_mapped, on_step=False, on_epoch=True, logger=True,
                          sync_dist=True)
            self.trainer.train_dataloader.dataset.match_label = self.match_label
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]

        eeg_z, img_z, loss, colour_z, clip_loss, colour_loss = self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True, batch_size=batch_size)
        self.log(
            "val_clip_loss",
            clip_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size
        )

        self.log(
            "val_colour_loss",
            colour_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size
        )

        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)

        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        return loss

    def on_validation_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct) / len(top_k_correct)
        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        self.log('val_top5_acc', top_k_accuracy, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []

    def test_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss, colour_z, clip_loss, colour_loss = self(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True, batch_size=batch_size)
        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
        np.save(self.config['object_res_path'], eeg_z.detach().cpu().numpy())
        # np.save(self.config['colour_res_path'], colour_z.detach().cpu().numpy())
        self.test_colour_preds.append(colour_z.detach().cpu())
        self.test_colour_gts.append(batch['colour_gt'].detach().cpu())

        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())
        self.f1_colour = compute_colour_metrics(colour_z, batch['colour_gt'])['f1']

        # compute sim and map
        self.match_similarities.extend(similarity.diag().detach().cpu().tolist())

        for i in range(similarity.shape[0]):
            true_index = i
            sims = similarity[i, :]
            sorted_indices = torch.argsort(-sims)
            rank = (sorted_indices == true_index).nonzero()[0][0] + 1
            ap = 1 / rank
            self.mAP_total += ap

        return loss

    def on_test_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)

        all_colour_preds = torch.cat(self.test_colour_preds, dim=0).numpy()
        all_colour_gts = torch.cat(self.test_colour_gts, dim=0).numpy()

        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct) / len(top_k_correct)

        self.mAP = (self.mAP_total / len(all_true_labels)).item()
        self.match_similarities = np.mean(self.match_similarities) if self.match_similarities else 0

        self.log('test_top1_acc', top_1_accuracy, sync_dist=True)
        self.log('test_top5_acc', top_k_accuracy, sync_dist=True)
        self.log('test_f1_colour', self.f1_colour, sync_dist=True)
        self.log('mAP', self.mAP, sync_dist=True)
        self.log('similarity', self.match_similarities, sync_dist=True)

        np.save(self.config['colour_res_path'], all_colour_preds)
        np.save(
            self.config['colour_res_path'].replace(".npy", "_gt.npy"),
            all_colour_gts
)

        self.all_predicted_classes = []
        self.all_true_labels = []

        self.test_colour_preds = []
        self.test_colour_gts = []

        avg_test_loss = self.trainer.callback_metrics['test_loss']
        return {
            'test_loss': avg_test_loss.item(),
            'test_top1_acc': top_1_accuracy.item(),
            'test_top5_acc': top_k_accuracy.item(),
            'test_f1_colour': self.f1_colour,
            'mAP': self.mAP,
            'similarity': self.match_similarities
        }

    def configure_optimizers(self):
        optimizer = globals()[self.config['train']['optimizer']](
            self.parameters(), lr=self.config['train']['lr'], weight_decay=1e-4
        )
        return [optimizer]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="baseline.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        help="path to DB",
    )
    parser.add_argument(
        "--colour_gt_path",
        type=str,
        help="path to colour gt",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="experiment name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eeg",
        choices=["eeg", "meg"],
        help="Choose dataset: 'eeg' or 'meg'"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default='sub-08',
        help="the subjects",
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default='intra-subject',
        help="the exp_setting",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=50,
        help="train epoch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="lr",
    )
    parser.add_argument(
        "--brain_backbone",
        type=str,
        help="brain_backbone",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=6,
        help="c",
    )
    parser.add_argument(
        "--all_chs",
        action='store_true',
        default=False,
        help="Use all EEG channels",
    )
    parser.add_argument(
        "--leison_time",
        type=int,
        default=None,
        help="Leison temporal information",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="path to save directory",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Optional EEG time steps (must provide exactly two integers if used)",
    )

    # setting experiment configurations
    opt = parser.parse_args()
    config = OmegaConf.load(f"{opt.config}")
    if opt.timesteps is not None:
        config['timesteps'] = opt.timesteps
        opt.name = f"{opt.name}__{opt.timesteps[0]:04d}-{opt.timesteps[1]:04d}"
    else:
        opt.timesteps = config['timesteps']

    if opt.save_dir is not None:
        config['save_dir'] = opt.save_dir
    else:
        opt.save_dir = config['save_dir']
    config = update_config(opt, config)

    seed_everything(opt.seed)
    config['data']['leison_time'] = opt.leison_time
    config['data']['subjects'] = [opt.subjects]
    if opt.all_chs:
        config['data']['selected_ch'] = None
        config['models']['brain']['params']['c_num'] = 63

    seed_str = f"_seed{config['seed']}"
    leison_str = '' if opt.leison_time is None else f"_leison_{opt.leison_time:04d}"
    leison_dir = '' if opt.leison_time is None else f"leison/"
    test_out_dir = f"{config['save_dir']}/{opt.name}/{opt.subjects}{seed_str}/{leison_dir}"
    config['object_res_path'] = f"{test_out_dir}test_outs_object{leison_str}.npy"
    config['colour_res_path'] = f"{test_out_dir}test_outs_colour{leison_str}.npy"
    if os.path.exists(config['object_res_path']) and os.path.exists(config['colour_res_path']):
        print('**** Skipped', leison_dir)
        return
    else:
        os.makedirs(test_out_dir, exist_ok=True)

    logger = TensorBoardLogger(
        config['save_dir'],
        name=config['name'],
        version=f"{'_'.join({opt.subjects})}{seed_str}"
    )
    os.makedirs(logger.log_dir, exist_ok=True)

    config['z_dim'] = get_z_dim(config)
    print(config)
    with open(os.path.join(logger.log_dir, f'run_args.json'), 'w') as f:
        json.dump(OmegaConf.to_container(config, resolve=True), f, indent=4)

    shutil.copy(opt.config, os.path.join(logger.log_dir, opt.config.rsplit('/', 1)[-1]))

    train_loader, val_loader, test_loader = load_dataset(config)

    print(
        f"train num: {len(train_loader.dataset)}, "
        f"val num: {len(val_loader.dataset)}, "
        f"test num: {len(test_loader.dataset)}"
    )
    pl_model = load_model(config, train_loader)

    checkpoint_callback = ModelCheckpoint(save_last=True)

    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        min_delta=0.001,
        patience=5,
        verbose=False,
        mode='min'
    )

    device = get_device('auto')
    trainer = Trainer(
        log_every_n_steps=10, 
        strategy = "auto", #strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[checkpoint_callback],#,early_stop_callback],
        max_epochs=config['train']['epoch'],
        devices=1, #[device], 
        accelerator='cuda',
        logger=logger
    )

    trainer.fit(
        pl_model, ckpt_path='last',
        train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    test_results = trainer.test(ckpt_path='last', dataloaders=test_loader)

    with open(os.path.join(test_out_dir, f'test_results{leison_str}.json'), 'w') as f:
        json.dump(test_results, f, indent=4)


if __name__ == "__main__":
    main()
