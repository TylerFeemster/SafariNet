from sklearn.metrics import multilabel_confusion_matrix, mean_squared_error, roc_auc_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

REDUCED_CLASSES = ['giraffe_reticulated', 'zebra_grevys',
                   'turtle_sea', 'zebra_plains',
                   'giraffe_masai', 'whale_fluke']

def _get_stdevs():
    num_nontest = 4623
    num_animals = np.zeros((num_nontest, 6))
    trainval = open('wild/ImageSets/Main/trainval.txt', 'r')
    filenames = trainval.readlines()
    for idx, file in enumerate(filenames):
        num_animals[idx, :] = np.load('wild/count_annotations/' +
                                      file[:-1] + '.npy')
    return np.std(num_animals, axis=0)

# training loop over batches; forward and backward propagation
def train_batch_loop(model, optimizer, train_dataloader, device, ratio: int = 1):
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    for _, (imgs, targets_classify, targets_counts) in enumerate(train_dataloader):
      imgs = imgs.to(device)
      targets_classify = targets_classify.to(device).float()
      targets_counts = targets_counts.to(device).float()

      outputs = model(imgs)
      bce_loss = bce(outputs[0], targets_classify)
      mse_loss = mse(outputs[1], targets_counts)
      loss = bce_loss + ratio * mse_loss  # ratio scales mse against bce (default: 1)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    return

# classwise errors for classification problem
def eval_classify(targets_classify, preds_classify):
    targets_classify_array = targets_classify.cpu().detach().numpy()
    preds_classify_array = preds_classify.cpu().detach().numpy()

    confusion_matrix = multilabel_confusion_matrix(targets_classify_array,
                                                   preds_classify_array)

    classwise_precisions = []
    for i in range(6):
      positives_i = confusion_matrix[i, 1, 1] + confusion_matrix[i, 0, 1]
      precision_i = confusion_matrix[i, 1, 1] / positives_i  # TP / (TP + FP)
      classwise_precisions.append(np.round(precision_i, 3))
    print(f'Average precision by class: {classwise_precisions}')

    roc_auc_scores = [roc_auc_score(targets_classify_array[:, i],
                                    preds_classify_array[:, i])
                      for i in range(6)]
    return roc_auc_scores

# classwise errors for counting problem
def eval_count(targets_count, preds_count, animal_stdevs: np.array = None):
    targets_count_array = targets_count.cpu().detach().numpy()
    preds_count_array = preds_count.cpu().detach().numpy()

    if animal_stdevs is None:
       animal_stdevs = np.ones(6)

    relrmse_classwise = []
    for i in range(6):
      # renormalize for fair judging
      preds = preds_count_array[:, i] * animal_stdevs[i]
      trgts = targets_count_array[:, i] * animal_stdevs[i]
      relrmse = np.sqrt(np.mean(np.square(preds - trgts) /
                        (preds + 1)))
      relrmse_classwise.append(relrmse)

    avg_relrmse = sum(relrmse_classwise) / len(relrmse_classwise)
    print(f'Mean Relative Root Mean Squared Error: {np.round(avg_relrmse, 4)}')
    return relrmse_classwise

# evaluation loop over batches
def eval_batch_loop(model, validation_dataloader, device, animal_stdevs: np.array = None):
    total_targets_classify = torch.Tensor().to(device)
    total_preds_classify = torch.Tensor().to(device)
    total_targets_count = torch.Tensor().to(device)
    total_preds_count = torch.Tensor().to(device)
    for _, (imgs, targets_classify, targets_counts) in enumerate(validation_dataloader):
      imgs = imgs.to(device)
      targets_classify = targets_classify.to(device).int()
      targets_counts = targets_counts.to(device).int()
      outputs = model(imgs)

      total_targets_classify = torch.cat(
          (total_targets_classify, targets_classify), dim=0)
      total_preds_classify = torch.cat(
          (total_preds_classify, torch.round(nn.Sigmoid()(outputs[0])).int()), dim=0)

      total_targets_count = torch.cat(
          (total_targets_count, targets_counts), dim=0)
      total_preds_count = torch.cat(
          (total_preds_count, torch.round(nn.ReLU()(outputs[1])).int()), dim=0)

    class_roc_auc = eval_classify(total_targets_classify,
                                  total_preds_classify)
    class_relrmse = eval_count(total_targets_count, total_preds_count, animal_stdevs=animal_stdevs)
    return class_roc_auc, class_relrmse

def train(model, train_dataloader, val_dataloader, save_path: str = None,
          num_epochs: int = 100, ratio: int = 1, stdevs: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if stdevs:
      animal_stdevs = _get_stdevs()
    else:
      animal_stdevs = None

    epoch_roc_aucs = np.zeros((num_epochs, 6))
    epoch_relrmses = np.zeros((num_epochs, 6))

    # formatting save_path
    if save_path is not None and save_path[-1] != '/':
       save_path = save_path + '/'

    for epoch in range(1, num_epochs + 1):
      print('='*70)
      print(f'Epoch: {epoch}')
      print('='*70)

      # forward & backward propagation in batch loop
      model.train()  # set model to training mode
      train_batch_loop(model, optimizer, train_dataloader, device, ratio=ratio)

      # save model weights
      if epoch % 10 == 0 and save_path is not None:
        torch.save(model.state_dict(), save_path + 'epoch_{epoch}.pth')

      # evaluating model at current epoch after training
      model.eval() # set model to evaluation mode
      roc_aucs, relrmses = eval_batch_loop(model, val_dataloader, device, animal_stdevs=animal_stdevs)

      epoch_roc_aucs[epoch - 1, :] = roc_aucs
      epoch_relrmses[epoch - 1, :] = relrmses

    return epoch_roc_aucs, epoch_relrmses

def epoch_graphs(aurocs, mses):
  # relrmses
  for i in range(6):
    plt.plot(np.arange(1,101), 
             mses[:,i], 
             label=REDUCED_CLASSES[i])
  plt.legend()
  plt.title('RelRMSE vs Epoch by Species')
  plt.xlabel('Epoch')
  plt.ylabel('Relative RMSE')
  plt.show()

  # avg relrmse
  relrmses = np.mean(mses, axis=1)
  plt.plot(np.arange(1,101), relrmses)
  plt.title('Averge RelRMSE vs Epoch')
  plt.xlabel('Epoch')
  plt.ylabel('Relative RMSE')
  plt.show()

  # roc aucs
  for i in range(6):
    plt.plot(np.arange(1, 101),
             aurocs[:, i], 
             label=REDUCED_CLASSES[i])
  plt.legend()
  plt.title('ROC AUC vs Epoch by Species')
  plt.xlabel('Epoch')
  plt.ylabel('ROC AUC')
  plt.show()

  # avg roc aucs
  roc_aucs = np.mean(aurocs, axis=1)
  plt.plot(np.arange(1, 101), roc_aucs)
  plt.title('Averge ROC AUC vs Epoch')
  plt.xlabel('Epoch')
  plt.ylabel('ROC AUC')
  plt.show()

  return

def display_results(relrmses, aurocs):
  formatter = 'RelRMSE = {}  &  AUROC = {}  <--  {}'
  for idx, label in enumerate(REDUCED_CLASSES):
     print(formatter.format(
         str(round(relrmses[idx], 2)),
         str(round(aurocs[idx], 2)),
         label
     ))
  avg_relrmse = np.mean(relrmses)
  avg_auroc = np.mean(aurocs)
  print(formatter.format(
      str(round(avg_relrmse, 2)),
      str(round(avg_auroc, 2)),
      'average'
  ))
