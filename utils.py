import os
import itertools
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

print(__doc__)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join('output', 'result', title + '.jpg'))
    plt.show()


def make_weights_for_balanced_classes(images, nclasses):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight


def get_latest_models(data_root):
    model_file_list = glob.glob(os.path.join(data_root, '*.pth'))
    model_dict = {}
    for model_file in model_file_list:
        fold_num, epoch_num = list(map(int, re.findall('fold-(\d+)-epoch-(\d+)', model_file)[0]))
        if fold_num not in model_dict:
            model_dict[fold_num] = (epoch_num, model_file)
        else:
            if epoch_num > model_dict[fold_num][0]:
                model_dict[fold_num] = (epoch_num, model_file)

    return [model_dict[fold_num][-1] for fold_num in model_dict]

def save_model_old_format(model_path=os.path.join('../models/checkpoint', 'best_models', 'resnet34-epoch-14_80.pth')):
    import torch
    from torchvision import models
    import torch.nn as nn

    saved_model = torch.load(model_path)
    model = models.resnet34()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(saved_model)

    out_model_path = os.path.join(os.path.dirname(model_path), 'old_' + os.path.basename(model_path))
    torch.save(model.state_dict(), out_model_path, _use_new_zipfile_serialization=False)

if __name__ == '__main__':
    save_model_old_format()
    save_model_old_format(os.path.join('../models/checkpoint', 'best_models', 'resnet34-epoch-12_128.pth'))