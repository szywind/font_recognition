from __future__ import print_function, division

import time
import os
import copy
import PIL

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from utils import plot_confusion_matrix, make_weights_for_balanced_classes, get_latest_models
from loss import f1_loss, mixed_f1_ce

from metrics import compute_accuracy
from config import configurations


class TraindataSet(torch.utils.data.Dataset):
    def __init__(self, train_features, train_labels, classes, transform):
        self.x_data = train_features
        self.y_data = train_labels
        self.len = len(train_labels)
        self.imgs = list(zip(self.x_data, self.y_data))
        self.targets = list(self.y_data)
        self.classes = classes
        self.transform = transform

    def __getitem__(self, index):
        with PIL.Image.open(self.x_data[index]) as img:
            image = img.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.y_data[index]

    def __len__(self):
        return self.len


def get_sampler(image_dataset):
    # create a weighted random sampler to process imbalanced data
    weights = make_weights_for_balanced_classes(image_dataset.imgs, len(image_dataset.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return sampler


##  Visualize Images
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.savefig('output/result/image.jpg')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


## Train the Model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:  # or epoch_loss < best_loss
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_ROOT, '{}-epoch-{}.pth'.format(MODEL, epoch)))
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'train':
                scheduler.step()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_model_cv(model, criterion, optimizer, scheduler, num_epochs=25):
    init_model_wts = copy.deepcopy(model.state_dict())
    model_list = []

    # for inputs, labels in dataloaders['trainval']:

    for fold in range(len(dataloaders_cv)):
        since = time.time()

        best_model_wts = copy.deepcopy(init_model_wts)
        model.load_state_dict(best_model_wts)
        best_acc = 0.0
        best_loss = float("inf")

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders_cv[fold][phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes_cv[fold][phase]
                epoch_acc = running_corrects.double() / dataset_sizes_cv[fold][phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:  # or epoch_loss < best_loss
                    best_model_path = os.path.join(CHECKPOINT_ROOT,
                                                   '{}-fold-{}-epoch-{}.pth'.format(MODEL, fold, epoch))
                    torch.save(model.state_dict(), best_model_path)
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                
                if phase == 'train':
                    scheduler.step()

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        # model.load_state_dict(best_model_wts)
        # model_list.append(model)
        model_list.append(best_model_path)
    return model_list


## Visualize the Model Predictions
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


## Finetuning the convnet
def fine_tune(num_classes, num_folds=1, criterion=nn.CrossEntropyLoss(), do_dropout=False, num_epochs=100):
    model_ft = backbones[MODEL](pretrained=True)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    if do_dropout:
        model_ft.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.2, training=m.training))
    torch.nn.init.xavier_uniform(model_ft.fc.weight)

    model_ft = model_ft.to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.5)

    if num_folds == 1:
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                               num_epochs=num_epochs)

        visualize_model(model_ft)
        model_ft = [model_ft]
    else:
        model_ft = train_model_cv(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
    return model_ft


## ConvNet as fixed feature extractor
def train_clf(num_classes):
    model_conv = backbones[MODEL](pretrained=True)

    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_classes)
    torch.nn.init.xavier_uniform(model_conv.fc.weight)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=25)

    visualize_model(model_conv)


## Test
def test(model_path, num_classes):
    # compute final accuracy on training and test sets
    if isinstance(model_path, str):
        saved_model = torch.load(model_path)
        model = backbones[MODEL]()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(saved_model)
    else:
        model = model_path

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # forward
    y_test, y_pred = [], []
    test_dataloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=False, num_workers=4)

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # y_test.extend(labels.float().squeeze().tolist())
            y_test.extend(labels.float().tolist())
            y_pred.extend(preds.tolist())
        
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        
        if num_classes == 2:
            # Compute f1-score
            te_acc, te_thresh = compute_accuracy(y_test, y_pred, do_search_thresh=False)
            print('* Accuracy on test set: %0.2f%% with thresh = %0.4f' % (100 * te_acc, te_thresh))

            # Compute confusion matrix
            cnf_matrix = confusion_matrix(list(y_test), list(y_pred > te_thresh))
        else:
            cnf_matrix = confusion_matrix(list(y_test), list(y_pred))

        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=range(2),
                            title='Confusion matrix without normalization({})'.format(MODEL))

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=range(2), normalize=True,
                            title='Normalized confusion matrix({})'.format(MODEL))


def test_cv(model_path_list, num_classes):
    # compute final accuracy on training and test sets
    num_fold = len(model_path_list)
    model = backbones[MODEL]()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    y_test_all, y_pred_all = [], []
    for model_path in model_path_list:
        saved_model = torch.load(model_path)
        model.load_state_dict(saved_model)

        # forward
        y_test_i, y_pred_i = [], []
        test_dataloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=4, shuffle=False,
                                                      num_workers=4)

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                # y_test.extend(labels.float().squeeze().tolist())
                y_test_i.extend(labels.float().tolist())
                y_pred_i.extend(preds.tolist())
            if not isinstance(y_pred_i, np.ndarray):
                y_pred_i = np.array(y_pred_i)

            y_test_all.append(y_test_i)
            y_pred_all.append(y_pred_i)

    y_pred = np.zeros(y_pred_all[0].shape)
    for y_pred_i in y_pred_all:
        y_pred += y_pred_i
    y_pred /= len(y_pred_all)

    assert y_test_all[0] == y_test_all[1] and y_test_all[0] == y_test_all[-1]
    y_test = y_test_all[0]

    # Compute f1-score
    te_acc, te_thresh = compute_accuracy(y_test, y_pred, do_search_thresh=False)
    print('* %d-Fold Accuracy on Test Set: %0.2f%% with thresh = %0.4f' % (num_fold, 100 * te_acc, te_thresh))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(list(y_test), list(y_pred > te_thresh))
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=range(2),
                          title='{}-Fold Confusion matrix without normalization({})'.format(num_fold, MODEL))

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=range(2), normalize=True,
                          title='{}-Fold Normalized confusion matrix({})'.format(num_fold, MODEL))


if __name__ == "__main__":
    config_idx = 1
    is_train = True

    cfg = configurations[config_idx]
    MODEL = cfg['MODEL']
    DATA_ROOT = cfg['DATA_ROOT']
    CHECKPOINT_ROOT = cfg['CHECKPOINT_ROOT']
    RESULT_ROOT = cfg['RESULT_ROOT']
    BATCH_SIZE = cfg['BATCH_SIZE']
    NUM_FOLD = cfg.get('NUM_FOLD', 1)
    RANDOM_STATE = cfg.get('RANDOM_STATE', 1234)

    if not os.path.exists(CHECKPOINT_ROOT):
        os.makedirs(CHECKPOINT_ROOT)
    if not os.path.exists(RESULT_ROOT):
        os.makedirs(RESULT_ROOT)

    backbones = {"resnet18": models.resnet18, "resnet34": models.resnet34,  "resnet50": models.resnet50, "inception": models.inception_v3}

    print(CHECKPOINT_ROOT)
    best_models = {"resnet18": os.path.join(CHECKPOINT_ROOT, 'best_models', 'resnet18-epoch-46.pth'),
                   # "resnet34": os.path.join(CHECKPOINT_ROOT, 'best_models', 'resnet34-epoch-9_128.pth'),
                   "resnet34": os.path.join(CHECKPOINT_ROOT, 'resnet34-epoch-0.pth'),
                   "resnet50": os.path.join(CHECKPOINT_ROOT, 'resnet50-epoch-13.pth'),
                   "inception": os.path.join(CHECKPOINT_ROOT, 'best_models', 'inception-epoch-9.pth')}

    is_inception = (MODEL == "inception")

    if is_inception:
        in_size = 340
        crop_size = 299
    else:
        # in_size = 256
        # crop_size = 224
        in_size = 128
        crop_size = 112

    plt.ion()  # interactive mode

    ## Load the Data
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(in_size),
            # transforms.RandomRotation(180, PIL.Image.BILINEAR),
            # transforms.CenterCrop(crop_size),
            transforms.RandomCrop(crop_size),
            transforms.GaussianBlur(kernel_size=3),
            # transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.125, 0.125, 0.125, 0.125),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(in_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(in_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = DATA_ROOT
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test']}

    for phase in ['train', 'val', 'test']:
        print(phase, image_datasets[phase].class_to_idx)


    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, sampler=get_sampler(image_datasets[x]),
                                       shuffle=False, num_workers=4) for x in ['train', 'val']}

    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=4, shuffle=False,
                                                      num_workers=4)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    print('dataset size: ', dataset_sizes)

    ## for cross validation
    if NUM_FOLD > 1 and os.path.isdir(os.path.join(data_dir, 'trainval')):
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=NUM_FOLD, random_state=RANDOM_STATE)
        image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'trainval'))

        dataloaders_cv = dict()
        dataset_sizes_cv = dict()

        y = image_dataset.targets
        X = [img[0] for img in image_dataset.imgs]
        classes = image_dataset.classes

        X = np.array(X)
        y = np.array(y, dtype=np.int64)

        for i, (train_index, val_index) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            print('{} fold:'.format(i))
            print('\ttrain set: ', X_train.shape, y_train.shape)
            print('\tval set: ', X_val.shape, y_val.shape)

            dataset_train = TraindataSet(X_train, y_train, classes, data_transforms['train'])
            dataset_val = TraindataSet(X_val, y_val, classes, data_transforms['val'])
            dataloaders_cv[i] = {
                'train': torch.utils.data.DataLoader(dataset_train, BATCH_SIZE, sampler=get_sampler(dataset_train),
                                                     shuffle=False, num_workers=4),
                'val': torch.utils.data.DataLoader(dataset_val, BATCH_SIZE, sampler=get_sampler(dataset_val),
                                                   shuffle=False, num_workers=4)
            }
            dataset_sizes_cv[i] = {'train': len(dataset_train), 'val': len(dataset_val)}

    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])

    if is_train:
        num_classes = len(image_datasets['train'].classes)
        print('# classes: ', num_classes)
        model = fine_tune(num_classes=num_classes, num_folds=NUM_FOLD, num_epochs=100)
        if len(model) > 1:
            test_cv(model, num_classes)
        else:
            test(model[0], num_classes)
    else:
        num_classes = len(image_datasets['test'].classes)
        print('# classes: ', num_classes)
        if NUM_FOLD > 1:
            model_list = get_latest_models(os.path.join(CHECKPOINT_ROOT, 'model_bce'))
            test_cv(model_list, num_classes)
        else:
            model = best_models[MODEL]
            test(model, num_classes)
            # test(os.path.join(CHECKPOINT_ROOT, 'resnet34-epoch-36.pth'))

    plt.ioff()
    plt.show()