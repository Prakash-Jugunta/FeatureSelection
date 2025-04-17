import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import matplotlib.pyplot as plt
import pyswarms as ps
import os
from model import AttentionModel, select_idx_F, sorted_idx, add_more_features, add_one_features, log, pso_fitness

def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)

clfl = [
    KNeighborsClassifier(n_neighbors=3),
    svm.LinearSVC(dual=False),
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(random_state=0)
]
l = ["knn", "svm", "decision_tree", "random_forest"]

path = "data03.csv"  # Update to your dataset path
lr = 0.01
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epoch = 200
batch_size = 100
validation_split = 0.3
threshold = 0.6
pso_iterations = 50
pso_particles = 20
pso_c1 = 2.0
pso_c2 = 2.0
pso_w = 0.7

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

data = np.loadtxt(path, delimiter=",", skiprows=1)
data, label = data[:, :-1], data[:, -1]
(a, n_feature) = data.shape
min_max_scaler = MinMaxScaler()
data = min_max_scaler.fit_transform(data)
num_label = int(max(label) + 1)

data1, label1 = data, label
data, label = torch.from_numpy(data).float(), torch.from_numpy(label).long()
dataset = Data.TensorDataset(data, label)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = Data.Subset(dataset, train_indices)
valid_sampler = Data.Subset(dataset, val_indices)
train_loader = Data.DataLoader(train_sampler, batch_size=batch_size)
validation_loader = Data.DataLoader(valid_sampler, batch_size=batch_size)

logpath = './log_hybrid.txt'
logger = log(logpath)

rd = [42, 123, 7, 99, 56]  # Reduced for brevity; expand as needed

for random_seed in rd:
    setup_seed(random_seed)
    ensure_dir(f'./{str(random_seed)}/png/')
    ensure_dir(f'./{str(random_seed)}/prop/')
    ensure_dir(f'./{str(random_seed)}/y/')

    # Step 1: Train Attention Model
    model = AttentionModel(n_feature, num_label).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, epoch, 1, eta_min=1e-3)
    loss_fun = nn.CrossEntropyLoss()

    for i in range(epoch):
        acc = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fun(output, targets)
            loss += 1e-2 / torch.sum((model.feature_selection_MLP.get_selection_rate() - 0.5) ** 2)
            loss.backward()
            optimizer.step()
            predicts = output.argmax(dim=1)
            acc += torch.eq(predicts, targets).sum().float().item()
        scheduler.step()
        logger.info(
            f'EPOCH: {i} train: lr: {optimizer.param_groups[0]["lr"]} acc rate: {acc / train_sampler.__len__()}'
        )
        if i % 10 == 0:
            test_acc = 0
            with torch.no_grad():
                for inputs, targets in validation_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    output = model(inputs)
                    predicts = output.argmax(dim=1)
                    test_acc += torch.eq(predicts, targets).sum().float().item()
                logger.info(
                    f'EPOCH: {i} test: lr: {optimizer.param_groups[0]["lr"]} acc rate: {test_acc / valid_sampler.__len__()}'
                )

    select_rate = model.feature_selection_MLP.get_selection_rate().cpu()
    torch.save(select_rate, f'./{str(random_seed)}/select_rate.pt')

    # Step 2: PSO Optimization
    count = -1
    record = []
    for clf in clfl:
        count += 1
        name = l[count]
        options = {'c1': pso_c1, 'c2': pso_c2, 'w': pso_w}
        init_pos = (select_rate[0].detach().numpy() > 0.5).astype(float)
        init_pos += np.random.rand(n_feature) * 0.1
        init_pos = np.clip(init_pos, 0, 1)  # Ensure values are within [0, 1]
        init_pos = np.tile(init_pos, (pso_particles, 1))
        optimizer = ps.single.GlobalBestPSO(
            n_particles=pso_particles,
            dimensions=n_feature,
            options=options,
            init_pos=init_pos,
            bounds=([0] * n_feature, [1] * n_feature)
        )

        def fitness_wrapper(particles):
            return pso_fitness(particles, data1, label1, clf, validation_split)

        cost, pos = optimizer.optimize(fitness_wrapper, iters=pso_iterations)
        selected_features = pos > 0.5
        if np.sum(selected_features) == 0:
            selected_features = select_rate[0].detach().numpy() > 0.5

        # Evaluate PSO-selected features
        X_subset = data1[:, selected_features]
        data_train, data_test, target_train, target_test = train_test_split(
            X_subset, label1, test_size=validation_split, random_state=42
        )
        clf.fit(data_train, target_train)
        score = clf.score(data_test, target_test)
        logger.info(f'Classifier: {name} PSO Feature_shape: {X_subset.shape} Score: {score}')

        # Step 3: Evaluate feature subsets as in original code
        idx = sorted_idx(select_rate)
        lit = np.arange(0.02, 1.01, 0.02)
        lis = [int(i * n_feature) for i in lit]
        y = []
        plt.figure(dpi=80)
        x = lit
        ct = False
        for i in lis:
            if not ct:
                selected_f = select_idx_F(torch.from_numpy(data1), idx, int(i))
                ct = True
            else:
                selected_f = add_more_features(torch.from_numpy(data1), selected_f, idx, int(i), int(j))
            j = i
            data_train, data_test, target_train, target_test = train_test_split(
                selected_f, label1, test_size=validation_split, random_state=42
            )
            clf.fit(data_train, target_train)
            score = clf.score(data_test, target_test)
            logger.info(f'Classifier: {name} Feature_shape: {selected_f.shape} Score: {score}')
            y.append(score)

        m = max(y)
        i = y.index(m)
        prop = (i + 1) * 0.02
        features = round(n_feature * prop) - 25
        plt.scatter(x, y, label='Subsets', marker='*')
        plt.xlabel('Proportion of features')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Change with Proportion of Features ({name})')
        plt.legend(bbox_to_anchor=(1, 1), loc="upper right", ncol=1, mode="None", borderaxespad=0, shadow=False, fancybox=True)
        plt.savefig(f'./{str(random_seed)}/png/{name}_subset_prop.png')
        plt.close()
        np.save(f'./{str(random_seed)}/prop/{name}y.npy', np.array(y))

        low = features if features > 1 else 2
        high = min(low + 50, n_feature)
        lit_n = list(range(low, high))
        y = []
        plt.figure(dpi=80)
        x = lit_n
        ct_1 = False
        for i in lit_n:
            if not ct_1:
                selected_f = select_idx_F(torch.from_numpy(data1), idx, int(i))
                ct_1 = True
            else:
                selected_f = add_one_features(torch.from_numpy(data1), selected_f, idx, int(i))
            data_train, data_test, target_train, target_test = train_test_split(
                selected_f, label1, test_size=validation_split, random_state=42
            )
            clf.fit(data_train, target_train)
            score = clf.score(data_test, target_test)
            logger.info(f'Classifier: {name} Feature_shape: {selected_f.shape} Score: {score}')
            y.append(score)

        m = max(y)
        Y = y.index(max(y)) + low
        record.append(m)
        record.append(Y)
        plt.scatter(x, y, label='Subsets', marker='*')
        plt.scatter(Y, m, marker='*')
        show_max = f'[{Y} {round(max(y), 4)}]'
        plt.annotate(show_max, xytext=(Y, m + 0.0001), xy=(Y, m))
        plt.xlabel('Number of features')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Change with Number of Features ({name})')
        plt.legend(bbox_to_anchor=(1, 1), loc="upper right", ncol=1, mode="None", borderaxespad=0, shadow=False, fancybox=True)
        plt.savefig(f'./{str(random_seed)}/png/{name}subset.png')
        plt.close()
        np.save(f'./{str(random_seed)}/y/{name}y.npy', np.array(y))

    np.save(f'./{str(random_seed)}/record.npy', record)