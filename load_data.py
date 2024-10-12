def load_data(batch_size=64, num_workers=4, validation_split=0.1):
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, SubsetRandomSampler
    import numpy as np

    transform = transforms.Compose([
        transforms.ToTensor(),
        # 添加其他数据增强或预处理操作
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)

    # 创建数据集索引
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_split * num_train))

    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    trainloader = DataLoader(train_dataset, batch_size=batch_size,
                             sampler=train_sampler, num_workers=num_workers)

    valloader = DataLoader(train_dataset, batch_size=batch_size,
                           sampler=val_sampler, num_workers=num_workers)

    testloader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return trainloader, testloader, valloader
