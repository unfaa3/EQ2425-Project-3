from load_data import load_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from net import Net, Net_A1, Net_A2, Net_B, Net_C, Net_D, Net_E

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    net = Net()
    net.to(device)
    trainloader, testloader = load_data()
    criterion = nn.CrossEntropyLoss()

    def train_net(net, trainloader, optimizer, epochs=300):
        net.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()  # 使用传入的优化器

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()  # 使用传入的优化器进行更新

                running_loss += loss.item()

            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')
        print('Finished Training')


    def test_net(net, testloader):
        net.eval()
        num_classes = 10  # Assuming 10 classes
        true_positives = [0] * num_classes
        false_negatives = [0] * num_classes

        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)

                for cls in range(num_classes):
                    true_positives[cls] += ((predicted == cls) & (labels == cls)).sum().item()
                    false_negatives[cls] += ((predicted != cls) & (labels == cls)).sum().item()

        recall_per_class = []
        for cls in range(num_classes):
            tp = true_positives[cls]
            fn = false_negatives[cls]
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0
            recall_per_class.append(recall)
            print(f'Recall of class {cls}: {recall * 100:.2f}%')

        average_recall = sum(recall_per_class) / num_classes
        print(f'Average Top-1 Recall Rate: {average_recall * 100:.2f}%')
        return average_recall

    optimizer = optim.SGD(net.parameters(), lr=1e-3)
    print("Training Net...")
    train_net(net, trainloader, optimizer)
    print("Testing Net...")
    recall = test_net(net, testloader)

    net_a1 = Net_A1().to(device)
    net_a2 = Net_A2().to(device)
    optimizer_a1 = optim.SGD(net_a1.parameters(), lr=1e-3)
    optimizer_a2 = optim.SGD(net_a2.parameters(), lr=1e-3)
    # 训练模型A1
    print("Training Net_A1...")
    train_net(net_a1, trainloader, optimizer_a1)
    print("Testing Net_A1...")
    recall_a1 = test_net(net_a1, testloader)
    print("Training Net_A2...")
    train_net(net_a2, trainloader, optimizer_a2)
    print("Testing Net_A2...")
    recall_a2 = test_net(net_a2, testloader)

    net_b = Net_B().to(device)
    optimizer_b = optim.SGD(net_b.parameters(), lr=1e-3)
    print("Training Net_B...")
    train_net(net_b, trainloader, optimizer_b)
    print("Testing Net_B...")
    recall_b = test_net(net_b, testloader)

    net_c = Net_C().to(device)
    optimizer_c = optim.SGD(net_c.parameters(), lr=1e-3)
    print("Training Net_C...")
    train_net(net_c, trainloader, optimizer_c)
    print("Testing Net_C...")
    recall_c = test_net(net_c, testloader)

    net_d = Net_D().to(device)
    optimizer_d = optim.SGD(net_d.parameters(), lr=1e-3)
    print("Training Net_D...")
    train_net(net_d, trainloader, optimizer_d)
    print("Testing Net_D...")
    recall_d = test_net(net_d, testloader)

    net_e = Net_E().to(device)
    optimizer_e = optim.SGD(net_e.parameters(), lr=1e-3)
    print("Training Net_E...")
    train_net(net_e, trainloader, optimizer_e)
    print("Testing Net_E...")
    recall_e = test_net(net_e, testloader)