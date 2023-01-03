import argparse
import torch
from tqdm import trange, tqdm
from torchvision import datasets, transforms
from utils import resnet_cifar10
import difftopk

torch.set_num_threads(min(8, torch.get_num_threads()))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-eid', '--experiment_id', type=int, default=None)

    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--nloglr', type=float, default=3)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--method', type=str, choices=[
        'softmax_cross_entropy',
        'bitonic', 'bitonic__non_sparse', 'odd_even', 'splitter_selection',
        'neuralsort', 'softsort',
        # 'sinkhorn_sort',
        'smooth_topk', 'smooth_hard_topk',
    ])
    parser.add_argument('--p_k', type=float, nargs='+', required=True)
    parser.add_argument('--inverse_temperature', type=float, default=1)
    parser.add_argument('-m', '--m', type=int, default=None)
    parser.add_argument('--distribution', type=str, default=None)
    parser.add_argument('--art_lambda', type=float, default=.5)
    parser.add_argument('--top1_mode', type=str, default='smce', choices=['sm', 'smce', 'sort'])

    args = parser.parse_args()

    #####################################################

    if args.experiment_id is not None:
        from utils.results_json import ResultsJSON

        results = ResultsJSON(eid=args.experiment_id, path='./results/')
        results.store_args(args)

    #####################################################

    print('P_K', args.p_k)

    #####################################################

    print(vars(args))

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    main_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    val_size = len(main_set) // 20
    train_size = len(main_set) - val_size
    trainset, validset = torch.utils.data.random_split(main_set, [train_size, val_size])

    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
        validset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    best_v_top1 = 0
    best_v_top5 = 0
    best_v_top10 = 0
    best_v_top20 = 0

    num_classes = 100
    print('num_classes', num_classes)

    model = resnet_cifar10.resnet18(num_classes=num_classes).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=10**(-args.nloglr))

    if args.method == 'softmax_cross_entropy':
        loss_fn = torch.nn.CrossEntropyLoss()
        assert args.p_k[0] == 1, args.p_k
        assert sum(args.p_k) == 1, args.p_k
    elif args.method == 'smooth_hard_topk':
        loss_fn = difftopk.SmoothHardTopKLoss(num_classes, len(args.p_k), tau=args.inverse_temperature, device=device)
        assert args.p_k[-1] == 1, args.p_k
        assert sum(args.p_k) == 1, args.p_k
    else:
        loss_fn = difftopk.TopKCrossEntropyLoss(
            diffsort_method=args.method,
            inverse_temperature=args.inverse_temperature,
            p_k=args.p_k,
            n=num_classes,
            m=args.m,
            distribution=args.distribution,
            art_lambda=args.art_lambda,
            device=args.device,
            top1_mode=args.top1_mode,
        )

    for ep in range(args.n_epochs):
        print('Epoch', ep)

        ################################################################################################################
        # TRAINING #####################################################################################################
        ################################################################################################################

        model.train()
        for it, (data, labels) in tqdm(enumerate(train_loader), leave=False):

            outputs = model(data.to(device))

            loss = loss_fn(outputs, labels.to(device))

            optim.zero_grad()
            loss.backward()
            optim.step()

        ################################################################################################################
        # EVALUATION ###################################################################################################
        ################################################################################################################

        model.eval()

        # VALID

        v_top1 = 0
        v_top5 = 0
        v_top10 = 0
        v_top20 = 0
        total = 0
        for data, labels in valid_loader:
            with torch.no_grad():
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)

                v_top1 += (torch.max(outputs, dim=1)[1] == labels).float().sum().item()
                v_top5 += (torch.topk(outputs, k=5, dim=1)[1] == labels.unsqueeze(1)).float().sum().item()
                v_top10 += (torch.topk(outputs, k=10, dim=1)[1] == labels.unsqueeze(1)).float().sum().item()
                v_top20 += (torch.topk(outputs, k=20, dim=1)[1] == labels.unsqueeze(1)).float().sum().item()
                total += outputs.shape[0]
        print('Valid: Top-1 {:.4f} | Top-5 {:.4f}'.format(v_top1/total, v_top5/total))

        if args.experiment_id is not None:
            results.store_results({
                'v_top1': v_top1/total,
                'v_top5': v_top5/total,
                'v_top10': v_top10/total,
                'v_top20': v_top20/total,
                'v_loss': loss.item(),
            })

        # TEST

        top1 = 0
        top5 = 0
        top10 = 0
        top20 = 0
        total = 0
        for data, labels in test_loader:
            with torch.no_grad():
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)

                top1 += (torch.max(outputs, dim=1)[1] == labels).float().sum().item()
                top5 += (torch.topk(outputs, k=5, dim=1)[1] == labels.unsqueeze(1)).float().sum().item()
                top10 += (torch.topk(outputs, k=10, dim=1)[1] == labels.unsqueeze(1)).float().sum().item()
                top20 += (torch.topk(outputs, k=20, dim=1)[1] == labels.unsqueeze(1)).float().sum().item()
                total += outputs.shape[0]
        print('Test: Top-1 {:.4f} | Top-5 {:.4f}'.format(top1/total, top5/total))

        if args.experiment_id is not None:
            results.store_results({
                'top1': top1/total,
                'top5': top5/total,
                'top10': top10/total,
                'top20': top20/total,
                'loss': loss.item(),
            })

            if v_top1 > best_v_top1:
                results.store_final_results({'top1': top1 / total})
                best_v_top1 = v_top1
            if v_top5 > best_v_top5:
                results.store_final_results({'top5': top5 / total})
                best_v_top5 = v_top5
            if v_top10 > best_v_top10:
                results.store_final_results({'top10': top10 / total})
                best_v_top10 = v_top10
            if v_top20 > best_v_top20:
                results.store_final_results({'top22': top20 / total})
                best_v_top20 = v_top20

            results.save()

    if args.experiment_id is not None:
        results.store_final_results({'finished': True})
        results.save()
