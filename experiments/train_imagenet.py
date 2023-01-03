import argparse
import torch
from tqdm import trange
import difftopk

torch.set_num_threads(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-eid', '--experiment_id', type=int, default=None)

    parser.add_argument('--data_path', '-d', required=True, type=str)

    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--nloglr', type=float, default=4.5)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cpu', type=str)

    parser.add_argument('--method', type=str, choices=[
        'softmax_cross_entropy',
        'bitonic', 'bitonic__non_sparse', 'odd_even', 'splitter_selection',
        'neuralsort', 'softsort',
        # 'sinkhorn_sort',
        'smooth_hard_topk',
    ])
    parser.add_argument('--p_k', type=float, nargs='+', required=True)
    parser.add_argument('--inverse_temperature', type=float, default=1)
    parser.add_argument('-m', '--m', type=int, default=16)
    parser.add_argument('--distribution', type=str, default=None)
    parser.add_argument('--art_lambda', type=float, default=.5)
    parser.add_argument('--top1_mode', type=str, default='sm', choices=['sm', 'smce', 'sort'])

    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[])
    parser.add_argument('--eval_freq', default=100, type=int)
    parser.add_argument('--apply_relu', action='store_true')
    parser.add_argument('--apply_dropout', action='store_true')

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

    data = torch.load(args.data_path)

    if 'ImageNet21K-P' in args.data_path:
        data = {
            'train_labels': data['imagenet21k_train_labels'],
            'test_labels': data['imagenet21k_val_labels'],
            'train_embeddings': data['imagenet21k_train_embeddings'],
            'test_embeddings': data['imagenet21k_val_embeddings'],
        }

    if 'tf_efficientnet_l2_ns' in args.data_path:
        data = {
            'train_labels': data['train_labels'],
            'test_labels': data['val_labels'],
            'train_embeddings': data['train_embeddings'],
            'test_embeddings': data['val_embeddings'],
        }

    num_valid_points = 100_000 if 'ImageNet21K-P' in args.data_path else 50_000
    indices = torch.randperm(data['train_labels'].shape[0])
    train_indices = indices[num_valid_points:]
    valid_indices = indices[:num_valid_points]
    data['valid_labels'] = data['train_labels'][valid_indices]
    data['valid_embeddings'] = data['train_embeddings'][valid_indices]
    data['train_labels'] = data['train_labels'][train_indices]
    data['train_embeddings'] = data['train_embeddings'][train_indices]

    best_v_top1 = 0
    best_v_top5 = 0
    best_v_top10 = 0
    best_v_top20 = 0

    num_classes = data['train_labels'].max().item() + 1
    print('num_classes', num_classes)

    embedding_dim = data['train_embeddings'].shape[1]
    print('embedding_dim', embedding_dim)

    if len(args.hidden_dims) == 0:
        model = torch.nn.Linear(embedding_dim, num_classes).to(device)
    else:
        layers = []
        prev_dim = embedding_dim
        for dim in args.hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, dim))
            layers.append(torch.nn.ReLU())
            prev_dim = dim
        layers.append(torch.nn.Linear(prev_dim, num_classes))
        model = torch.nn.Sequential(*layers).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=10**(-args.nloglr))

    # data['test_embeddings'] = data['test_embeddings'].to(device)
    # data['test_labels'] = data['test_labels'].to(device)
    # data['train_embeddings'] = data['train_embeddings'].to(device)
    # data['train_labels'] = data['train_labels'].to(device)

    assert (data['test_embeddings'].shape[0] // args.batch_size) * args.batch_size == data['test_embeddings'].shape[0],\
        ((data['test_embeddings'].shape[0] // args.batch_size) * args.batch_size, data['test_embeddings'].shape[0])

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

        rp = torch.randperm(data['train_embeddings'].shape[0])
        for it in trange(data['train_embeddings'].shape[0] // args.batch_size):
            embeddings = data['train_embeddings'][rp[it * args.batch_size: (it + 1) * args.batch_size]]
            embeddings = embeddings.to(device).float()

            if args.apply_relu:
                embeddings = torch.relu(embeddings)
            if args.apply_dropout:
                embeddings = torch.nn.Dropout()(embeddings)

            labels = data['train_labels'][rp[it * args.batch_size: (it + 1) * args.batch_size]].to(device)

            outputs = model(embeddings)

            loss = loss_fn(outputs, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            ############################################################################################################
            # EVALUATION ###############################################################################################
            ############################################################################################################
            if (
                    it == 0
                    or (it + 1) % args.eval_freq == 0
                    or it == data['train_embeddings'].shape[0] // args.batch_size - 1
            ):

                # VALID

                v_top1 = 0
                v_top5 = 0
                v_top10 = 0
                v_top20 = 0
                total = 0
                for it in range(data['valid_embeddings'].shape[0] // args.batch_size):
                    embeddings = data['valid_embeddings'][it * args.batch_size: (it + 1) * args.batch_size]
                    embeddings = embeddings.to(device).float()
                    if args.apply_relu:
                        embeddings = torch.relu(embeddings)
                    labels = data['valid_labels'][it * args.batch_size: (it + 1) * args.batch_size].to(device)

                    with torch.no_grad():
                        outputs = model(embeddings)

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
                for it in range(data['test_embeddings'].shape[0] // args.batch_size):
                    embeddings = data['test_embeddings'][it * args.batch_size: (it + 1) * args.batch_size]
                    embeddings = embeddings.to(device).float()
                    if args.apply_relu:
                        embeddings = torch.relu(embeddings)
                    labels = data['test_labels'][it * args.batch_size: (it + 1) * args.batch_size].to(device)

                    with torch.no_grad():
                        outputs = model(embeddings)

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
