from avalanche.evaluation.metrics.accuracy import Accuracy
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from dataset import *
from utils import *
from timm.optim import create_optimizer_v2
from CVPT import CVPT
from VPT import VPT
from torch import nn


def train(result, model, dl, opt, scheduler, epoch, criterion=nn.CrossEntropyLoss()):
    for ep in range(epoch):
        model.train()
        model = model.cuda()
        total_loss = 0
        total_batch = 0
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()

            out = model(x)
            loss = criterion(out, y)

            total_loss += loss
            total_batch += 1

            opt.zero_grad()
            loss.backward()
            opt.step()

        print('Epoch:' + str(ep) + ' ' + 'avg_loss:' + str(float(total_loss) / int(total_batch)))

        if scheduler is not None:
            scheduler.step(ep)
        if ep % 10 == 9:
            acc = test(model, test_dl)
            print('ep' + str(ep) + ': ' + str(acc))
            if acc > result['best_acc']:
                result['best_ep'] = ep
                result['best_acc'] = acc
    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    model = model.cuda()
    for batch in dl:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y, 0)

    return acc.result()[0]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--opt', type=str, default='AdamW')
    parser.add_argument('--output', type=str, default='result.log')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--prompt_drop', type=float, default=0.1)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--init', type=int, default=1)
    parser.add_argument('--insert', type=int, default=3)
    parser.add_argument('--test', type=bool, default=False)

    args = parser.parse_args()
    config = get_config(args.dataset)
    if args.test:
        args.seed = config['seed']
        args.lr = config['lr']
        args.num = config['num']

    torch.cuda.set_device(args.device)
    print(args)
    set_seed(args.seed)

    model = CVPT(drop_path_rate=0.1, Prompt_num=args.num, PromptDrop=args.prompt_drop,
                init=args.init, insert=args.insert)
    model.load_pretrained('/ssh/VPT/ViT-B_16.npz')

    model.cuda()
    train_dl, test_dl = get_data(args.dataset, batch_size=args.batch_size)
    trainable = []
    model.reset_classifier(config['class_num'])

    result = {}
    result['best_acc'] = 0


    for n, p in model.named_parameters():
        if 'head' in n or 'Prompt_Tokens' in n:
            trainable.append(p)
            print(n)
        else:
            p.requires_grad = False
    opt = create_optimizer_v2(trainable, lr=args.lr, weight_decay=args.wd, opt=args.opt, momentum=0.9)
    scheduler = CosineLRScheduler(opt, t_initial=100, warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, cycle_decay=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    model = train(result, model, train_dl, opt, scheduler, epoch=100, criterion=criterion)

    with open(args.output, 'a') as f:
        f.write(
            str(args.seed) + ' : ' + str(args.dataset) + ' acc: ' + str(result['best_acc']) + ' epoch: ' +
            str(result['best_ep']) + '\n')
    print(result['best_acc'])
