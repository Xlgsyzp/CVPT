from avalanche.evaluation.metrics.accuracy import Accuracy
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from utils import *
from torch import nn
from timm.optim import create_optimizer_v2
from timm.models.vision_transformer import *
from src.configs.config import get_cfg
from src.data import loader as data_loader
from CVPT import CVPT

def train(config, model, dl, opt, scheduler, epoch, criterion=nn.CrossEntropyLoss()):
    for ep in range(epoch):
        model.train()
        model = model.cuda()
        total_loss = 0
        total_batch = 0
        for i, batch in enumerate(dl):
            x, y = batch['image'].cuda(), batch['label'].cuda()

            out = model(x)  # 前向传播
            loss = criterion(out, y)  # 计算损失

            total_loss += loss
            total_batch += 1
            # 反向传播并优化
            opt.zero_grad()
            loss.backward()
            opt.step()

        print('Epoch:' + str(ep) + ' ' + 'avg_loss:' + str(float(total_loss) / int(total_batch)))
        # print('cur_lr:'+str(opt.param_groups[0]['lr']))  # 获取当前学习率

        if scheduler is not None:
            scheduler.step(ep)
        if ep % 1 == 0:
            acc = test(model, test_dl)
            print('ep' + str(ep) + ': ' + str(acc))
            if acc > config['best_acc']:
                config['best_ep'] = ep
                config['best_acc'] = acc
    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    # pbar = tqdm(dl)
    model = model.cuda()
    torch.cuda.empty_cache()
    for batch in dl:  # pbar:
        x, y = batch['image'].cuda(), batch['label'].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y, 0)

    torch.cuda.empty_cache()
    return acc.result()[0]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--dataset', type=str, default='cars')
    parser.add_argument('--opt', type=str, default='AdamW')
    parser.add_argument('--output', type=str, default='result.log')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--prompt_drop', type=float, default=0.1)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--init', type=int, default=1)
    parser.add_argument('--insert', type=int, default=3)

    args = parser.parse_args()
    config = get_config(args.dataset)
    torch.cuda.set_device(args.device)
    print(args)
    set_seed(args.seed)

    model = CVPT(drop_path_rate=0.1, Prompt_num=args.num, PromptDrop=args.prompt_drop,
                init=args.init, insert=args.insert)
    model.load_pretrained('./ViT-B_16.npz')

    cfg = get_cfg()
    cfg.DATA.NAME = config['name']
    cfg.DATA.DATAPATH = config['path']
    cfg.DATA.BATCH_SIZE = args.batch_size
    train_dl = data_loader.construct_train_loader(cfg)
    test_dl = data_loader.construct_test_loader(cfg)


    trainable = []
    model.reset_classifier(config['class_num'])
    config['best_acc'] = 0

    for n, p in model.named_parameters():
        if 'head' in n or 'Prompt_Tokens' in n:
            trainable.append(p)
            print(n)
        else:
            p.requires_grad = False
    opt = create_optimizer_v2(trainable, lr=args.lr, weight_decay=args.wd, opt=args.opt, momentum=0.9)
    scheduler = CosineLRScheduler(opt, t_initial=100, warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, cycle_decay=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    model = train(config, model, train_dl, opt, scheduler, epoch=100, criterion=criterion)

    with open(args.output, 'a') as f:
        f.write(
            str(args.lr) + ' : ' + str(args.dataset) + ' acc: ' + str(config['best_acc']) + ' epoch: ' +
            str(config['best_ep']) + '\n')
    print(config['best_acc'])


