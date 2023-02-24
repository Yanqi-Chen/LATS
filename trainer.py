import time
import torch
import tqdm

from torch.cuda import amp
from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
import math

def sinpInc(n, N):
    # S-LATS
    return math.sin(math.pi * float(n) / N) / math.pi + float(n) / N

def sinppghInc(n, N, beta):
    # PGH
    x = float(n) / N
    lbeta = math.log(beta)
    beta_powerx = beta ** x
    return ((math.pi ** 2) * (beta_powerx - 1) + (beta_powerx - 2) * (lbeta ** 2) + beta_powerx * lbeta * (lbeta * math.cos(math.pi * x) + math.pi * math.sin(math.pi * x))) / ((math.pi ** 2) * (beta - 1) - 2 * (lbeta ** 2))

def sinpLowfBaseInc(n, N):
    # LATS
    return 0.5 * (1.0 + 2 * n + (math.sin(math.pi * (n - 0.5) / N) / math.sin(math.pi * 0.5 / N))) / (N + 1.0)

__all__ = ["train", "validate"]


def train(train_loader, model, criterion, optimizer, epoch, args, writer, scaler=None):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    
    if epoch >= args.pruning_start_epoch:
        step = epoch * num_batches
        total_step = num_batches * args.epochs
        begin_step = args.pruning_start_epoch * num_batches

        if args.low_freq and args.gradual == 'sinp':
            base_threshold = sinpLowfBaseInc(epoch - args.pruning_start_epoch, args.epochs - args.pruning_start_epoch)
            flat_step = (1 + math.cos((epoch - args.pruning_start_epoch) * math.pi / (args.epochs - args.pruning_start_epoch))) / num_batches / (1 + args.epochs - args.pruning_start_epoch)
            b_step = 0
        
    
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True).long()

        # compute output
        if scaler is not None:
            with amp.autocast():
                output = model(images)
                loss = criterion(output, target.view(-1))
        else:
            output = model(images)
            loss = criterion(output, target.view(-1))

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # update threshold
        if epoch >= args.pruning_start_epoch:
            step = step + 1
            if args.gradual is not None:
                if args.low_freq:
                    b_step = b_step + 1
                    flat_width = (base_threshold + b_step * flat_step) * args.flat_width
                    # writer.add_scalar("threshold", flat_width, step)
                    for module in model.modules():
                        if hasattr(module, 'setFlatWidth'):
                           module.setFlatWidth(flat_width) 
                else: 
                    if args.gradual == 'sinp':
                        flat_width = sinpInc(step - begin_step, total_step - begin_step)
                    elif args.gradual == 'sinppgh':
                        flat_width = sinppghInc(step - begin_step, total_step - begin_step, args.beta)
                    else:
                        raise NotImplementedError
                    
                    normal_flat_width = flat_width * args.flat_width
                    # writer.add_scalar("threshold", flat_width, step)
                    for module in model.modules():
                        if hasattr(module, 'setFlatWidth'):
                            module.setFlatWidth(normal_flat_width)
                                

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True).long()

            # compute output
            output = model(images)

            loss = criterion(output, target.view(-1))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    return top1.avg, top5.avg

