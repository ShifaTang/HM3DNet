def clip_gradient(optimizer, grad_clip):
    """
    对模型的梯度进行裁剪（gradient clipping），防止梯度爆炸，保持训练稳定性
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip) # 使用 clamp_ 方法将梯度限制在 [-grad_clip, grad_clip] 范围内


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    """
    动态调整学习率，通常随着训练的进行，逐渐减小学习率，帮助模型更好地收敛。
    """
    # decay = decay_rate ** (epoch // decay_epoch)
    decay = 0.85 ** (epoch // 20)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

class AverageMeter(object):
    """
    用于计算和存储一段时间内的平均值，常用于记录训练过程中各种指标（如损失、精度等）的移动平均。
    """
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
