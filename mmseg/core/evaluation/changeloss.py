from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class IterBasedLossAdjustHook(Hook):
    def __init__(self, switch_iter=48000, warmup_iter=16000):
        self.switch_iter = switch_iter
        self.warmup_iter = warmup_iter  # 可选：线性过渡
        
    def before_train_iter(self, runner):
        current_iter = runner.iter
        
        # 线性过渡示例（可选）
        if self.warmup_iter > 0 and current_iter >= self.switch_iter:
            progress = min(1.0, (current_iter - self.switch_iter) / self.warmup_iter)
            progress = max(0.2, progress)
            ce_weight = max(0.0, 1.0 - progress)
            dice_weight = min(1.0, progress)
            
            model = runner.model.module
            for loss in model.decode_head.loss_decode:
                if 'CrossEntropyLoss' in str(type(loss)):
                    loss.loss_weight = ce_weight
                if 'FocalLoss' in str(type(loss)):
                    loss.loss_weight = dice_weight

        # 硬切换方案
        else:
            if current_iter == self.switch_iter:
                model = runner.model.module
                for loss in model.decode_head.loss_decode:
                    if 'CrossEntropyLoss' in str(type(loss)):
                        loss.loss_weight = 0.0
                    if 'FocalLoss' in str(type(loss)):
                        loss.loss_weight = 1.0
