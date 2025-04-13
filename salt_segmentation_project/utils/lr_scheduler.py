import math
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, ReduceLROnPlateau


class WarmupScheduler(_LRScheduler):
    """
    Scheduler with linear warmup and then hands off to another scheduler.
    
    Args:
        optimizer: The optimizer to wrap
        warmup_epochs: Number of epochs for warmup
        warmup_start_lr: Initial learning rate for warmup (usually small)
        target_lr: Learning rate to reach at the end of warmup
        after_scheduler: Scheduler to use after warmup
    """
    def __init__(self, optimizer, warmup_epochs, warmup_start_lr, target_lr, after_scheduler=None):
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (self.target_lr - self.warmup_start_lr) 
                   for _ in self.base_lrs]
        
        if self.after_scheduler:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()
        
        return self.base_lrs
        
    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if isinstance(self.after_scheduler, ReduceLROnPlateau):
                if metrics is None:
                    raise ValueError("metrics parameter is required for ReduceLROnPlateau")
                self.after_scheduler.step(metrics)
            else:
                self.after_scheduler.step(epoch)
        else:
            return super().step(epoch)


def create_scheduler_with_warmup(optimizer, config, num_epochs):
    """
    Factory function to create a scheduler with warmup based on config settings.
    
    Args:
        optimizer: The optimizer to use with the scheduler
        config: Training configuration dictionary
        num_epochs: Total number of epochs
        
    Returns:
        A scheduler with warmup capability
    """
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    
    if warmup_epochs <= 0:
        # No warmup needed, return the regular scheduler
        if config['training']['scheduler'] == 'cosine':
            return CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=config['training']['lr'] * 0.01
            )
        elif config['training']['scheduler'] == 'plateau':
            return ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            return None
            
    # With warmup
    warmup_start_lr = config['training'].get('warmup_start_lr', config['training']['lr'] * 0.1)
    target_lr = config['training']['lr']
    
    # Create the base scheduler that will take over after warmup
    if config['training']['scheduler'] == 'cosine':
        after_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=config['training']['lr'] * 0.01
        )
    elif config['training']['scheduler'] == 'plateau':
        after_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        after_scheduler = None
        
    # Create and return the warmup scheduler
    return WarmupScheduler(
        optimizer, 
        warmup_epochs=warmup_epochs,
        warmup_start_lr=warmup_start_lr,
        target_lr=target_lr,
        after_scheduler=after_scheduler
    )
