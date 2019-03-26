class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
                
    def update(self, val, weight=1):
        self.add(val, weight)
            
    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        if self.count != 0:
            self.avg = self.sum / self.count
                
    def value(self):
        return self.val
                
    def average(self):
        return self.avg
                
