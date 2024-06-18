from importlib import import_module

from dataloader import MSDataLoader
from torch.utils.data.dataloader import default_collate

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())   #import_module(data.rainheavy)
            trainset = getattr(module_train, args.data_train)(args)    #trainset : data.rainheavy에서 class RainHeavy 불러옴 (class안에 정의된 def 사용가능)
            self.loader_train = MSDataLoader(
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu
            )      #batch_size = 16, pin_memory = True

        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100']:
            module_test = import_module('data.benchmark')
            testset = getattr(module_test, 'Benchmark')(args, train=False)    # 무시
        else:
            module_test = import_module('data.' +  args.data_test.lower())    #import_module(data.rainheavytest)
            testset = getattr(module_test, args.data_test)(args, train=False) #trainset : data.rainheavytest에서 class RainHeavyTest 불러옴 (class안에 정의된 def 사용가능)

        self.loader_test = MSDataLoader(
            args,
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu
        )     # batch_size = 1, pin_memory = True

