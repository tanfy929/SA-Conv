import torch

import utility
import data
import model
import loss
from option_train_argument import args
from trainer import Trainer

# torch.autograd.set_detect_anomaly(True)

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            # _model = torch.compile(model.Model(args, checkpoint))

            if args.load == '' and not args.test_only:
                print("Loading HyperNet pre-trained weights for initialization...")
                hyp_ckpt = torch.load('./model/model_epoch_360.pth')
                real_model = _model.model if hasattr(_model, 'model') else _model
                if hasattr(real_model, 'HyperNet'):
                    real_model.HyperNet.load_state_dict(hyp_ckpt['model_state_dict'], strict=True)
                    print(">> HyperNet weights initialized successfully.")
                else:
                    print(">> Warning: HyperNet module not found in the model definition!")

            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
#            if args.resume ==1 and not args.test_only:
#                t.test()
            while not t.terminate():
                t.train() 
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
