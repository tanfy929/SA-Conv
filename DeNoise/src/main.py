import torch

import utility
import data
import model
import loss
from option_train import args
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
            
            # ==================== 加载 TL 版本预训练权重 ====================
            print("Loading pre-trained TL weights for initialization...")
            checkpoint_path = args.dir_TL
            
            # 建议加上 map_location='cpu' 避免 GPU 显存残留问题
            pretrained_dict = torch.load(checkpoint_path, map_location='cpu') 
            
            # EDSR 框架默认保存的 state_dict 通常直接在顶层，或者包在 'state_dict' 键下
            if 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']
            elif 'model' in pretrained_dict:
                pretrained_dict = pretrained_dict['model']
                
            new_state_dict = {}
            for k, v in pretrained_dict.items():
                # 改为切片拼接，确保只修改最外层的层级名称
                if k.startswith('head.0.'):
                    new_k = 'head_conv.' + k[7:]  
                elif k.startswith('body.'):
                    new_k = 'body_modules.' + k[5:]
                else:
                    new_k = k
                new_state_dict[new_k] = v
            
            # EDSR 框架中 _model 通常是个 Wrapper，真实的 EDSR_plus 存在 _model.model 中
            actual_model = _model.get_model() if hasattr(_model, 'get_model') else (_model.model if hasattr(_model, 'model') else _model)
            
            # 传入 strict=False 允许忽略 HyperNet 缺失的参数
            actual_model.load_state_dict(new_state_dict, strict=False)
            print("TL weights loaded successfully with Key Mapping!")
            # ======================================================================
            
            total_params = sum(p.numel() for p in _model.parameters())
            trainable_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
            print(f"===> Total params:     {total_params / 1e6:.2f} M")  # Total params: 3.44 M
            print(f"===> Trainable params: {trainable_params / 1e6:.2f} M")
            
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
#            if args.resume ==1 and not args.test_only:
#                t.test()

            # # ==================== 退化验证 ==================== （已完成！）
            # print("======================================================")
            # print("Running initial validation to verify loaded TL weights...")
            # t.test() 
            # print("======================================================")
            # # ============================================================
            
            while not t.terminate():
                t.train() 
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
