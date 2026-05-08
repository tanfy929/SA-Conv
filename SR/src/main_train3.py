import torch

import utility
import data
import model
import loss
from option_train_argument import args
from trainer import Trainer
import torch.nn as nn
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
            
            # ==================== 终极修复：加载 RCAN TL 权重 ====================
            print("Loading pre-trained RCAN TL weights...")
            checkpoint_path = args.dir_TL
            
            pretrained_dict = torch.load(checkpoint_path, map_location='cpu') 
            if 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']
            elif 'model' in pretrained_dict:
                pretrained_dict = pretrained_dict['model']

            new_state_dict = {}
            for k, v in pretrained_dict.items():
                if 'GetBasis' in k:
                    # 修复关键：必须放行 U 矩阵，防止 PCA 符号翻转导致特征错乱！
                    if not (k.endswith('Cx') or k.endswith('Cy') or k.endswith('theta0') or k.endswith('U')):
                        continue 
                
                # 结构解包映射 (保留)
                new_k = k
                if k.startswith('head.0.'):
                    new_k = k.replace('head.0.', 'head.', 1)
                elif k.startswith('tail.0.'):
                    new_k = k.replace('tail.0.', 'out.', 1)
                elif k.startswith('tail.1.'):
                    new_k = k.replace('tail.1.', 'tail.0.', 1)
                elif k.startswith('tail.2.'):
                    new_k = k.replace('tail.2.', 'tail.1.', 1)
                new_state_dict[new_k] = v
                
            actual_model = _model.get_model() if hasattr(_model, 'get_model') else (_model.model if hasattr(_model, 'model') else _model)
            
            # 加载权重
            actual_model.load_state_dict(new_state_dict, strict=False)

            # --- [修复：精准镇压] 只将 HyperNet 最后一层输出置 0 ---
            try:
                # 动态寻找所有的 Linear 层
                linears = [m for m in actual_model.HyperNet.modules() if isinstance(m, nn.Linear)]
                if linears:
                    last_linear = linears[-1] # 取最后的一层
                    nn.init.zeros_(last_linear.weight)
                    if last_linear.bias is not None:
                        nn.init.zeros_(last_linear.bias)
                print("🛡️ [诊断] 仅锁定 HyperNet 最后一层为 0，保留深层梯度流。")
            except Exception as e:
                print(f"⚠️ [诊断] 锁定 HyperNet 失败，报错: {e}")
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
