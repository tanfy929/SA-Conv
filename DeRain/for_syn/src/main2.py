import torch
import os
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import multiprocessing
import time
import torch.nn as nn

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        print_network(model)
        
        # ==================== 加载去雨网络 TL 权重 ====================
        print("Loading pre-trained Rain Removal TL weights...")
        
    
        import os
        checkpoint_path = args.dir_TL 
        
        pretrained_dict = torch.load(checkpoint_path, map_location='cpu') 
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        elif 'model' in pretrained_dict:
            pretrained_dict = pretrained_dict['model']

        new_state_dict = {}
        for k, v in pretrained_dict.items():
            if 'GetBasis' in k:
                # 核心过滤：抛弃带 Batch 维度的网格，保留 Cx, Cy, theta0, U
                if not (k.endswith('Cx') or k.endswith('Cy') or k.endswith('theta0') or k.endswith('U')):
                    continue 
            
            # 因为 Sequential 到 ModuleList 的命名规则完全一致
            # 所以不需要任何 replace，直接全盘接收！
            new_state_dict[k] = v
            
        actual_model = model.get_model() if hasattr(model, 'get_model') else (model.model if hasattr(model, 'model') else model)
        
        # 加载权重
        actual_model.load_state_dict(new_state_dict, strict=False)

        # --- [修复：镇压所有 HyperNet 的噪声] ---
        # 由于 Xnet 被实例化了多次 (xnet, x_stage, fxnet)，模型里有多个 HyperNet
        # 需要遍历全模型，把每一个 HyperNet 的最后那一层都置零。
        try:
            hypernet_count = 0
            # 遍历模型的所有子模块
            for name, module in actual_model.named_modules():
                # 如果这个模块的名字是以 'HyperNet' 结尾的
                if name.endswith('HyperNet'):
                    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
                    if linears:
                        last_linear = linears[-1] # 取该 HyperNet 的最后一层
                        nn.init.zeros_(last_linear.weight)
                        if last_linear.bias is not None:
                            nn.init.zeros_(last_linear.bias)
                        hypernet_count += 1
            print(f"🛡️ [诊断] 成功锁定了 {hypernet_count} 个 HyperNet 的最后层输出为 0。")
        except Exception as e:
            print(f"⚠️ [诊断] 锁定 HyperNet 失败，报错: {e}")
            # ======================================================================
            
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"===> Total params:     {total_params / 1e6:.2f} M")  
        print(f"===> Trainable params: {trainable_params / 1e6:.2f} M")
        
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, _loss, checkpoint)

        # # ==================== 退化验证 ==================== 
        # print("======================================================")
        # print("Running initial validation to verify loaded TL weights...")
        # t.test() 
        # print("======================================================")
        # # ============================================================
        
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()
        checkpoint.done()
    



