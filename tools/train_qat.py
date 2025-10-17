import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import torch
from torch.quantization import convert, get_default_qconfig, prepare_qat
import src.misc.dist_utils as dist
from src.core import YAMLConfig
from src.solver import TASKS
from kd_loss import KnowledgeDistillationLoss

def main(args):
    dist.setup_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scratch or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        use_amp=args.amp,
        tuning=args.tuning
    )

    # Student model for QAT
    student_solver = TASKS[cfg.yaml_cfg['task']](cfg)
    student_solver._setup()  # Initialize the model and other components
    student_model = student_solver.model
    student_model.train()

    # Prepare the student model for QAT
    student_model.qconfig = get_default_qconfig('fbgemm')
    prepare_qat(student_model, inplace=True)

    # Manually set up optimizer and scheduler for the custom loop
    student_solver.optimizer = student_solver.cfg.optimizer
    student_solver.lr_scheduler = student_solver.cfg.lr_scheduler

    # Teacher model for knowledge distillation
    if cfg.yaml_cfg.get('distill', False):
        teacher_cfg = YAMLConfig(args.config)
        teacher_solver = TASKS[teacher_cfg.yaml_cfg['task']](teacher_cfg)
        teacher_solver._setup() # Initialize the teacher model
        teacher_model = teacher_solver.model
        teacher_weights = torch.load(cfg.yaml_cfg['teacher_weights'], map_location='cpu')['model']
        
        # Load state dict for teacher model
        unwrapped_teacher_model = dist.de_parallel(teacher_model)
        unwrapped_teacher_model.load_state_dict(teacher_weights)
        teacher_model.eval()
        
        kd_loss = KnowledgeDistillationLoss(alpha=cfg.yaml_cfg.get('distill_loss_alpha', 0.1))

    # Dataloader setup (from solver.train())
    student_solver.train_dataloader = dist.warp_loader(student_solver.cfg.train_dataloader, \
        shuffle=student_solver.cfg.train_dataloader.shuffle)
    student_solver.val_dataloader = dist.warp_loader(student_solver.cfg.val_dataloader, \
        shuffle=student_solver.cfg.val_dataloader.shuffle)


    if args.test_only:
        student_solver.eval()
    else:
        # Custom training loop for QAT and KD
        for epoch in range(student_solver.last_epoch, student_solver.cfg.epochs):
            for i, data in enumerate(student_solver.train_dataloader):
                
                # Move data to device
                data = {k: v.to(student_solver.device) if hasattr(v, 'to') else v for k, v in data.items()}
                
                student_solver.optimizer.zero_grad()

                # Forward pass for the student model
                with torch.cuda.amp.autocast(enabled=student_solver.scaler is not None):
                    outputs = student_model(data)
                    loss_dict = student_solver.criterion(outputs, data)
                
                total_loss = sum(loss_dict.values())

                # If KD is enabled, calculate and add the KD loss
                if cfg.yaml_cfg.get('distill', False):
                    with torch.no_grad():
                        teacher_outputs = teacher_model(data)
                    
                    student_queries = outputs['dn_meta']['output_proposals']
                    teacher_queries = teacher_outputs['dn_meta']['output_proposals']

                    distillation_loss = kd_loss(student_queries, teacher_queries)
                    total_loss += distillation_loss

                if student_solver.scaler:
                    student_solver.scaler.scale(total_loss).backward()
                    student_solver.scaler.step(student_solver.optimizer)
                    student_solver.scaler.update()
                else:
                    total_loss.backward()
                    student_solver.optimizer.step()

            # Validation, saving, etc.
            student_solver.eval()
            # Save the quantized model
            if (epoch + 1) % student_solver.cfg.save_period == 0 or (epoch + 1) == student_solver.cfg.epochs:
                student_model.eval()
                quantized_model = convert(dist.de_parallel(student_model).to('cpu'), inplace=False)
                torch.save(quantized_model.state_dict(), os.path.join(student_solver.output_dir, f'quantized_model_epoch_{epoch+1}.pth'))
                student_model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed',)
    args = parser.parse_args()

    main(args)