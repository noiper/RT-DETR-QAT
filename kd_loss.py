import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha

    def forward(self, student_queries, teacher_queries):
        """
        Calculates the knowledge distillation loss.
        Args:
            student_queries (torch.Tensor): The object queries from the student model.
            teacher_queries (torch.Tensor): The object queries from the teacher model.
        Returns:
            torch.Tensor: The knowledge distillation loss.
        """
        loss = F.mse_loss(student_queries, teacher_queries)
        return self.alpha * loss
