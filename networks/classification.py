import torch
from transformers import DistilBertModel

class DistillBertClass(torch.nn.Module):
    """
    Transformer architecture for fine tuning a
    pre-trained DistilBertModel on arXiv data
    """
    def __init__(self, classes):
        super(DistillBertClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased");
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, classes)

    def forward(self, input_ids, attention_mask):
        output_l = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        # additional fine-tuning layers on pre-trained model
        hiddent_state = output_l[0]
        pooler = hiddent_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
