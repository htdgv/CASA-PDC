import torch
import torch.nn as nn
from transformers import AutoModel

class HybridDefenseClassifier1(nn.Module):
    def __init__(self, n_classes, n_meta_features, n_dmrs_features, model_name, token):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name, token=token)

        self.meta_layer = nn.Sequential(
            nn.Linear(n_meta_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.dmrs_layer = nn.Sequential(
            nn.Linear(n_dmrs_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 768 (BERT CLS) + 32 (meta) + 32 (DMRS) = 832
        self.fusion = nn.Sequential(
            nn.Linear(768 + 32 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, input_ids, attention_mask, meta_features, dmrs_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        meta_out = self.meta_layer(meta_features)
        dmrs_out = self.dmrs_layer(dmrs_features)
        combined = torch.cat((pooled, meta_out, dmrs_out), dim=1)
        return self.fusion(combined)

class HybridDefenseClassifier(nn.Module):
        def __init__(self, n_classes, n_meta_features, model_name, token):
            super(HybridDefenseClassifier, self).__init__()
            self.bert = AutoModel.from_pretrained(model_name, token=token)
            
            # Metadata branch
            self.meta_layer = nn.Sequential(
                nn.Linear(n_meta_features, 32),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
            # Fusion Classifier
            self.fusion = nn.Sequential(
                nn.Linear(768 + 32, 128),
                nn.ReLU(),
                nn.Dropout(0.4), # Increased dropout to prevent "memorizing" AI patterns
                nn.Linear(128, n_classes)
            )

        def forward(self, input_ids, attention_mask, meta_features):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0, :] # Use <s> token for RoBERTa
            
            meta_out = self.meta_layer(meta_features)
            combined = torch.cat((pooled_output, meta_out), dim=1)
            return self.fusion(combined)

class ClinicalFusionModel(nn.Module):
    def __init__(self, token, model_name="mental/mental-roberta-base", num_labels=9):
        super(ClinicalFusionModel, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_name, token=token)
        self.dropout = nn.Dropout(0.3)
        
        # 768 (RoBERTa) + 4 (Clinical Features) = 772
        self.classifier = nn.Sequential(
            nn.Linear(768 + 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, extra_features):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use the pooler_output or the first [CLS] token
        pooled_output = outputs.last_hidden_state[:, 0, :] 
        
        # Concatenate text vector with your 4 clinical features
        combined = torch.cat((pooled_output, extra_features), dim=1)
        
        logits = self.classifier(self.dropout(combined))
        return logits