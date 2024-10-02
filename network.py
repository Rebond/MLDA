import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, num_fts,dr=0.05):
        super(Encoder, self).__init__()
        self.dr = dr
        self.features = nn.Sequential(
            nn.Linear(num_fts, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dr),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dr),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dr),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(self.dr),
        )
    def forward(self, x):
        x = self.features(x)
        return x

class ClassClassifier(nn.Module):
    def __init__(self, num_cls):
        super(ClassClassifier, self).__init__()
        self.classifier = nn.Linear(32, num_cls)

    def forward(self, x):
        x = self.classifier(x)
        return x

class MLDA(nn.Module):
    def __init__(self,num_feats,num_class):
        super(MLDA,self).__init__()
        self.encoder = Encoder(num_feats)
        self.cls_classifier = ClassClassifier(num_class)

    def forward(self,src_data,tar_data):
        if tar_data == None:
            src_feature = self.encoder(src_data)
            src_output_cls = self.cls_classifier(src_feature)
            return src_feature, src_output_cls
        else:
            src_feature = self.encoder(src_data)
            tar_feature = self.encoder(tar_data)
            src_output_cls = self.cls_classifier(src_feature)
            tar_output_cls = self.cls_classifier(tar_feature)
            return src_feature,tar_feature,src_output_cls,tar_output_cls

class U(nn.Module):
    def __init__(self, num_fts=32,dr=0.05):
        super(U, self).__init__()
        self.dr = dr
        self.features = nn.Sequential(
            nn.Linear(num_fts, num_fts),
            nn.BatchNorm1d(num_fts),
            nn.ReLU(),
            nn.Dropout(self.dr),
        )
    def forward(self, x):
        x = self.features(x)
        return x

class V(nn.Module):
    def __init__(self, num_fts=32,dr=0.05):
        super(V, self).__init__()
        self.dr = dr
        self.features = nn.Sequential(
            nn.Linear(num_fts, num_fts),
            nn.BatchNorm1d(num_fts),
            nn.ReLU(),
            nn.Dropout(self.dr),
        )
    def forward(self, x):
        x = self.features(x)
        return x
