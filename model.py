import torch.nn as nn

from modules.xvitstr import create_xvitstr


class Model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.xvitstr = create_xvitstr(
            num_tokens=opt.num_class, model=opt.TransformerModel, pretrained=opt.isTrain)

    def forward(self, input, seqlen=25):
        prediction = self.xvitstr(input, seqlen=seqlen)
        return prediction

    def get_qkv_weights(self):
        return self.xvitstr.get_qkv_weights()
    
    def get_qkv_features(self):
        return self.xvitstr.get_qkv_features()
    
    def get_features_q(self):
        return self.xvitstr.features_q
    
    def get_features_k(self):
        return self.xvitstr.features_k
    
    def get_features_v(self):
        return self.xvitstr.features_v
    
    def get_loss_fq(self):
        return self.xvitstr.get_loss_fq()
    
    def get_loss_fk(self):
        return self.xvitstr.get_loss_fk()
    
    def get_loss_fv(self):
        return self.xvitstr.get_loss_fv()