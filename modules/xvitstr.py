from __future__ import absolute_import, division, print_function

import os
from functools import partial

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from timm.models import create_model
from timm.models.registry import register_model
from timm.models.vision_transformer import (Attention, Block, VisionTransformer, _cfg)
from timm.models.layers import to_2tuple

from utils import Loss_f


__all__ = [
    'xvitstr_tiny_patch16_224',
    'xvitstr_small_patch16_224',
    'xvitstr_base_patch16_224',
]

def create_xvitstr(num_tokens, model=None, checkpoint_path='', pretrained=True):
    """ 创建xvitstr模型

    参数：
        num_tokens: XViTSTR可识别的字符数，XViTSTR的分类数。XViTSTR识别94个字符+1个开始标记+1个结束标记，共96个字符。
        model: 要实例化的模型名称
        checkpoint_path: 初始化模型后要加载的检查点的路径
        pretrained: 是否加载deit的预训练模型，默认为True。训练时，将pretrained改为True, 测试时，将pretrained改为False
    返回：
        返回一个重置分类数后的XViTSTR模型，包含了XViTSTR的模型结构

    """
    xvitstr = create_model(
        model,
        pretrained=pretrained,
        num_classes=num_tokens,
        checkpoint_path=checkpoint_path)
    # 可能需要运行以获得零初始头进行转移学习
    # 重置分类头的分类数为96
    xvitstr.reset_classifier(num_classes=num_tokens)

    return xvitstr


class PatchEmbedWithPool(nn.Module):
    """ 将图片通过卷积操作分割成块，并使用MaxPool 
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches // 4

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.pool(self.proj(x)).flatten(2).transpose(1, 2)
        return x


class MyAttention(Attention):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 直接在这里计算正交约束，测试时可注释掉
        self.loss_fq = Loss_f(q)
        self.loss_fk = Loss_f(k)
        self.loss_fv = Loss_f(v)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MyBlock(Block):

    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0, attn_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
                         drop, attn_drop, drop_path, act_layer, norm_layer)
        self.attn = MyAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)



class XViTSTR(VisionTransformer):
    '''
    定义XViTSTR的模型结构（前向传播的过程）。实例化XViTSTR时传递的参数可作为自定义的XViTSTR模型的参数。
    XViTSR基本上是一种使用DeiT权重的ViT。修改head以支持STR的字符序列预测。
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = kwargs.get('depth')
        self.loss_fq = 0
        self.loss_fk = 0
        self.loss_fv = 0
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, 0.1, kwargs["depth"])]
        self.blocks = nn.ModuleList([MyBlock(dim=kwargs["embed_dim"], num_heads=kwargs["num_heads"], mlp_ratio=kwargs["mlp_ratio"], qkv_bias=kwargs["qkv_bias"], qk_scale=None, drop=0.1, attn_drop=0.1, drop_path=dpr[i], norm_layer=norm_layer) for i in range(kwargs["depth"])])

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        # 使用了Phil Wang的cls_token实现，谢谢
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 每次要先置0，防止累加
        self.loss_fq = 0
        self.loss_fk = 0
        self.loss_fv = 0
        for blk in self.blocks:
            x = blk(x)
            # 累加每个编码器块中的正交约束
            self.loss_fq += blk.attn.loss_fq
            self.loss_fk += blk.attn.loss_fk
            self.loss_fv += blk.attn.loss_fv
        x = self.norm(x)
        return x

    def forward(self, x, seqlen=25):
        x = self.forward_features(x)
        x = x[:, :seqlen]

        # batch, seqlen, embsize
        b, s, e = x.size()
        x = x.reshape(b*s, e)
        x = self.head(x).view(b, s, self.num_classes)
        return x

    def get_qkv_weights(self):
        qkv_weights = []
        for i in range(self.depth):
            qkv_weights.append(self.blocks[i].attn.qkv.weight)
        return qkv_weights
    
    def get_loss_fq(self):
        return self.loss_fq
    
    def get_loss_fk(self):
        return self.loss_fk
    
    def get_loss_fv(self):
        return self.loss_fv


def load_pretrained(model, cfg=None, num_classes=1000, in_chans=1, filter_fn=None, strict=True):
    '''
    装载预先训练好的检查点
    来自旧版本的timm
    '''
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is not None and 'url' in cfg and os.path.isfile(cfg['url']):
        # 从本地加载Torch序列化对象
        state_dict = torch.load(cfg['url'], map_location='cuda')
    else:
        # 从给定的URL加载Torch序列化对象
        state_dict = model_zoo.load_url(
            cfg['url'], progress=True, map_location='cpu')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        print("使用随机初始化，预训练模型URL无效。")
        return
    if "model" in state_dict.keys():
        state_dict = state_dict["model"]

    if filter_fn is not None:
        # 将补丁嵌入权重从手动 patchify + linear proj 转换为 conv
        # 其实里面的patch_embed.proj.weight的维度是符合需要的，在函数中的操作并没有改变维度
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        print('将第1层卷积（%s）的预训练权重从3个通道转换为1个通道' % conv1_name)
        key = conv1_name + '.weight'
        if key in state_dict.keys():
            conv1_weight = state_dict[conv1_name + '.weight']
        else:
            print('键(%s)不在state_dict中' % key)
            return
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # 对于带有space2depth杆的模型
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            # 对通道维度求和
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:
        # 完全放弃预训练模型和创建模型之间的所有其它差异的完全连接
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    print("加载预训练的vision transformer权重从%s ..." % cfg['url'])
    # 将从URL中加载的预训练权重进行一些修改后，包括合并通道维度的权重，删除最后一个head分类头的全连接的权重。strict=False表示state_dict中的key不需要和model.state_dict()返回的key严格一致。
    model.load_state_dict(state_dict, strict=strict)


def _conv_filter(state_dict, patch_size=16):
    """ 将patch embedding权重手动从patchify + linear proj转为conv """
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


@register_model
def xvitstr_tiny_patch16_224(pretrained=False, **kwargs):
    """ 
    @register_model是装饰器方法，调用xvitstr_tiny_patch16_224等价于调用register_model(xvitstr_tiny_patch16_224),该装饰器方法会该函数添加到一个模型入口点的字典中，键为函数名，值为函数本身。
    该方法返回一个具有XViTSTR模型结构的，加载了预训练权重的模型
    """
    # 在这里修改的输入图片的通道数
    # 使用rgb输入：kwargs['in_chans'] = 3
    kwargs['in_chans'] = 1

    # 配置自定义的参数，得到ViTSTR模型实例
    # 配置Dropout和DropPath：drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1
    model = XViTSTR(patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
                    qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1, **kwargs)

    model.default_cfg = _cfg(url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth')
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get(
            'in_chans', 1), filter_fn=_conv_filter)

    return model


@register_model
def xvitstr_small_patch16_224(pretrained=False, **kwargs):
    kwargs['in_chans'] = 1
    model = XViTSTR(patch_size=16, embed_dim=384, depth=12,
                    num_heads=6, mlp_ratio=4, qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1, **kwargs)
    model.default_cfg = _cfg(url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth")
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get(
            'in_chans', 1), filter_fn=_conv_filter)
    return model


@register_model
def xvitstr_base_patch16_224(pretrained=False, **kwargs):
    kwargs['in_chans'] = 1
    model = XViTSTR(patch_size=16, embed_dim=768, depth=12,
                    num_heads=12, mlp_ratio=4, qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1, **kwargs)
    model.default_cfg = _cfg(url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth')
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get(
            'in_chans', 1), filter_fn=_conv_filter)
    return model
