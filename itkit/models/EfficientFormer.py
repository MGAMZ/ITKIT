# pyright: reportCallIssue=false
from collections.abc import Callable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientFormerV2(torch.nn.Module):
    def __init__(self,
                 out_channels:int=1,
                 model_name:str='efficientformerv2_s2.snap_dist_in1k',
                 in_chans:int=3,
                 activation:Callable|None=None,
                 *args, **kwargs):
        super().__init__()

        self.encoder = timm.create_model(
            model_name=model_name,
            pretrained=False,
            features_only=True,
            in_chans=in_chans,
            *args, **kwargs)
        self.activation = activation if activation is not None else nn.Identity()
        encoder_channels = self.encoder.feature_info.channels()
        decoder_channels = encoder_channels[::-1]
        self.decoder_blocks = nn.ModuleList()
        self.decoder_blocks.append(
            self._make_decoder_block(encoder_channels[-1], decoder_channels[0])
        )

        # 其余解码器块（包含跳跃连接）
        for i in range(1, len(encoder_channels)):
            in_channels = decoder_channels[i-1] + encoder_channels[-(i+1)]  # 上采样特征 + 跳跃连接特征
            layer_out_channels = decoder_channels[i]
            self.decoder_blocks.append(
                self._make_decoder_block(in_channels, layer_out_channels)
            )

        # 最终输出层
        self.final_conv = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)

    def _make_decoder_block(self, in_channels, out_channels):
        """构建解码器块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x:torch.Tensor):
        # 编码器前向传播
        encoder_features = self.encoder(x)

        # 解码器前向传播
        decoder_output = self.decoder_blocks[0](encoder_features[-1])
        # 逐层上采样并融合跳跃连接
        for i in range(1, len(self.decoder_blocks)):
            # 上采样到与跳跃连接特征相同的尺寸
            decoder_output = F.interpolate(
                decoder_output,
                size=encoder_features[-(i+1)].shape[2:],
                mode='bilinear',
                align_corners=False
            )

            # 融合跳跃连接特征
            decoder_output = torch.cat([decoder_output, encoder_features[-(i+1)]], dim=1)

            # 通过解码器块
            decoder_output = self.decoder_blocks[i](decoder_output)
        # 最终上采样到输入尺寸
        decoder_output = F.interpolate(
            decoder_output,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        return self.activation(self.final_conv(decoder_output))


if __name__ == '__main__':
    # 测试分割网络
    model = EfficientFormerV2(out_channels=3)  # 假设3类分割任务
    x = torch.randn(1, 3, 224, 224)

    print(f"输入形状: {x.shape}")

    # 测试编码器输出
    encoder_features = model.encoder(x)
    print("\n编码器特征图形状:")
    for i, feature in enumerate(encoder_features):
        print(f"Level {i}: {feature.shape}")

    # 测试完整网络输出
    output = model(x)
    print(f"\n最终输出形状: {output.shape}")
    print(f"输入输出形状匹配: {x.shape[2:] == output.shape[2:]}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
