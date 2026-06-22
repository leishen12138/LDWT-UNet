import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2


class UnderwaterColorCompensation(nn.Module):
    """Physics-inspired color compensation for underwater images"""

    def __init__(self, in_channels):
        super().__init__()

        # Learnable color channel gains (R, G, B)
        # 红色衰减最快，需要更大的补偿；绿色居中；蓝色衰减最慢
        self.channel_gain = nn.Parameter(torch.tensor([1.5, 1.0, 0.8]))

        # Adaptive white balance estimator
        self.wb_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, in_channels, 1),
            nn.Softplus()  # Ensure positive gains
        )

        # Color correction network
        self.color_correct = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),  # Depthwise
            nn.Conv2d(in_channels, in_channels, 1),  # Pointwise
            nn.BatchNorm2d(in_channels),
        )

        # Residual scaling factor (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        B, C, H, W = x.shape

        # Estimate adaptive channel-wise gains
        channel_gains = self.wb_estimator(x)  # [B, C, 1, 1]

        # Apply channel-wise compensation
        # 假设前3个通道是RGB (如果是SAM特征，则作用于所有通道)
        compensated = x * channel_gains

        # Clamp to reasonable range to avoid extreme values
        compensated = torch.clamp(compensated, -3, 3)

        # Further color correction through convolution
        corrected = self.color_correct(compensated)

        # Residual connection with learnable scaling
        out = x + corrected * self.alpha

        return out

class LearnableDWT(nn.Module):
    """可学习的离散小波变换层"""

    def __init__(self, in_channels, wavelet_type='haar', learnable=True):
        super().__init__()
        self.in_channels = in_channels
        self.learnable = learnable

        if wavelet_type == 'haar' and not learnable:
            lpf = torch.tensor([1.0, 1.0]) / torch.sqrt(torch.tensor(2.0))
            hpf = torch.tensor([-1.0, 1.0]) / torch.sqrt(torch.tensor(2.0))
        else:
            lpf = torch.randn(2) * 0.1 + torch.tensor([0.5, 0.5])
            hpf = torch.randn(2) * 0.1 + torch.tensor([-0.5, 0.5])
            lpf = lpf / torch.norm(lpf)
            hpf = hpf / torch.norm(hpf)

        if learnable:
            self.low_pass = nn.Parameter(lpf)
            self.high_pass = nn.Parameter(hpf)
        else:
            self.register_buffer('low_pass', lpf)
            self.register_buffer('high_pass', hpf)

        self._build_wavelet_kernels()

    def _build_wavelet_kernels(self):
        ll_kernel = torch.outer(self.low_pass, self.low_pass)
        lh_kernel = torch.outer(self.low_pass, self.high_pass)
        hl_kernel = torch.outer(self.high_pass, self.low_pass)
        hh_kernel = torch.outer(self.high_pass, self.high_pass)

        wavelet_kernel = torch.stack([ll_kernel, lh_kernel, hl_kernel, hh_kernel])
        wavelet_kernel = wavelet_kernel.unsqueeze(1)

        self.wavelet_conv = nn.Conv2d(
            self.in_channels, 4 * self.in_channels,
            kernel_size=2, stride=2, groups=self.in_channels, bias=False
        )

        with torch.no_grad():
            self.wavelet_conv.weight.data = wavelet_kernel.repeat(
                self.in_channels, 1, 1, 1
            )

        if not isinstance(self.low_pass, nn.Parameter):
            self.wavelet_conv.weight.requires_grad = False

    def forward(self, x):
        x_dwt = self.wavelet_conv(x)
        batch_size, _, h, w = x_dwt.shape
        x_dwt = x_dwt.view(batch_size, 4, self.in_channels, h, w)
        ll, lh, hl, hh = x_dwt[:, 0], x_dwt[:, 1], x_dwt[:, 2], x_dwt[:, 3]
        return ll, lh, hl, hh


class LearnableIDWT(nn.Module):
    """可学习的逆离散小波变换层"""

    def __init__(self, in_channels, wavelet_type='haar', learnable=True):
        super().__init__()
        self.in_channels = in_channels

        if wavelet_type == 'haar' and not learnable:
            ilpf = torch.tensor([1.0, 1.0]) / torch.sqrt(torch.tensor(2.0))
            ihpf = torch.tensor([1.0, -1.0]) / torch.sqrt(torch.tensor(2.0))
        else:
            ilpf = torch.randn(2) * 0.1 + torch.tensor([0.5, 0.5])
            ihpf = torch.randn(2) * 0.1 + torch.tensor([0.5, -0.5])
            ilpf = ilpf / torch.norm(ilpf)
            ihpf = ihpf / torch.norm(ihpf)

        if learnable:
            self.inv_low_pass = nn.Parameter(ilpf)
            self.inv_high_pass = nn.Parameter(ihpf)
        else:
            self.register_buffer('inv_low_pass', ilpf)
            self.register_buffer('inv_high_pass', ihpf)

        self._build_inv_wavelet_kernels()

    def _build_inv_wavelet_kernels(self):
        ill_kernel = torch.outer(self.inv_low_pass, self.inv_low_pass)
        ilh_kernel = torch.outer(self.inv_low_pass, self.inv_high_pass)
        ihl_kernel = torch.outer(self.inv_high_pass, self.inv_low_pass)
        ihh_kernel = torch.outer(self.inv_high_pass, self.inv_high_pass)

        inv_kernel = torch.stack([ill_kernel, ilh_kernel, ihl_kernel, ihh_kernel])
        inv_kernel = inv_kernel.unsqueeze(1)

        self.idwt_conv = nn.ConvTranspose2d(
            4 * self.in_channels, self.in_channels,
            kernel_size=2, stride=2, groups=self.in_channels, bias=False
        )

        with torch.no_grad():
            self.idwt_conv.weight.data = inv_kernel.repeat(
                self.in_channels, 1, 1, 1
            )

        if not isinstance(self.inv_low_pass, nn.Parameter):
            self.idwt_conv.weight.requires_grad = False

    def forward(self, ll, lh, hl, hh):
        batch_size, _, h, w = ll.shape
        x_cat = torch.cat([ll, lh, hl, hh], dim=1)
        x_recon = self.idwt_conv(x_cat, output_size=(
            batch_size, self.in_channels, h * 2, w * 2
        ))
        return x_recon


class GatedDynamicDecoupler(nn.Module):
    """门控动态解耦器 - 对小波变换后的特征进行内容敏感的处理"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 为低频分量生成门控
        self.low_gate_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        # 为高频分量生成门控 (针对3个高频子带)
        self.high_gate_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3 * channels, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 3 * channels, 1),
            nn.Sigmoid()
        )

        # 初始化门控权重
        self._initialize_weights()

    def _initialize_weights(self):
        # 低频门控初始化为接近1，保护结构信息
        for m in self.low_gate_gen.modules():
            if isinstance(m, nn.Conv2d):
                if m.out_channels == self.channels:  # 最后一层
                    nn.init.constant_(m.weight, 0.1)
                    nn.init.constant_(m.bias, 0.8)  # 偏向于1

    def forward(self, ll, lh, hl, hh):
        # 对低频分量应用门控
        low_gate = self.low_gate_gen(ll)
        ll_gated = ll * low_gate

        # 将高频分量拼接后应用门控
        high_combined = torch.cat([lh, hl, hh], dim=1)
        high_gate = self.high_gate_gen(high_combined)
        high_gated = high_combined * high_gate

        # 分离门控后的高频分量
        B, _, H, W = lh.shape
        lh_gated, hl_gated, hh_gated = torch.split(high_gated, self.channels, dim=1)

        return ll_gated, lh_gated, hl_gated, hh_gated


class FrequencyPromptModule(nn.Module):
    """频域提示模块 - 为高低频特征注入可学习的提示信息"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 低频提示生成器 - 关注全局结构
        self.low_prompt_gen = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

        # 高频提示生成器 - 关注边缘细节 (针对每个高频子带)
        self.high_prompt_gen = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),  # 深度卷积
                nn.BatchNorm2d(channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1),
                nn.Sigmoid()
            ) for _ in range(3)  # LH, HL, HH
        ])

        # 跨频带交互模块
        self.cross_freq_interaction = nn.Sequential(
            nn.Conv2d(4 * channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout(0.1)  # 轻微正则化
        )

    def forward(self, ll, lh, hl, hh):
        # 生成低频提示
        low_prompt = self.low_prompt_gen(ll) * 0.5
        ll_prompted = ll * (1 + low_prompt)  # 加性提示

        # 生成高频提示
        high_feats = [lh, hl, hh]
        high_prompted = []
        for i, feat in enumerate(high_feats):
            high_prompt = self.high_prompt_gen[i](feat)
            prompted_feat = feat * (1 + high_prompt)  # 加性提示
            high_prompted.append(prompted_feat)

        lh_prompted, hl_prompted, hh_prompted = high_prompted

        # 跨频带交互 - 融合不同频带的信息
        all_feats = torch.cat([ll_prompted, lh_prompted, hl_prompted, hh_prompted], dim=1)
        cross_feat = self.cross_freq_interaction(all_feats)

        # 将交互信息加回到各个频带
        ll_final = ll_prompted + cross_feat * 0.3
        lh_final = lh_prompted + cross_feat * 0.7
        hl_final = hl_prompted + cross_feat * 0.7
        hh_final = hh_prompted + cross_feat * 0.7

        return ll_final, lh_final, hl_final, hh_final


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ImprovedWaveletRFB(nn.Module):
    """改进的基于小波变换的RFB模块 - 集成门控机制和频域提示"""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = nn.ReLU(True)

        # 初始特征变换
        self.conv1 = BasicConv2d(in_channel, out_channel, 1)

        # 小波变换层
        self.dwt = LearnableDWT(out_channel, learnable=True)
        self.idwt = LearnableIDWT(out_channel, learnable=True)

        # 门控动态解耦器
        self.gated_decoupler = GatedDynamicDecoupler(out_channel)

        # 频域提示模块
        self.freq_prompt = FrequencyPromptModule(out_channel)

        # 子带特异性处理模块
        self. low_enhancer = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

        self.high_enhancers = nn.ModuleList([
            nn.Sequential(
                BasicConv2d(out_channel, out_channel, 3, padding=1, dilation=1),
                BasicConv2d(out_channel, out_channel, 3, padding=1, dilation=1)
            ),
            nn.Sequential(
                BasicConv2d(out_channel, out_channel, 3, padding=2, dilation=2),
                BasicConv2d(out_channel, out_channel, 3, padding=2, dilation=2)
            ),
            nn.Sequential(
                BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
                BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
            )
        ])

        # 残差连接
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

        # 最终融合
        self.final_conv = BasicConv2d(out_channel, out_channel, 3, padding=1)

    def forward(self, x):
        # 初始特征变换
        x_conv = self.conv1(x)

        # 小波变换分解
        ll, lh, hl, hh = self.dwt(x_conv)

        # 应用门控动态解耦
        ll_gated, lh_gated, hl_gated, hh_gated = self.gated_decoupler(ll, lh, hl, hh)

        # 应用频域提示
        ll_prompted, lh_prompted, hl_prompted, hh_prompted = self.freq_prompt(
            ll_gated, lh_gated, hl_gated, hh_gated
        )

        # 子带特异性增强
        ll_enhanced = self.low_enhancer(ll_prompted)

        high_feats = [lh_prompted, hl_prompted, hh_prompted]
        high_enhanced = []
        for i, feat in enumerate(high_feats):
            enhanced = self.high_enhancers[i](feat)
            high_enhanced.append(enhanced)

        lh_enhanced, hl_enhanced, hh_enhanced = high_enhanced

        # 小波逆变换重建
        x_recon = self.idwt(ll_enhanced, lh_enhanced, hl_enhanced, hh_enhanced)

        # 残差连接
        res = self.conv_res(x)
        if x_recon.shape[2:] != res.shape[2:]:
            x_recon = F.interpolate(x_recon, size=res.shape[2:], mode="bilinear", align_corners=False)

        # 最终输出
        out = self.relu(x_recon + res)
        out = self.final_conv(out)

        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

class TextEncoder(nn.Module):
    """将类别标签映射为嵌入向量"""
    def __init__(self, num_classes, embed_dim=64):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, labels):
        # labels: (B,) 类别索引
        labels = labels.to(self.embed.weight.device)
        emb = self.embed(labels)          # (B, embed_dim)
        emb = self.fc(emb)                # (B, embed_dim)
        return emb

class LDWTUNet(nn.Module):
    def __init__(self, checkpoint_path=None, num_classes=7, text_embed_dim=64):
        super(LDWTUNet, self).__init__()
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk
        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )

        self.uccb1 = UnderwaterColorCompensation(144)
        self.uccb2 = UnderwaterColorCompensation(288)
        self.uccb3 = UnderwaterColorCompensation(576)
        self.uccb4 = UnderwaterColorCompensation(1152)

        # 使用改进的基于小波变换的RFB模块
        self.rfb1 = ImprovedWaveletRFB(144, 64)
        self.rfb2 = ImprovedWaveletRFB(288, 64)
        self.rfb3 = ImprovedWaveletRFB(576, 64)
        self.rfb4 = ImprovedWaveletRFB(1152, 64)

        self.up1 = (Up(128, 64))
        self.up2 = (Up(128, 64))
        self.up3 = (Up(128, 64))
        self.up4 = (Up(128, 64))
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

        self.text_encoder = TextEncoder(num_classes, text_embed_dim)
        # 将文本嵌入投影到与RFB输出相同的通道数（64）
        self.text_proj = nn.Linear(text_embed_dim, 64)
        # 在每个上采样块之后增加融合层（将文本特征与特征图拼接并压缩回64通道）
        self.fusion_up1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.fusion_up2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.fusion_up3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, labels):
        B = x.size(0)
        x1, x2, x3, x4 = self.encoder(x)
        # 先应用UCCB进行颜色补偿，再应用RFB
        x1 = self.uccb1(x1)
        x2 = self.uccb2(x2)
        x3 = self.uccb3(x3)
        x4 = self.uccb4(x4)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        text_feat = self.text_encoder(labels)          # (B, text_embed_dim)
        text_feat = self.text_proj(text_feat)          # (B, 64)
        text_feat = text_feat.unsqueeze(-1).unsqueeze(-1)
        x = self.up1(x4, x3)                           # 输出 [B, 64, H/4, W/4] (假设原始H,W为352，此处约为88)
        text_tile = text_feat.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, text_tile], dim=1)           # [B, 128, H/4, W/4]
        x = self.fusion_up1(x)                         # [B, 64, H/4, W/4]
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')

        # up2 阶段 (x, x2)
        x = self.up2(x, x2)                            # [B, 64, H/2, W/2]
        text_tile = text_feat.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, text_tile], dim=1)           # [B, 128, H/2, W/2]
        x = self.fusion_up2(x)                         # [B, 64, H/2, W/2]
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')

        # up3 阶段 (x, x1)
        x = self.up3(x, x1)                            # [B, 64, H, W]
        text_tile = text_feat.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, text_tile], dim=1)           # [B, 128, H, W]
        x = self.fusion_up3(x)                         # [B, 64, H, W]
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')   # 最终输出与原始分辨率一致

        return out, out1, out2


if __name__ == "__main__":
    with torch.no_grad():
        model = LDWTUNet().cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        out, out1, out2 = model(x)
        print(out.shape, out1.shape, out2.shape)