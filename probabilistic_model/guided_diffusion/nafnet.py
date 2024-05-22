import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .nn import timestep_embedding


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
        ) if time_emb_dim else None

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, x):
        inp, time = x
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

        x = inp

        x = self.norm1(x)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x, time


class ConditionalNAFNet(nn.Module):
    def __init__(
        self, 
        img_channel=3, 
        width=16, 
        middle_blk_num=1, 
        enc_blk_nums=[], 
        dec_blk_nums=[], 
        upscale=1,
        lq_cannel=None
    ):
        super().__init__()
        self.upscale = upscale
        self.model_channels = width
        # sinu_pos_emb = SinusoidalPosEmb(model_channels)
        time_dim = width * 4

        self.time_embed = nn.Sequential(
            # sinu_pos_emb,
            nn.Linear(self.model_channels, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )

        self.intro = nn.Conv2d(in_channels=img_channel*2, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        # self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
        #                       bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel*2, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, time_dim) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, x, timesteps, cond):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        x = torch.cat([x, cond], dim=1)

        B, C, H, W = x.shape
        x = self.check_image_size(x)

        x = self.intro(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x, _ = encoder([x, emb])
            encs.append(x)
            x = down(x)

        x, _ = self.middle_blks([x, emb])

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x, _ = decoder([x, emb])

        x = self.ending(x)

        x = x[..., :H, :W]

        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

##############################################################################
######### condition feature ##################################################
    
from guided_diffusion.condition_module import EncoderUNetModelWT



class NAFBlockCondFet(nn.Module):
    def __init__(self, c, time_emb_dim=None, cond_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
        ) if time_emb_dim else None

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.concat_cond = nn.Conv2d(in_channels=c+cond_dim, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, x):
        inp, time, cond = x
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

        x = self.concat_cond(torch.cat((inp, cond),dim=1))

        x = self.norm1(x)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x, time, cond
    
class ConditionalNAFNetCondFet(nn.Module):
    def __init__(
        self, 
        img_channel=3, 
        width=16, 
        middle_blk_num=1, 
        enc_blk_nums=[], 
        dec_blk_nums=[], 
        upscale=1,
        cond_dim=32,
        lq_cannel=3,
        cond_channels=256,
    ):
        super().__init__()
        self.upscale = upscale
        self.model_channels = width
        # sinu_pos_emb = SinusoidalPosEmb(model_channels)
        time_dim = width * 4

        self.time_embed = nn.Sequential(
            # sinu_pos_emb,
            nn.Linear(self.model_channels, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel*2, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlockCondFet(chan, time_dim, cond_dim) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlockCondFet(chan, time_dim, cond_dim) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlockCondFet(chan, time_dim, cond_dim) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)   

        self.make_condition_feature = EncoderUNetModelWT(
            image_size=None,
            in_channels=lq_cannel,
            model_channels=cond_channels,
            out_channels=cond_dim,
            num_res_blocks=2,
            dropout=0,
            channel_mult=(1, 2, 4, 8, 16),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
        )
        
        
    def create_condition(self, y, t_replace):  #y=torch.Size([batch_size, burst_size, 4, 48, 48])
        y = self.make_condition_feature(y, t_replace)  
        return y

    def forward(self, x, timesteps, cond):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        cond = self.create_condition(cond, timesteps)

        B, C, H, W = x.shape
        x = self.check_image_size(x)

        x = self.intro(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x, _, _ = encoder([x, emb, cond[str(x.shape[3])]])
            encs.append(x)
            x = down(x)

        x, _, _ = self.middle_blks([x, emb, cond[str(x.shape[3])]])

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x, _, _ = decoder([x, emb, cond[str(x.shape[3])]])

        x = self.ending(x)

        x = x[..., :H, :W]

        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x




class ConditionalNAFNetCondFet_(nn.Module):
    def __init__(
        self, 
        img_channel=3, 
        width=16, 
        middle_blk_num=1, 
        enc_blk_nums=[], 
        dec_blk_nums=[], 
        upscale=1,
        cond_dim=32,
        lq_cannel=3,
    ):
        super().__init__()
        self.upscale = upscale
        self.model_channels = width
        # sinu_pos_emb = SinusoidalPosEmb(model_channels)
        time_dim = width * 4

        self.time_embed = nn.Sequential(
            # sinu_pos_emb,
            nn.Linear(self.model_channels, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel*2, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlockCondFet(chan, time_dim, cond_dim) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlockCondFet(chan, time_dim, cond_dim) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlockCondFet(chan, time_dim, cond_dim) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)   

        self.make_condition_feature = EncoderUNetModelWT(
            image_size=None,
            in_channels=lq_cannel,
            model_channels=128,
            out_channels=cond_dim,
            num_res_blocks=2,
            dropout=0,
            channel_mult=(1, 2, 4, 8, 16),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
        )
        
        
    def create_condition(self, y, t_replace):  #y=torch.Size([batch_size, burst_size, 4, 48, 48])
        y = self.make_condition_feature(y, t_replace)  
        return y

    def forward(self, x, timesteps, cond):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        cond = self.create_condition(cond, timesteps)

        B, C, H, W = x.shape
        x = self.check_image_size(x)

        x = self.intro(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x, _, _ = encoder([x, emb, cond[str(x.shape[3])]])
            encs.append(x)
            x = down(x)

        x, _, _ = self.middle_blks([x, emb, cond[str(x.shape[3])]])

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x, _, _ = decoder([x, emb, cond[str(x.shape[3])]])

        x = self.ending(x)

        x = x[..., :H, :W]

        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x