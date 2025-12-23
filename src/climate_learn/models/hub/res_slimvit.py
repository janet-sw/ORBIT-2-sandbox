# Local application
from .components.cnn_blocks import PeriodicConv2D
from .components.pos_embed import get_2d_sincos_pos_embed
from .utils import register

# Third party
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_
from einops import rearrange

@register("res_slimvit")
class Res_Slim_ViT(nn.Module):
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        history,
        superres_factor = 4,
        cnn_ratio = 4,
        patch_size=16,
        drop_path=0.1,
        drop_rate=0.1,
        learn_pos_emb=False,
        embed_dim=1024,
        depth=24,
        decoder_depth=8,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.cnn_ratio = cnn_ratio
        self.superres_factor = superres_factor
        self.in_channels = in_channels * history
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(img_size, patch_size, self.in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=learn_pos_emb
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    proj_drop=drop_rate,
                    attn_drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        #skip connection path
        self.path2 = nn.ModuleList()
        # self.path2.append(nn.Conv2d(in_channels=in_channels, out_channels=cnn_ratio*superres_factor*superres_factor, kernel_size=(3, 3), stride=1, padding=1)) 
        self.path2.append(nn.Conv2d(in_channels=self.in_channels, out_channels=cnn_ratio*superres_factor*superres_factor, kernel_size=(3, 3), stride=1, padding=1)) ### should work for both downscaling and forecasting
        self.path2.append(nn.GELU())
        self.path2.append(nn.PixelShuffle(superres_factor))
        self.path2.append(nn.Conv2d(in_channels=cnn_ratio, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1)) 
        self.path2 = nn.Sequential(*self.path2)


        #vit path
        self.path1 = nn.ModuleList()
        self.path1.append(nn.Conv2d(in_channels=out_channels, out_channels=cnn_ratio*superres_factor*superres_factor, kernel_size=(3, 3), stride=1, padding=1)) 
        self.path1.append(nn.GELU())
        self.path1.append(nn.PixelShuffle(superres_factor))
        self.path1.append(nn.Conv2d(in_channels=cnn_ratio, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1)) 
        self.path1 = nn.Sequential(*self.path1)


        self.to_img = nn.Linear(embed_dim, out_channels * patch_size**2)

        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(self.img_size[1]*superres_factor, self.img_size[1]*superres_factor))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(self.img_size[1]*superres_factor, self.img_size[1]*superres_factor))
        self.head = nn.Sequential(*self.head)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x: torch.Tensor, scaling =1):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = self.out_channels
        h = self.img_size[0] * scaling // p
        w = self.img_size[1] *scaling // p
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_encoder(self, x: torch.Tensor):
        # x.shape = [B,C,H,W]
        x = self.patch_embed(x)
        # x.shape = [B,num_patches,embed_dim]

        #if torch.distributed.get_rank()==0:
        #    print("after patch_embed x.shape",x.shape,flush=True)


        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        # x.shape = [B,num_patches,embed_dim]
        x = self.norm(x)
        return x

    def forward(self, x):
        if len(x.shape) == 5:  # x.shape = [B,T,in_channels,H,W]
            x = x.flatten(1, 2)
        # x.shape = [B,T*in_channels,H,W]

 
        path2_result = self.path2(x)
        
        x = self.forward_encoder(x)

        # x.shape = [B,num_patches,embed_dim]
        #x = self.head(x)


        x = self.to_img(x) 
        # x.shape = [B,num_patches,out_channels*patch_size*patch_size]
        x = self.unpatchify(x)
        # x.shape = [B,num_patches,h*patch_size, w*patch_size]
   
        preds = self.path1(x) + path2_result

        #decoder
        preds = self.head(preds) 
        # preds.shape = [B,out_channels,H,W]
        return preds
