import torch
import torch.nn as nn


class Nerface_no_expr(torch.nn.Module):

    def __init__(self, num_encoding_fn_xyz=6, num_encoding_fn_dir=4, include_input_dir=False):
        super(Nerface_no_expr, self).__init__()

        include_input_xyz = 3
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 64

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz # 63
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir # 24
        self.dim_expression = include_expression # + 2 * 3 * num_encoding_fn_expr
        self.dim_latent_code = 32

        self.layers_xyz = torch.nn.ModuleList()
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz  + self.dim_latent_code, 256))
        for i in range(1, 6):
            if i == 3:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_latent_code + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x,  expr=None, latent_code=None, **kwargs):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        #print(x.shape, expr.shape, latent_code.shape)
        x = xyz
        latent_code = latent_code.repeat(xyz.shape[0], 1)
        if self.dim_expression > 0:
            #expr_encoding = (expr * 1 / 3).repeat(xyz.shape[0], 1)
            initial = torch.cat((xyz, latent_code), dim=1)
            x = initial
        for i in range(6):
            if i == 3:
                x = self.layers_xyz[i](torch.cat((initial, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        
        x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)

class Nerface(torch.nn.Module):

    def __init__(self, num_encoding_fn_xyz=6, num_encoding_fn_dir=4, include_input_dir=False):
        super(Nerface, self).__init__()

        include_input_xyz = 3
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 76#TODO 이거 64로 바꿔야함

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz # 63
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir # 24
        self.dim_expression = include_expression # + 2 * 3 * num_encoding_fn_expr
        self.dim_latent_code = 32

        self.layers_xyz = torch.nn.ModuleList()
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code, 256))
        for i in range(1, 6):
            if i == 3:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x,  expr=None, latent_code=None, **kwargs):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        #print(x.shape, expr.shape, latent_code.shape)
        #import pdb; pdb.set_trace()
        x = xyz
        latent_code = latent_code.repeat(xyz.shape[0], 1) ##torch.Size([65536, 32])
        if self.dim_expression > 0:
            expr_encoding = (expr * 1 / 3).repeat(xyz.shape[0], 1)
            initial = torch.cat((xyz, expr_encoding, latent_code), dim=1)
            x = initial
        for i in range(6):
            if i == 3:
                x = self.layers_xyz[i](torch.cat((initial, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        
        x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)


class Nerface_trans_opt(torch.nn.Module):

    def __init__(self, num_encoding_fn_xyz=10, num_encoding_fn_dir=4, include_input_dir=False):
        super(Nerface_trans_opt, self).__init__()

        include_input_xyz = 3
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 76#TODO 이거 64로 바꿔야함

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz # 63
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir # 24
        self.dim_expression = include_expression # + 2 * 3 * num_encoding_fn_expr
        self.dim_latent_code = 32

        #self.layer_trans = 16

        self.layers_xyz = torch.nn.ModuleList()
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code, 256))
        for i in range(1, 6):
            if i == 3:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x,  expr=None, latent_trans=None,latent_code=None, **kwargs):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]#torch.Size([65536, 32])//torch.Size([65536, 24])
        #print(x.shape, expr.shape, latent_code.shape)
        x = xyz
        print(latent_trans,": Monitor trans")
        latent_code = latent_code.repeat(xyz.shape[0], 1)
        latent_trans = latent_trans.repeat(xyz.shape[0], 1)
        #XXX both are per -frame too~

        if self.dim_expression > 0:
            xyz = xyz + latent_trans
            dirs = dirs + latent_trans
            expr_encoding = (expr * 1 / 3).repeat(xyz.shape[0], 1)
            initial = torch.cat((xyz, expr_encoding, latent_code), dim=1)
            x = initial
        for i in range(6):
            if i == 3:
                x = self.layers_xyz[i](torch.cat((initial, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        
        x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)

if __name__=="__main__":
    #Nerface_trans_opt
    pass
