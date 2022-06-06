import torch
import math
from typing import Optional
from einops import reduce, rearrange
import numpy as np



def img2mse(img_src, img_tgt):
    return torch.nn.functional.mse_loss(img_src, img_tgt)


def mse2psnr(mse):
    # For numerical stability, avoid a zero mse loss.
    if mse == 0:
        mse = 1e-5
    return -10.0 * math.log10(mse)


def meshgrid_xy(tensor1: torch.Tensor, tensor2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.

    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod
def PE_and_gather(pts, rd, pts_PE_fn, rd_PE_fn):
    pts_PE = pts_PE_fn(pts.reshape((-1, pts.shape[-1]))) # 2048*64x63->포지셔널 임베딩 후 각 좌표의 차원 증가3차원에서 63차원 좌표로
    #pts_flat_canon = None
    
              
    #viewdirs = ray_batch[..., None, -3:]# rd를 지칭 torch.Size([2048, 1, 3])
    rd_expanded = rd.expand(pts.shape)# torch.Size([2048, 64, 3])#각 레이마다 포인트가 64개라서 똑같이 복사시켜줘야함
    #input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))#torch.Size([131072, 3])
    rd_PE = rd_PE_fn(rd_expanded.reshape((-1, rd_expanded .shape[-1]))) # torch.Size([131072, 24])24차원 좌표로
    input_PE = torch.cat((pts_PE, rd_PE), dim=-1)#torch.Size([131072, 87]) X 63개/ D 24개 큐브뎁스가 64고 63은 X의 PE임

    return input_PE

def stratified_sampling(ray_info, n_p_samples):
   
    num_rays = ray_info.shape[0]
    co   = ray_info[..., :3]
    rd   = ray_info[..., 3:6].clone() 
    near = ray_info[..., -2].reshape(-1,1)
    far  = ray_info[..., -1].reshape(-1,1)
    #import pdb; pdb.set_trace()
    t_vals = torch.linspace(#near far bound 사이에서 몇개를 볼것이냐
        0.0,
        1.0,
        n_p_samples,
        dtype=co.dtype,
        device=co.device,
    )#보고싶은 point sample의 수==64개

    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand([num_rays, n_p_samples])#2048x64


    
        # Get intervals between samples.
    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
    lower = torch.cat((z_vals[..., :1], mids), dim=-1)
    # Stratified samples in those intervals.
    t_rand = torch.rand(z_vals.shape, dtype=co.dtype, device=co.device)
    z_vals = lower + (upper - lower) * t_rand#ray가 얼마나 진행할 것이냐에 대한 값
    # pts -> (num_rays, N_samples, 3)
    pts = co[..., None, :] + rd[..., None, :] * z_vals[..., :, None]#2048x64x3 평면x깊이x좌표차원(3) 큐브의 좌표를 생각해보자
    

    return pts, z_vals

def canon_ray(height: int, width: int, intrinsics , tform_cam2world: torch.Tensor, center = [0.5,0.5]):

    # TESTED
    ii, jj = meshgrid_xy(
        torch.arange(
            width, dtype=tform_cam2world.dtype, device=tform_cam2world.device
        ).to(tform_cam2world),
        torch.arange(
            height, dtype=tform_cam2world.dtype, device=tform_cam2world.device
        ),
    )#좌표 
    
    if intrinsics.shape<(4,):
        intrinsics = [intrinsics, intrinsics, 0.5, 0.5]
    directions = torch.stack(#img2cam #normalized image plane으로 바꿔줌
        [
            (ii - width * intrinsics[2]) / intrinsics[0],
            -(jj - height * intrinsics[3]) / intrinsics[1],
            -torch.ones_like(ii),
        ],
        dim=-1,
    #camera @ origin/ image_plane @ z=-1
    )#이미지 평면에서의 좌표는 좌상단 기준 #픽셀상 위치랑 카메라 인트린직 알면 카메라에서 픽셀상의 위치를 볼때의 좌표값을 얻을 수 있음.
    
    ray_directions = torch.sum(
        directions[..., None, :] * tform_cam2world[:3, :3], dim=-1
    )#여기에 cam 2 world곱해서 물체의 중심 입장에서 보는 ray direction을 알 수 있음. 실제로 모델에 넣을때는 좌표계가 바뀌면 안되기 때문에 이렇게 넣음
    #print(ray_directions)
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)#translation만 알면됨.
    return ray_origins, ray_directions



def positional_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True) -> torch.Tensor:
    r"""Apply positional encoding to the input.

    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def get_embedding_function(num_encoding_functions=6, include_input=True, log_sampling=True):
    r"""Returns a lambda function that internally calls positional_encoding.
    """
    return lambda x: positional_encoding(x, num_encoding_functions, include_input, log_sampling)



def gather_cdf_util(cdf, inds):
    r"""A very contrived way of mimicking a version of the tf.gather()
    call used in the original impl.
    """
    orig_inds_shape = inds.shape
    inds_flat = [inds[i].view(-1) for i in range(inds.shape[0])]
    valid_mask = [
        torch.where(ind >= cdf.shape[1], torch.zeros_like(ind), torch.ones_like(ind))
        for ind in inds_flat
    ]
    inds_flat = [
        torch.where(ind >= cdf.shape[1], (cdf.shape[1] - 1) * torch.ones_like(ind), ind)
        for ind in inds_flat
    ]
    cdf_flat = [cdf[i][ind] for i, ind in enumerate(inds_flat)]
    cdf_flat = [cdf_flat[i] * valid_mask[i] for i in range(len(cdf_flat))]
    cdf_flat = [
        cdf_chunk.reshape([1] + list(orig_inds_shape[1:])) for cdf_chunk in cdf_flat
    ]
    return torch.cat(cdf_flat, dim=0)


def sample_pdf(bins, weights, num_samples, det=False):
    r"""sample_pdf function from another concurrent pytorch implementation
    by yenchenlin (https://github.com/yenchenlin/nerf-pytorch).
    """
    #bins = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
    weights = weights + 1e-5#weight의 제일 앞 뒤쪽 값은 빼서 torch.Size([2048, 62])
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)#torch.Size([2048, 62])
    cdf = torch.cumsum(pdf, dim=-1)#torch.Size([2048, 62])
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
    )  # (batchsize, len(bins))
    #torch.Size([2048, 64])

    # Take uniform samples
    if det:
        u = torch.linspace(
            0.0, 1.0, steps=num_samples, dtype=weights.dtype, device=weights.device
        )
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        u = torch.rand(
            list(cdf.shape[:-1]) + [num_samples],
            dtype=weights.dtype,
            device=weights.device,
        )

    # Invert CDF
    u = u.contiguous()
    cdf = cdf.contiguous()
    #inds = torchsearchsorted.searchsorted(cdf, u, side="right")
    inds = torch.searchsorted(cdf.detach(), u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), dim=-1)  # (batchsize, num_samples, 2)

    matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def volume_rendering(radiance_field, depth_values, ray_directions, 
                                radiance_field_noise_std=0.0, background_prior = None):
    # TESTED
    #import pdb; pdb.set_trace()
    #TODO for coarse
    #torch.Size([2048, 64, 4])->radiance field RGB랑 sigma가 있으니까
    inf = torch.tensor([1e10], dtype=ray_directions.dtype, device=ray_directions.device)
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],#TODO 아 이렇게하면 뺼셈이라 거리가 하나씩나오네
            inf.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)#torch.Size([2048, 64])

    if background_prior is not None:
        rgb = torch.sigmoid(radiance_field[:, :-1, :3])
        rgb = torch.cat((rgb, radiance_field[:, -1, :3].unsqueeze(1)), dim=1)#마지막꺼는 bggrnd라서 그냥 cat
    else:
        rgb = torch.sigmoid(radiance_field[..., :3])
    
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., 3].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
    # TODO sigma_a [-0.04 ~ -0.01]
    # TODO noise [-0.4 ~ 0.4]

    sigmas = torch.nn.functional.relu(radiance_field[..., 3] + noise)
    #TODO relu태워서 0~0.4에 있음
    sigmas[:,-1] += 1e-6 # todo commented this for FCB demo !!!!!!#torch.Size([2048, 64])
    alpha = 1.0 - torch.exp(-sigmas * dists) #TODO alphavalue로 얼마나 투명하지 않은지? torch.Size([2048, 64])
    transmittance = cumprod_exclusive(1.0 - alpha + 1e-10)
    weights = alpha * transmittance #TODO alpha*transmittance torch.Size([2048, 64]) Alpha composition
    
    
    rgb_map = weights[..., None] * rgb #torch.Size([2048, 64, 3])
    rgb_map = rgb_map.sum(dim=-2) #TODO 레이에 있는 64개의 point에 대해서 다 sum -> torch.Size([2048, 3])
    #depth_map = 
    depth_map = (weights * depth_values).sum(dim=-1)
    
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / (depth_map / acc_map)

    return rgb_map, disp_map, acc_map, weights, depth_map

def ray_sampling(grid, prob_map, n_samples, GT, bg, cam_o, r_d):

    ray_sample_locs = np.random.choice(grid.shape[0], size=(n_samples), \
                replace=False, p=prob_map)#2048x2 == ray sample곱하기 좌표xy
    ray_sample_locs = grid[ray_sample_locs]#2048x2 로 좌표값들이 총 2048개
    cam_o = cam_o[ray_sample_locs[:, 0], ray_sample_locs[:, 1], :]#world 입장에서 봤을때 카메라가 나랑 얼마나 떨어져?
    #pose의 translation 3개 값들이 2048로 동일하게 들어가 있음
    r_d = r_d[ray_sample_locs[:, 0], ray_sample_locs[:, 1], :]

    GT_color = GT[ray_sample_locs[:, 0], ray_sample_locs[:, 1], :]
    bg_color = bg[ray_sample_locs[:, 0], ray_sample_locs[:, 1], :]

    return GT_color, bg_color, cam_o, r_d



def torch_normal_map(depthmap,focal,weights=None,clean=True, central_difference=False):
    W,H = depthmap.shape
    #normals = torch.zeros((H,W,3), device=depthmap.device)
    cx = focal[2]*W
    cy = focal[3]*H
    fx = focal[0]
    fy = focal[1]
    ii, jj = meshgrid_xy(torch.arange(W, device=depthmap.device),
                         torch.arange(H, device=depthmap.device))
    points = torch.stack(
        [
            ((ii - cx) * depthmap) / fx,
            -((jj - cy) * depthmap) / fy,
            depthmap,
        ],
        dim=-1)
    difference = 2 if central_difference else 1
    dx = (points[difference:,:,:] - points[:-difference,:,:])
    dy = (points[:,difference:,:] - points[:,:-difference,:])
    normals = torch.cross(dy[:-difference,:,:],dx[:,:-difference,:],2)
    normalize_factor = torch.sqrt(torch.sum(normals*normals,2))
    normals[:,:,0]  /= normalize_factor
    normals[:,:,1]  /= normalize_factor
    normals[:,:,2]  /= normalize_factor
    normals = normals * 0.5 +0.5

    if clean and weights is not None: # Use volumetric rendering weights to clean up the normal map
        mask = weights.repeat(3,1,1).permute(1,2,0)
        mask = mask[:-difference,:-difference]
        where = torch.where(mask > 0.22)
        normals[where] = 1.0
        normals = (1-mask)*normals + (mask)*torch.ones_like(normals)
    normals *= 255
    #plt.imshow(normals.cpu().numpy().astype('uint8'))
    #plt.show()
    return normals

if __name__ == "__main__":
    pass