import torch

from models.nerface.base_function \
    import sample_pdf, volume_rendering, stratified_sampling, PE_and_gather


def PE_forwarding(nerf, pts, rd_r, chunksize, PE4xyz, PE4d, expressions = None, latent_code = None, latent_code_t=None, wild=False, latent_trans=None):

    #pts_flat = pts.reshape((-1, pts.shape[-1])) # 2048*64x3 flatening then sampling cube

    input_PE = PE_and_gather(pts, rd_r, PE4xyz, PE4d)

    input_batches = [input_PE [i : i + chunksize] for i in range(0, input_PE.shape[0], chunksize)] # batch하나당 2048,87 shape chunksize인데 random_sample사이즈랑 같음
    #TODO chunksize가 ray 갯수랑 같아서, 64개의 batch, ray가 하나의 배치라고 보는게 맞음
    #TODO it is for ablation
    if expressions is None:
        preds = [nerf(batch) for batch in input_batches]
    elif latent_code is None:
        preds = [nerf(batch, expressions) for batch in input_batches] # 한 이미지에 대한거라서 사이즈 안맞춰도댐
    else:
        preds = [nerf(batch, expressions, latent_code) for batch in input_batches]

    radiance_field = torch.cat(preds, dim=0).reshape((-1, pts.shape[1], 4))#131072x4==cubesize x rgba
 #networkfn(**params)
 

    return radiance_field

def rendering_flow(ray_info, model_coarse, model_fine, options, mode="train", encode_position_fn=None, 
                                encode_direction_fn=None, expressions = None, background_prior = None, latent_code = None, latent_code_t=None,wild = False,latent_trans=None):
    params = options.train_params if mode == 'train' else options.val_params
    #import pdb; pdb.set_trace()
    # TESTED
    
    co, rd = ray_info[..., :3], ray_info[..., 3:6] 
    rd_reshaped = rd[..., None, :]
    

    pts_coarse, z_vals = stratified_sampling(ray_info, options.model['num_coarse'])
   
    output_coarse = PE_forwarding(
        model_coarse, 
        pts_coarse, 
        rd_reshaped, 
        params['chunksize'],
        encode_position_fn, 
        encode_direction_fn, 
        expressions, 
        latent_code,
        latent_code_t, 
        wild,
        latent_trans)
    # TODO torch.Size([2048, 64, 4])

    # make last RGB values of each ray, the background
    if background_prior is not None:
        output_coarse[:,-1,:3] = background_prior
        #TODO 마지막의 point의 color를 그냥 background로 넣기
        #rgb_map_total, weights_total, sigma_t, beta_map_t
   
        
    (rgb_coarse, _,_, weights, _) = volume_rendering(
        output_coarse, z_vals, rd,
        radiance_field_noise_std=params['radiance_field_noise_std'],
        background_prior=background_prior)
    #TODO torch.Size([2048, 3]) torch.Size([2048]) torch.Size([2048]) torch.Size([2048, 64]) torch.Size([2048])
    #TODO 각 레이별 칼라(이미 integral됨)/  각 레이별 차이/             /각 포인트별 웨이트    
    #z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])#TODO 각 인터벌의 중간의 위치값이라 생각
    
    bins =  0.5 * (z_vals[..., 1:] + z_vals[..., :-1])

    z_vals_fine = sample_pdf(
        bins,
        weights[..., 1:-1],
        options.model['num_fine'],
        det=(options.model['perturb'] == 0.0)
        )
   
    z_vals_fine = z_vals_fine.detach()#TODO torch.Size([2048, 64])#If NO deterministic error
    #import pdb; pdb.set_trace()
    z_vals_full, _ = torch.sort(torch.cat((z_vals, z_vals_fine), dim=-1), dim=-1)
    pts_fine = co[..., None, :] + rd_reshaped * z_vals_full[..., :, None]
    #TODO torch.Size([2048, 128, 3])

    ##TODO from here model fine
    output_fine = PE_forwarding(
        model_fine,
        pts_fine,
        rd_reshaped,
        params['chunksize'],
        encode_position_fn,
        encode_direction_fn,
        expressions,
        latent_code,
        latent_code_t,
        wild
    )# XXX torch.Size([2048, 128, 4])
    # make last RGB values of each ray, the background
    if background_prior is not None:
        output_fine[:, -1, :3] = background_prior#이제 RGBA값임

   
        
    rgb_fine, disp_fine, acc_fine, weights, depth_fine = volume_rendering( # added use of weights
        output_fine,
        z_vals_full,
        rd,
        radiance_field_noise_std=params['radiance_field_noise_std'],
        background_prior=background_prior
    )
        #XXX RGB fine == torch.Size([2048, 3])
  
        ##여기부터 다시짜기
    if mode=='val':
        
        return rgb_fine, disp_fine, acc_fine, depth_fine, weights[:,-1] # changed last return val to fine_weights
    else:#마지막만 [:,-1]인 이유는 백그라운드 때문
        
        return rgb_coarse, rgb_fine, weights[:,-1]
        


def NerFACE(model_coarse, model_fine, co, rd, options, mode="train", 
                         encode_position_fn=None, encode_direction_fn=None, expressions=None, background_prior=None, latent_code=None, latent_code_t=None,wild=False,latent_trans=None):
    
    if mode=='val':
        reshaping_list = [
            rd.shape,#2048x3
            rd.shape[:-1],#2048
            rd.shape[:-1],
            rd.shape[:-1],
            rd.shape[:-1]
        ]
       
            
    else:
        reshaping_list = [
        rd.shape,
        rd.shape,
        rd.shape[:-1],
    ]

    
    #import pdb;pdb.set_trace()
    co = co.view((-1, 3))#torch.Size([2048, 3])
    rd = rd.view((-1, 3))#torch.Size([2048, 3])
        
    near = options.dataset['near'] * torch.ones_like(rd[..., :1])
    far = options.dataset['far'] * torch.ones_like(rd[..., :1])
    #TODO 아마도 z값만..? torch.Size([2048, 1]) 
    rays = torch.cat((co, rd, near, far), dim=-1)#dim=3,3,1,1
    
    if mode == 'train':#단위 ray를 하나의 데이터로 취급 
        chunksize = options.train_params['chunksize']
    else:
        chunksize = options.val_params['chunksize']
  
    pred = [
        rendering_flow(
            rays[i:i+chunksize],
            model_coarse,
            model_fine,
            options,
            mode,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            expressions = expressions,
            background_prior = background_prior[int(i/chunksize)] if background_prior is not None else background_prior,
            latent_code = latent_code,
            latent_code_t = latent_code_t,
            wild = wild,
            latent_trans = latent_trans
        )
        #for i,batch in enumerate(batches)
        for i in range(0, rays.shape[0], chunksize)
    ]
    #print([ i/chunksize for i in range(0, rays.shape[0], chunksize)])
    #import pdb;pdb.set_trace()
    #batches length==1, batches[0].shape == torch.Size([2048, 8])
    #XXX len(pred)==1 배치가 하나라서/pred[0] length==7 [rgb_coarse, disp_coarse, 
            #acc_coarse, rgb_fine, disp_fine, acc_fine, weights[:,-1]]
    #
    prediction_bunch = list(zip(*pred))#XXX 4x5->5x4
    #XXX synthesized_images[0][0].shape torch.Size([2048, 3])/ synthesized_images[6][0].shape torch.Size([2048])
    #XXX 앞자리는 rgb이런거 인덱싱 /뒷자리는 배치 인덱싱
    prediction_bunch = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in prediction_bunch
    ]#XXX 사실 chunk사이즈랑 ray사이즈랑 같아서 train떄는 지금 의미없음.
    
    if mode == "val":
        prediction_bunch = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(prediction_bunch, reshaping_list)
        ]
        return tuple(prediction_bunch)
    #7개짜리 튜플에 각각 rgb weights~~~
    return tuple(prediction_bunch)