import torch
from torch import nn, einsum
from einops import rearrange, reduce, repeat
import math
import torch.nn.functional as F
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
from random import random
from torch.optim.lr_scheduler import _LRScheduler
from DiffusionFreeGuidence.Unet1D_temp import UNet1D
import numpy as np

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def identity(t, *args, **kwargs):
    return t

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class MLP(nn.Module):
    def __init__(self,input_dim=16,output_dim=1,layers_num=4,hidden_dim=50,droupout=0.2):
        super(MLP, self).__init__()

        assert layers_num >= 2, "layers must be greater than 2"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim

        self.layers = []
        for i in range(layers_num):
            if i == 0:
                self.layers.append(nn.Linear(input_dim,hidden_dim))
                self.layers.append(Sin())
            elif i == layers_num-1:
                self.layers.append(nn.Linear(hidden_dim,output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim,hidden_dim))
                self.layers.append(Sin())
                self.layers.append(nn.Dropout(p=droupout))
        self.net = nn.Sequential(*self.layers)
        self._init()

    def _init(self):
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self,x):
        x = self.net(x)
        return x


class Predictor(nn.Module):
    def __init__(self,input_dim=40):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim,32),
            Sin(),
            nn.Linear(32,1)
        )
        self.input_dim = input_dim
    def forward(self,x):
        return self.net(x)

class Solution_u(nn.Module):
    def __init__(self, args):
        super(Solution_u, self).__init__()
        self.encoder = MLP(input_dim=args.input_size,output_dim=32,layers_num=3,hidden_dim=60,droupout=0.2)
        self.predictor = Predictor(input_dim=32)
        self._init_()

    def get_embedding(self,x):
        return self.encoder(x)

    def forward(self,x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x

    def _init_(self):
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)
            elif isinstance(layer,nn.Conv1d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
        
class GaussianDiffusion1D_cls_free(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        channels,
        args,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l2',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.
    ):
        super().__init__()
        # assert not (type(self) == GaussianDiffusion1D_cls_free and model.channels != model.out_dim)

        self.model = model
        self.channels = channels
        self.now_epoch = args.now_epoch
        self.epoch = args.epoch
        self.warmup = args.warmup
        self.dataset = args.dataset
        self.phylimit = args.phylimit
        self.argk = args.argk
        self.solution_u = Solution_u(args=args).to(args.device)

        self.seq_length = seq_length

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def correlation(self, x):
        selected_data = x[:, [4, 5, 6, 8, 10], :] 
        mask = selected_data[:, :3, :] >= 0.15  
        selected_data[:, :3, :] = mask.float()  
        mask_2 = selected_data[:, 4:5, :] <= 0.8  
        selected_data[:, 4:5, :] = mask_2.float()  

        diff_square_1 = (selected_data[:, :3, :] - selected_data[:, 3:4, :]) ** 2  
        diff_square_2 = (selected_data[:, 4:5, :] - selected_data[:, 3:4, :]) ** 2  
        diff_square_1 = diff_square_1 * 0.5
        diff_square_2 = diff_square_2 * 0.5
        loss = diff_square_1.mean() + diff_square_2.mean()

        return loss
    
    def onlyzero_one(self, x):
        row_nine = x[:, 8, :]
        processed_row_nine = torch.sin(2 * math.pi * row_nine - (math.pi / 2)) + 1
        processed_row_nine = processed_row_nine * 0.5
        loss = processed_row_nine.mean()
        
        return loss
    
    def discrete(self, x):
        if self.dataset == 'FD001':
            discrete_num = 12
        elif self.dataset == 'FD003':
            discrete_num = 11
        x = unnormalize_to_zero_to_one(x)
        x_12th_col = x[:, 11, :] * discrete_num 
        x_12th_col = x_12th_col % 1 
        pai = math.pi
        pai_half = pai *0.5
        penalty = torch.sin(x_12th_col*2*pai - pai_half) + 1
        penalty = penalty * 0.5
        penalty_loss = penalty.mean()
        return penalty_loss 
    
    def increase(self, x):
        _,_,length = x.shape
        if self.dataset == 'FD001':
            num1 = 6
            num2 = 10
        elif self.dataset == 'FD003':
            num1 = 6
            num2 = 10
        column_6 = x[:, num1 - 1,:] 
        column_6_mean = column_6[:, 0] + column_6[:, 1] + column_6[:, 2] + column_6[:, 3] + column_6[:, 4] - column_6[:, length-1] - column_6[:, length-2] - column_6[:, length-3] - column_6[:, length-4] - column_6[:, length-5]
        column_6_mean = column_6_mean.unsqueeze(1)
        column_6_mean = column_6_mean / 5
        column_10 = x[:, num2 - 1,:] 
        column_10_mean = column_10[:, 0] + column_10[:, 1] + column_10[:, 2] + column_10[:, 3] + column_10[:, 4] - column_10[:, length-1] - column_10[:, length-2] - column_10[:, length-3] - column_10[:, length-4] - column_10[:, length-5]
        column_10_mean = column_10_mean.unsqueeze(1)
        column_10_mean = column_10_mean / 5
        x_10_means = column_10_mean
        relu_differences_10 = F.relu(x_10_means - 8/length)*0.1
        x_6_means = column_6_mean
        relu_differences_6 = F.relu(x_6_means - 8/length)*0.1
        up_loss = relu_differences_10.mean() + relu_differences_6.mean()
        
        return up_loss
    
    def decrease(self, x):

        _,_,length = x.shape
        column_6 = x[:, 14 - 1,:]  
        column_6_mean = column_6[:, 0] + column_6[:, 1] + column_6[:, 2] + column_6[:, 3] + column_6[:, 4] - column_6[:, length-1] - column_6[:, length-2] - column_6[:, length-3] - column_6[:, length-4] - column_6[:, length-5]
        column_6_mean = column_6_mean.unsqueeze(1)
        column_6_mean = (-column_6_mean) / 5
        relu_differences_6 = F.relu(column_6_mean - (8/length)) * 0.002 * length
        up_loss = relu_differences_6.mean()
        return up_loss

    def loss01(self, x):
        x = unnormalize_to_zero_to_one(x)
        data_1 = x - 1
        loss_1 = F.relu(data_1) * 0.1
        data_0 = -x
        loss_0 = F.relu(data_0) * 0.1
        return loss_1.mean() + loss_0.mean()

    def model_predictions(self, x, t, classes, cond_scale = 3., clip_x_start = False):
        if hasattr(self.model, 'forward_with_cond_scale'):
            model_output = self.model.forward_with_cond_scale(x, t, classes, cond_scale=cond_scale)
        else:
            model_output = self.model(x, t, classes)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        x0 = x
        if self.objective == 'pred_noise':
            pred_noise = model_output
            if self.warmup == 'sample':
                x = self.predict_start_from_noise(x, t, model_output)
                x = unnormalize_to_zero_to_one(x)
                if self.dataset == 'FD001' or self.dataset == 'FD003':
                    gradient = torch.autograd.functional.jacobian(self.discrete, x) * 7 + torch.autograd.functional.jacobian(self.increase, x) * 8 + torch.autograd.functional.jacobian(self.decrease, x) * 8 + torch.autograd.functional.jacobian(self.loss01, x)
                else:
                    gradient = torch.autograd.functional.jacobian(self.onlyzero_one, x) * 10 + torch.autograd.functional.jacobian(self.correlation, x) * 20 + torch.autograd.functional.jacobian(self.loss01, x)
                sqrt_one_minus_alpha_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
                gradient_norm = sqrt_one_minus_alpha_t * gradient
                pred_noise = pred_noise - gradient_norm
            x_start = self.predict_start_from_noise(x0, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, classes, cond_scale, clip_denoised = True):
        preds = self.model_predictions(x, t, classes, cond_scale)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, classes, cond_scale = 3., clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, classes = classes, cond_scale = cond_scale, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_scale = 3.):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, x_start = self.p_sample(img, t, classes, cond_scale)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, classes, shape, cond_scale = 3., clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, cond_scale = cond_scale, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, classes, cond_scale = 3.):
        batch_size, seq_length, channels = classes.shape[0], self.seq_length, self.channels
        sample_fn = self.p_sample_loop #if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(classes, (batch_size, channels, seq_length), cond_scale)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def increase_train(self, data, number):
        _,_,length = data.shape
        column_6 = data[:, number - 1,:] 
        column_6_mean = column_6[:, 0] + column_6[:, 1] + column_6[:, 2] + column_6[:, 3] + column_6[:, 4] - column_6[:, length-1] - column_6[:, length-2] - column_6[:, length-3] - column_6[:, length-4] - column_6[:, length-5]
        column_6_mean = column_6_mean.unsqueeze(1)
        column_6_mean = column_6_mean / 5
        relu_differences_6 = F.relu(column_6_mean - (8/length)) * 0.0001 * length
        up_loss = relu_differences_6.mean()
        return up_loss
    
    def decrease_train(self, data, number):
        _,_,length = data.shape
        column_6 = data[:, number - 1,:] 
        column_6_mean = column_6[:, 0] + column_6[:, 1] + column_6[:, 2] + column_6[:, 3] + column_6[:, 4] - column_6[:, length-1] - column_6[:, length-2] - column_6[:, length-3] - column_6[:, length-4] - column_6[:, length-5]
        column_6_mean = column_6_mean.unsqueeze(1)
        column_6_mean = (-column_6_mean) / 5
        relu_differences_6 = F.relu(column_6_mean - (8/length)) * 0.0001 * length
        up_loss = relu_differences_6.mean()
        return up_loss

    def minmax(self, data):
        data_1 = data - 1
        loss_1 = F.relu(data_1) * 0.01
        data_0 = -data
        loss_0 = F.relu(data_0) * 0.01
        return loss_1.mean() + loss_0.mean()
    
    def discrete_train(self, x):
        if self.dataset == 'FD001':
            discrete_num = 12
        elif self.dataset == 'FD003':
            discrete_num = 11    
        x_12th_col = x[:, 11, :] * discrete_num  
        x_12th_col = x_12th_col % 1 
        pai = math.pi
        pai_half = pai *0.5
        penalty = torch.sin(x_12th_col*2*pai - pai_half) + 1
        penalty = penalty * 0.01
        penalty_loss = penalty.mean()
        return penalty_loss
    
    def exp_k(self, x):
        return 1 / (1 + np.exp(-self.argk*np.log(x / (1 - x))))
    
    def correlation_loss(self, data):
        _,_,length = data.shape
        selected_data = data[:, [4, 5, 6, 8, 10], :]  
        mask = selected_data[:, :3, :] >= 0.15  
        selected_data[:, :3, :] = mask.float() 
        mask_2 = selected_data[:, 4:5, :] <= 0.8  
        selected_data[:, 4:5, :] = mask_2.float() 

        diff_square_1 = (selected_data[:, :3, :] - selected_data[:, 3:4, :]) ** 2  
        diff_square_2 = (selected_data[:, 4:5, :] - selected_data[:, 3:4, :]) ** 2  
        diff_square_1 = diff_square_1  *0.05
        diff_square_2 = diff_square_2  *0.05
        loss = diff_square_1.mean() + diff_square_2.mean()

        return loss
    
    def onlyzero_one_loss(self, data):
        row_nine = data[:, 8, :]
        processed_row_nine = torch.sin(2 * math.pi * row_nine - (math.pi / 2)) + 1
        processed_row_nine = processed_row_nine 
        loss = processed_row_nine.mean() * 0.1
        
        return loss
    
    def p_losses(self, x_start, t, *, classes, noise = None):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        model_out = self.model(x, t, classes)
        

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        pred_x0 = self.predict_start_from_noise(x, t, model_out)
        pred_x0 = unnormalize_to_zero_to_one(pred_x0)
        if self.dataset == 'FD001':
            num = 5
            num2 = 9
        elif self.dataset == 'FD003':
            num = 5
            num2 = 9

        if self.warmup == 'true':
            if self.dataset == 'FD001' or self.dataset == 'FD003':
                total_loss = (self.minmax(pred_x0) + self.discrete_train(pred_x0) + self.increase_train(pred_x0, num) + self.increase_train(pred_x0, num2) + self.decrease_train(pred_x0, 14)) * self.exp_k(self.now_epoch/self.epoch) + loss.mean()
            elif self.dataset in ['2c', '3c', 'r25', 'r3', 'rw', 'sate', 'bat', 'battery']:
                u_input = pred_x0[:, :, -1] 
                u = self.solution_u(u_input)
                u = u.squeeze(-1)  
                y = classes.squeeze(-1)           
                sorted_y, indices = torch.sort(y)
                sorted_u = u[indices]
                u1 = sorted_u[:-1]
                u2 = sorted_u[1:]
                y1 = sorted_y[:-1]
                y2 = sorted_y[1:]
                loss3 = F.relu(torch.mul(u2 - u1, y1 - y2)).sum()
                total_loss = (loss3/20000) * self.exp_k(self.now_epoch/self.epoch) + loss.mean()
            else:
                total_loss = (self.minmax(pred_x0) + self.onlyzero_one_loss(pred_x0) + self.correlation_loss(pred_x0)) * self.exp_k(self.now_epoch/self.epoch) + loss.mean()
        elif self.warmup == 'none' or self.warmup == 'sample':
            return loss.mean()
        else:
            if self.dataset == 'FD001' or self.dataset == 'FD003':
                total_loss = (self.minmax(pred_x0) + self.discrete_train(pred_x0) + self.increase_train(pred_x0, num) + self.increase_train(pred_x0, num2) + self.decrease_train(pred_x0, 14)) + loss.mean()
            else:
                total_loss = (self.minmax(pred_x0) + self.onlyzero_one_loss(pred_x0) + self.correlation_loss(pred_x0)) + loss.mean()
        return total_loss
        #return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert n == seq_length , f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)

# example

if __name__ == '__main__':
    num_classes = 125

    model = UNet1D(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        # num_classes = num_classes,
        cond_drop_prob = 0.5
    )

    diffusion = GaussianDiffusion1D_cls_free(
        model,
        seq_length = 128,
        timesteps = 1000,
        sampling_timesteps=50
    ).cuda()

    training_signals = torch.randn(8, 3, 128).cuda() # images are normalized from 0 to 1
    signal_classes = torch.randint(0, num_classes, (8,1)).cuda()    # say 10 classes

    loss = diffusion(training_signals, classes = signal_classes)
    loss.backward()

    # do above for many steps

    sampled_signals = diffusion.sample(
        classes = signal_classes,
        cond_scale = 3.                # condition scaling, anything greater than 1 strengthens the classifier free guidance. reportedly 3-8 is good empirically
    )

    print(sampled_signals.shape) # (8, 3, 128)