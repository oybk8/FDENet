import torch
from torch import nn
import pywt
from typing import Sequence, Tuple, Union, List
from einops import rearrange
import torch.nn.functional as F
# 感受野枚举数组
dict_kd={5:(3,2),9:(5,2),13:(5,3),17:(5,4)}

def h_transform(x):
            shape = x.size()
            x = torch.nn.functional.pad(x, (0, shape[-1]))
            x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
            x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
            return x

def inv_h_transform(x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x
    
def v_transform(x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)
    
def inv_v_transform(x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

def _as_wavelet(wavelet):
    """Ensure the input argument to be a pywt wavelet compatible object.

    Args:
        wavelet (Wavelet or str): The input argument, which is either a
            pywt wavelet compatible object or a valid pywt wavelet name string.

    Returns:
        Wavelet: the input wavelet object or the pywt wavelet object described by the
            input str.
    """
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)
    else:
        return wavelet


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Torch implementation of numpy's outer for 1d vectors."""
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul * b_mul

def construct_2d_filt(lo, hi) -> torch.Tensor:
    """Construct two dimensional filters using outer products.

    Args:
        lo (torch.Tensor): Low-pass input filter.
        hi (torch.Tensor): High-pass input filter

    Returns:
        torch.Tensor: Stacked 2d filters of dimension
            [filt_no, 1, height, width].
            The four filters are ordered ll, lh, hl, hh.
    """
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    filt = torch.stack([ll, lh, hl, hh], 0)
    # filt = filt.unsqueeze(1)
    return filt


def get_filter_tensors(
        wavelet,
        flip: bool,
        device: Union[torch.device, str] = 'cpu',
        dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert input wavelet to filter tensors.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
        flip (bool): If true filters are flipped.
        device (torch.device) : PyTorch target device.
        dtype (torch.dtype): The data type sets the precision of the
               computation. Default: torch.float32.

    Returns:
        tuple: Tuple containing the four filter tensors
        dec_lo, dec_hi, rec_lo, rec_hi

    """
    wavelet = _as_wavelet(wavelet)

    def _create_tensor(filter: Sequence[float]) -> torch.Tensor:
        if flip:
            if isinstance(filter, torch.Tensor):
                return filter.flip(-1).unsqueeze(0).to(device)
            else:
                return torch.tensor(filter[::-1], device=device, dtype=dtype).unsqueeze(0)
        else:
            if isinstance(filter, torch.Tensor):
                return filter.unsqueeze(0).to(device)
            else:
                return torch.tensor(filter, device=device, dtype=dtype).unsqueeze(0)

    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo_tensor = _create_tensor(dec_lo)
    dec_hi_tensor = _create_tensor(dec_hi)
    rec_lo_tensor = _create_tensor(rec_lo)
    rec_hi_tensor = _create_tensor(rec_hi)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor


def _get_pad(data_len: int, filt_len: int) -> Tuple[int, int]:
    """Compute the required padding.

    Args:
        data_len (int): The length of the input vector.
        filt_len (int): The length of the used filter.

    Returns:
        tuple: The numbers to attach on the edges of the input.

    """
    # pad to ensure we see all filter positions and for pywt compatability.
    # convolution output length:
    # see https://arxiv.org/pdf/1603.07285.pdf section 2.3:
    # floor([data_len - filt_len]/2) + 1
    # should equal pywt output length
    # floor((data_len + filt_len - 1)/2)
    # => floor([data_len + total_pad - filt_len]/2) + 1
    #    = floor((data_len + filt_len - 1)/2)
    # (data_len + total_pad - filt_len) + 2 = data_len + filt_len - 1
    # total_pad = 2*filt_len - 3

    # we pad half of the total requried padding on each side.
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data_len % 2 != 0:
        padr += 1

    return padr, padl


def fwt_pad2(
        data: torch.Tensor, wavelet, mode: str = "replicate"
) -> torch.Tensor:
    """Pad data for the 2d FWT.

    Args:
        data (torch.Tensor): Input data with 4 dimensions.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        mode (str): The padding mode.
            Supported modes are "reflect", "zero", "constant" and "periodic".
            Defaults to reflect.

    Returns:
        The padded output tensor.

    """

    wavelet = _as_wavelet(wavelet)
    padb, padt = _get_pad(data.shape[-2], len(wavelet.dec_lo))
    padr, padl = _get_pad(data.shape[-1], len(wavelet.dec_lo))

    data_pad = F.pad(data, [padl, padr, padt, padb], mode=mode)
    return data_pad

class DWT(nn.Module):
    def __init__(self, dec_lo=None, dec_hi=None, wavelet='haar', level=1, mode="replicate"):
        super(DWT, self).__init__()
        self.wavelet = _as_wavelet(wavelet)
        self.dec_lo = dec_lo
        self.dec_hi = dec_hi
        if dec_lo is None:
            dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(
                wavelet, flip=True
            )
            self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
            self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)

        # # initial dec conv
        # self.conv = torch.nn.Conv2d(c1, c2 * 4, kernel_size=dec_filt.shape[-2:], groups=c1, stride=2)
        # self.conv.weight.data = dec_filt
        self.level = level
        self.mode = mode

    def forward(self, x):
        b, c, h, w = x.shape
        if self.level is None:
            self.level = pywt.dwtn_max_level([h, w], self.wavelet)
        wavelet_component: List[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = []

        l_component = x
        dwt_kernel = construct_2d_filt(lo=self.dec_lo, hi=self.dec_hi)
        dwt_kernel = dwt_kernel.repeat(c, 1, 1)
        dwt_kernel = dwt_kernel.unsqueeze(dim=1)
        for _ in range(self.level):
            l_component = fwt_pad2(l_component, self.wavelet, mode=self.mode)
            h_component = F.conv2d(l_component, dwt_kernel, stride=2, groups=c)
            res = rearrange(h_component, 'b (c f) h w -> b c f h w', f=4)
            l_component, lh_component, hl_component, hh_component = res.split(1, 2)
            wavelet_component.append((lh_component.squeeze(2), hl_component.squeeze(2), hh_component.squeeze(2)))
        wavelet_component.append(l_component.squeeze(2))
        return wavelet_component[::-1]


class IDWT(nn.Module):
    def __init__(self, rec_lo, rec_hi, wavelet='haar', level=1, mode="constant"):
        super(IDWT, self).__init__()
        self.rec_lo = rec_lo
        self.rec_hi = rec_hi
        self.wavelet = wavelet
        # self.convT = nn.ConvTranspose2d(c2 * 4, c1, kernel_size=weight.shape[-2:], groups=c1, stride=2)
        # self.convT.weight = torch.nn.Parameter(rec_filt)
        self.level = level
        self.mode = mode

    def forward(self, x, weight=None):
        l_component = x[0]
        _, c, _, _ = l_component.shape
        if weight is None:  # soft orthogonal
            idwt_kernel = construct_2d_filt(lo=self.rec_lo, hi=self.rec_hi)
            idwt_kernel = idwt_kernel.repeat(c, 1, 1)
            idwt_kernel = idwt_kernel.unsqueeze(dim=1)
        else:  # hard orthogonal
            idwt_kernel= torch.flip(weight, dims=[-1, -2])

        self.filt_len = idwt_kernel.shape[-1]
        for c_pos, component_lh_hl_hh in enumerate(x[1:]):
            l_component = torch.cat(
                # ll, lh, hl, hl, hh
                [l_component.unsqueeze(2), component_lh_hl_hh[0].unsqueeze(2),
                 component_lh_hl_hh[1].unsqueeze(2), component_lh_hl_hh[2].unsqueeze(2)], 2
            )
            # cat is not work for the strange transpose
            l_component = rearrange(l_component, 'b c f h w -> b (c f) h w')
            l_component = F.conv_transpose2d(l_component, idwt_kernel, stride=2, groups=c)

            # remove the padding
            padl = (2 * self.filt_len - 3) // 2
            padr = (2 * self.filt_len - 3) // 2
            padt = (2 * self.filt_len - 3) // 2
            padb = (2 * self.filt_len - 3) // 2
            if c_pos < len(x) - 2:
                pred_len = l_component.shape[-1] - (padl + padr)
                next_len = x[c_pos + 2][0].shape[-1]
                pred_len2 = l_component.shape[-2] - (padt + padb)
                next_len2 = x[c_pos + 2][0].shape[-2]
                if next_len != pred_len:
                    padr += 1
                    pred_len = l_component.shape[-1] - (padl + padr)
                    assert (
                            next_len == pred_len
                    ), "padding error, please open an issue on github "
                if next_len2 != pred_len2:
                    padb += 1
                    pred_len2 = l_component.shape[-2] - (padt + padb)
                    assert (
                            next_len2 == pred_len2
                    ), "padding error, please open an issue on github "
            if padt > 0:
                l_component = l_component[..., padt:, :]
            if padb > 0:
                l_component = l_component[..., :-padb, :]
            if padl > 0:
                l_component = l_component[..., padl:]
            if padr > 0:
                l_component = l_component[..., :-padr]
        return l_component

class  SConvx8(nn.Module):
    def __init__(self, dim, out_dim, cov_size=5, use_up=False):
        super(SConvx8, self).__init__() 
        self.deconv1 = nn.Conv2d(
            dim , out_dim//4, (1, cov_size), padding=(0, cov_size//2))
        self.deconv2 = nn.Conv2d(
            out_dim//4 , out_dim//4 , (cov_size, 1), padding=(cov_size//2, 0))
        self.deconv3 = nn.Conv2d(
            out_dim//4 , out_dim//4 , (cov_size, 1), padding=(cov_size//2, 0))
        self.deconv4 = nn.Conv2d(
            out_dim//4 , out_dim//4 , (1, cov_size), padding=(0, cov_size//2))
        if use_up:
            self.conv = nn.ConvTranspose2d(out_dim, out_dim, 3, stride=2, padding=1, output_padding=1)
        else :
            self.conv = nn.Conv2d(out_dim, out_dim, 1)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x1 = self.deconv1(x)
        x2 = self.deconv2(x1)
        x3 = inv_h_transform(self.deconv3(h_transform(x2)))
        x4 = inv_v_transform(self.deconv4(v_transform(x3)))
        y = torch.cat((x1,x2,x3,x4),dim=1)
        out = self.relu(self.bn(self.conv(y)))
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# 根据感受野，按照与k0和d0最小距离法，生成合适的k和d 
# k和d 满足条件：RF = k + (k-1)*(d-1) , k是>=3的奇数 d是>=2 k>=d k-d最小
def find_optimal_kd(receptive_field_size, k0=3, d0=2):
    """
    根据感受野大小，按照最小距离法生成合适的k和d
    
    参数:
        receptive_field_size: 目标感受野大小
        
    返回:
        (k, d): 满足条件的卷积核大小和膨胀率
    """
    min_diff = float('inf')
    best_k = None
    best_d = None
    # 遍历可能的d值（d >= 2）
    for d in range(2, (receptive_field_size - 1)//2+1):
        # 计算对应的k值，k必须是>=3的奇数，且k >= d
        # 根据公式 RF = k + (k-1)*(d-1) = k*d - (d-1)

        # 解方程得 k = (RF + d - 1) / d
        numerator = receptive_field_size + d - 1
        if numerator % d == 0:
            k_candidate = numerator // d
        else:
            continue
        
        # 确保k是奇数且>=3
        if k_candidate < 3 or k_candidate % 2 == 0:
            continue
        
        # 计算实际感受野大小
        actual_diff = (k_candidate-k0)*(k_candidate-k0) + (d-d0)*(d-d0)
            
        # 检查是否满足最小距离条件
        if actual_diff < min_diff and min_diff >100:
            min_diff = actual_diff
            best_k = k_candidate
            best_d = d
        elif actual_diff <= min_diff and k_candidate - d >= 0:
            min_diff = actual_diff
            best_k = k_candidate
            best_d = d
    
    return best_k, best_d

class FEM(nn.Module):
    def __init__(self, dim, out_dim, wavelet='haar', cov_size=5, cov_size2=9, use_up=True):
        super(FEM, self).__init__()
        self.dim = dim
        self.wavelet = _as_wavelet(wavelet)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(
            wavelet, flip=True
        )
        self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
        self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
        self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=True)
        self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=True)

        self.wavedec = DWT(self.dec_lo, self.dec_hi, wavelet=wavelet, level=1)
        self.waverec = IDWT(self.rec_lo, self.rec_hi, wavelet=wavelet, level=1)
        
        self.dsconv =  SConvx8(out_dim, out_dim, cov_size=cov_size2) 
        self.dsconv_aux = nn.Conv2d(out_dim , out_dim, 3, 1, 1)
        self.conv = nn.Conv2d(dim , out_dim, 1)
        self.cbr = BasicConv2d_relu(out_dim, out_dim, 1)  
        self.deconv1 = nn.Conv2d(
            out_dim, out_dim, (1, cov_size), padding=(0, cov_size//2))
        self.deconv2 = nn.Conv2d(
            out_dim, out_dim, (cov_size, 1), padding=(cov_size//2, 0))
        self.deconv3 = nn.Conv2d(
            out_dim, out_dim // 2, (cov_size, 1), padding=(cov_size//2, 0))
        self.deconv4 = nn.Conv2d(
            out_dim//2, out_dim // 2, (1, cov_size), padding=(0, cov_size//2))
        
        self.conv5 = nn.Conv2d(out_dim, out_dim, 5, padding=4, dilation=2, groups=out_dim//2)
        if use_up:
            self.conv4 = nn.ConvTranspose2d(out_dim , out_dim, 3, stride=2, padding=1, output_padding=1)
        else :
            self.conv4 = nn.Conv2d(out_dim , out_dim, 1)   
        self._init_weight()

    def forward(self, x):
        x = self.conv(x)
        ya, (yh, yv, yd) = self.wavedec(x)
        
        ya = self.dsconv(ya)+self.dsconv_aux(ya) # dim 
        yh = self.deconv1(yh) # dim 
        yv = self.deconv2(yv) # dim 
        yd1 = inv_h_transform(self.deconv3(h_transform(yd))) # dim //2
        yd2 = inv_v_transform(self.deconv4(v_transform(yd1))) # dim //2
        yd = torch.cat([yd1,yd2], dim=1)# dim 

        y = self.waverec([ya, (yh, yv, yd)], None) # dim
        z = self.conv4(x+self.cbr(y))
        out = self.conv5(z)
        return out
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()                

class BasicConv2d_relu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.bn(x))
        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
class AFM(nn.Module):
    def __init__(self, dim1, dim2, out_dim, reduction=4, use_up=False):
        super(AFM, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(dim1, out_dim, 1)
        if use_up:
            self.conv2 = nn.ConvTranspose2d(dim2 , out_dim, 3, stride=2, padding=1, output_padding=1)
        else :
            self.conv2 = nn.Conv2d(dim2 , out_dim, 1)
        d = out_dim//reduction
        self.fc1 = nn.Sequential(
            nn.Conv2d(out_dim*2, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, out_dim*2, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x_f = torch.cat([x1,x2],dim=1)
        B,N,_,_ = x_f.shape
        z = self.global_pool(x_f)
        attn = self.fc2(self.fc1(z))
        a_b = attn.reshape(B, 2, N // 2, -1)
        a_b = self.softmax(a_b)

        a,b =  a_b.chunk(2, dim=1)
        a = a.reshape(B,N // 2,1,1)
        b = b.reshape(B,N // 2,1,1)
        y = x1*a + x2*b
        return y

class SFCD(nn.Module):
    def __init__(self, filters=[64,128,256,512], a=7, b=4):
        super(SFCD,self).__init__()
        layers = len(filters)
        cov_sizes = [a + b*(layers - i - 1) for i in range(layers)]
        self.wsm1 =FEM(filters[0], filters[0]//2,cov_size=3,cov_size2=cov_sizes[0],use_up=True)
        self.wsm2 =FEM(filters[1], filters[1]//2,cov_size=3,cov_size2=cov_sizes[1],use_up=True)
        self.wsm3 =FEM(filters[2], filters[2]//2,cov_size=3,cov_size2=cov_sizes[2],use_up=True)
        self.wsm4 =FEM(filters[3], filters[3]//2,cov_size=3,cov_size2=cov_sizes[3],use_up=True)
        
        self.conv_e1 = AFM(filters[1]//2, filters[0], filters[0],1)
        self.conv_e2 = AFM(filters[2]//2, filters[1], filters[1],1)
        self.conv_e3 = AFM(filters[3]//2, filters[2], filters[2],1)

        
    def forward(self, fs):
        f1, f2, f3, f4 = fs
        d3= self.conv_e3(self.wsm4(f4),f3)
        d2= self.conv_e2(self.wsm3(d3),f2)
        d1= self.conv_e1(self.wsm2(d2),f1) # 64
        out= self.wsm1(d1)
        return out  
        
if __name__ == '__main__':
    filters =  [48, 96, 224, 448]
    input_tensor1 = torch.randn(2, filters[0], 256, 256)#输入 B C H W
    input_tensor2 = torch.randn(2, filters[1], 128, 128)#输入 B C H W
    input_tensor3 = torch.randn(2, filters[2], 64, 64)#输入 B C H W
    input_tensor4 = torch.randn(2, filters[3], 32, 32)#输入 B C H W 
    
    # 实例化 RCM 模块
    block = SFCD(filters)
    # 打印输入的形状
    print(input_tensor1.size())
    input = [input_tensor1, input_tensor2, input_tensor3, input_tensor4]
    # 将输入张量传递给 RCM 模块，并打印输出形状
    output_tensor = block(input)
    print(output_tensor.size())
    
