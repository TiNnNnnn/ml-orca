"""Device setup and completed-operation timings, independent of network/target."""
import torch


def training_device(name):
    device = torch.device(name)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('training device must be cpu or cuda')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable; refusing CPU fallback')
        torch.cuda.set_device(device.index if device.index is not None else 0)
        device = torch.device('cuda', torch.cuda.current_device())
        # Preserve full FP32 computation; GPU deployment does not authorize TF32/AMP.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return device


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def runtime_metadata(device):
    return {'device': str(device), 'torch': str(torch.__version__),
            'cuda': torch.version.cuda,
            'name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
            'precision': 'float32_no_amp_no_tf32'}
