import torch
d = torch.load('cgr_diag_logs/cgr_diag_seed0.pt', weights_only=False)
print({k: (v.shape, v.dtype) if torch.is_tensor(v) else v for k, v in d.items()})
print('cgr_conf range:', d['cgr_confidence_by_sample'].min().item(),
      d['cgr_confidence_by_sample'].max().item())
print('margin range:', d['diag_margin'].min().item(),
      d['diag_margin'].max().item())
print('epoch-0 acc:', d['diag_correct'][0].float().mean().item())
print('last-epoch acc:', d['diag_correct'][-1].float().mean().item())
