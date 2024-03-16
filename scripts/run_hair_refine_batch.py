import itertools
from subprocess import run
from torch.multiprocessing import Pool

A = ['chubby', 'slim']
B = ['Caucasian', 'African']
C = ['man', 'woman']
D = ['pageboy', 'bowl cut', 'chignon', 'finger waves', 'mohawk', 'pixie cut', 'full crown', 'short', 'undercut']
#['curly', 'long', 'short', 'sleek', 'pony-tail', 'wave', 'braid', 'bob cut', 'bowl cut', 'braid hair', 'caesar cut', 'chignon',
#      'cornrows', 'crown braid', 'crew cut', 'curtained', 'dido flip', 'dreadlocks', 'fad', 'fauxhawk', 'finger waves', 'french braid',
#      'frosted', 'full crown', 'harvard clip', 'hi-top fade', 'hime cut', 'jewfro', 'jheri curl', 'liberty spikes', 'marcel waves',
#      'mohawk', 'pageboy', 'perm', 'pixie cut', 'ponytail hair', 'psychobilly wedge', 'quiff', 'rachel', 'regular taper cut', 'ringlets',
#      'shingle bob', 'slicked back', 'spiky', 'surfer', 'taper cut', 'undercut']
E = ['Cuban collar blue shirt']

def wrapper(N):
    a, b, c, d, e = N
    geometry_text = f'a {a} {b} {c} with {d} hairstyle, front view, ID photo, headshot, portrait, a virtual eigen photo, high resolution, high quality, high details, uniform lighting, studio lighting, natural lighting'
    paint_text = f'a {a} {b} {c} wearing {e} with {d} hairstyle, {{}} view, a virtual eigen photo, high resolution, high quality, high details, uniform lighting, studio lighting, natural lighting'
    nerf_text = f'a {a} {b} {c} with {d} hairstyle, {{}} view, no watermark, a virtual eigen photo, high resolution, high quality, high details, uniform lighting, studio lighting, natural lighting'
    blip_text = f'a {a} {b} {c} with {d} hairstyle, {{}} view, no watermark, a virtual eigen photo, high resolution, high quality, high details, uniform lighting, studio lighting, natural lighting'
    exp_name = f'a {a} {b} {c} with {d} hairstyle refine'
    exp_root_path = f'a {a} {b} {c} with {d} hairstyle'
    ckpt = f'/home/work/hzhang/ckpt/a {a} {b} {c} with {d} hairstyle/checkpoints/step_003000.pth'
    run(['python', '-m', 'scripts.run_teca', '--config_path', 'configs/comparison/a_fat_European_woman_with_bowl_cut_hairstyle_refine.yaml', '--guide.geometry_text', geometry_text, '--guide.paint_text', paint_text, '--guide.nerf_text', nerf_text, '--guide.blip_text', blip_text, '--log.exp_name', exp_name, '--log.exp_root_path', exp_root_path, '--optim.ckpt', ckpt])

# Process two in parallel
with Pool(processes=2) as pool:
    list(pool.map(wrapper, itertools.product(A, B, C, D, E)))