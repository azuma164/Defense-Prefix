from datasets.caltech101 import Caltech101
from datasets.disentangling_ta import DisentanglingDataset
from datasets.dtd import DTD
from datasets.eurosat import EuroSAT
from datasets.fgvcaircraft import FGVCAircraft
from datasets.flowers102 import Flowers102
from datasets.food101 import Food101
from datasets.ImageNetV2 import ImageNetValDataset
from datasets.oxford_pets import OxfordIIITPet
from datasets.paint_ta import PAINTDataset
from datasets.rta100 import RTA100
from datasets.stanford_cars import StanfordCars
from datasets.sun397 import SUN397


def dataset(args, preprocess):
    # synthesized data
    if args.dataset == 'imagenet':
        data = ImageNetValDataset(location='datasets', transform=preprocess)
    elif args.dataset == 'caltech':
        data = Caltech101(root='datasets', transform=preprocess, download=True, make_typographic_dataset=True)
    elif args.dataset == 'pets':
        data = OxfordIIITPet(root='datasets', split='test', transform=preprocess, download=True, make_typographic_dataset=True)
    elif args.dataset == 'cars':
        data = StanfordCars(root='datasets', split='test', transform=preprocess, download=True, make_typographic_dataset=True)
    elif args.dataset == 'flowers':
        data = Flowers102(root='datasets', split='test', transform=preprocess, download=True, make_typographic_dataset=True)
    elif args.dataset == 'food':
        data = Food101(root='datasets', split='test', transform=preprocess, download=True)
    elif args.dataset == 'aircraft':
        data = FGVCAircraft(root='datasets', split='test', transform=preprocess, download=True)
    elif args.dataset == 'dtd':
        data = DTD(root='datasets', split='test', transform=preprocess, download=True)
    elif args.dataset == 'eurosat':
        data = EuroSAT(root='datasets', split='test', transform=preprocess, download=True)
    elif args.dataset == 'sun':
        data = SUN397(root='datasets', split='test', transform=preprocess, download=True)
    # real world data
    elif args.dataset == 'paint':
        data = PAINTDataset(root='datasets', transform=preprocess)
    elif args.dataset == 'disentangling':
        data = DisentanglingDataset(root='datasets', transform=preprocess)
    elif args.dataset == 'rta-100':
        data = RTA100(root='datasets', transform=preprocess)
    else:
        raise ValueError
    return data
