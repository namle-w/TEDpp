import config, os
from utils import supervisor
import torch
from torchvision import datasets, transforms
from PIL import Image
from networks.models import Generator, NetC_MNIST
from utils.resnet import ResNet18, ResNet34

class BackdoorDefense():
    def __init__(self, args):
        self.dataset = args.dataset
        if args.dataset == 'gtsrb':
            self.img_size = 32
            self.num_classes = 43
            self.input_channel = 3
            self.shape = torch.Size([3, 32, 32])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1
        elif args.dataset == 'cifar10':      
            self.img_size = 32
            self.num_classes = 10
            self.input_channel = 3
            self.shape = torch.Size([3, 32, 32])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1
        elif args.dataset == 'mnist':
            self.img_size = 28
            self.num_classes = 10
            self.input_channel = 1
            self.shape = torch.Size([1, 28, 28])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1
        elif args.dataset == 'cifar100':
            print('<To Be Implemented> Dataset = %s' % args.dataset)
            exit(0)
        elif args.dataset == 'imagenette':   
            self.img_size = 224
            self.num_classes = 10
            self.input_channel = 3
            self.shape = torch.Size([3, 224, 224])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1
        elif args.dataset == 'imagenet100':
            self.img_size = 224
            self.num_classes = 100
            self.input_channel = 3
            self.shape = torch.Size([3, 224, 224])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.01
        elif args.dataset == 'imagenet200':
            self.img_size = 224
            self.num_classes = 200
            self.input_channel = 3
            self.shape = torch.Size([3, 224, 224])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.01
        elif args.dataset == 'tinyimagenet200':
            self.img_size = 64
            self.num_classes = 200
            self.input_channel = 3
            self.shape = torch.Size([3, 64, 64])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.01
        elif args.dataset == 'imagenet':
            self.img_size = 224
            self.num_classes = 1000
            self.input_channel = 3
            self.shape = torch.Size([3, 224, 224])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1
        else:
            print('<Undefined> Dataset = %s' % args.dataset)
            exit(0)
        
        self.data_transform_aug, self.data_transform, self.trigger_transform, self.normalizer, self.denormalizer = supervisor.get_transforms(args)
        
        self.poison_type = args.poison_type
        self.poison_rate = args.poison_rate
        self.cover_rate = args.cover_rate
        self.alpha = args.alpha
        self.trigger = args.trigger
        self.target_class = config.target_class[args.dataset]
        self.device='cuda'

        if args.poison_type == 'SSDT':
            self.poison_transform = type('DummyTransform', (), {
                'transform': lambda self, x, y: (x, y)
            })()
        else:
            self.poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                                target_class=config.target_class[args.dataset], trigger_transform=self.trigger_transform,
                                                                is_normalized_input=(not args.no_normalize),
                                                                alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                                trigger_name=args.trigger, args=args)
        
        if args.poison_type == 'TaCT' or args.poison_type == 'SleeperAgent':
            self.source_classes = [config.source_class]
        else:
            self.source_classes = None

        if args.poison_type != 'SSDT':
            trigger_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            trigger_path = os.path.join(config.triggers_dir, args.trigger)
            print('trigger_path:', trigger_path)
            self.trigger_mark = Image.open(trigger_path).convert("RGB")
            self.trigger_mark = trigger_transform(self.trigger_mark).cuda()

            trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % args.trigger)
            if os.path.exists(trigger_mask_path): # if there explicitly exists a trigger mask (with the same name)
                print('trigger_mask_path:', trigger_mask_path)
                self.trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                self.trigger_mask = transforms.ToTensor()(self.trigger_mask)[0].cuda() # only use 1 channel
            else: # by default, all black pixels are masked with 0's (not used)
                print('No trigger mask found! By default masking all black pixels...')
                self.trigger_mask = torch.logical_or(torch.logical_or(self.trigger_mark[0] > 0, self.trigger_mark[1] > 0), self.trigger_mark[2] > 0).cuda()


        self.poison_set_dir = supervisor.get_poison_set_dir(args)

        if self.poison_type == 'SSDT':
            print(f"Loading SSDT model for dataset '{self.dataset}' targeting '{self.target_class}'...")
            self.model = self.load_model()
            state_dict = self.load_model_state()
            self.model = self.load_state(self.model, state_dict["netC"])
            print("SSDT model loaded successfully:")
            print(self.model)

            print("Args content:", args.__dict__)
            self.netG = Generator(args)
            self.netG = self.load_state(self.netG, state_dict["netG"])

            self.netM = Generator(args, out_channels=1)
            self.netM = self.load_state(self.netM, state_dict["netM"])
        else:
            if self.poison_type == 'SSDT':
                model_path = f'checkpoints/{args.dataset}/SSDT/target_{config.target_class[args.dataset]}/SSDT_{args.dataset}_ckpt.pth.tar'
            else:
                model_path = supervisor.get_model_dir(args)

            arch = supervisor.get_arch(args)
            self.model = arch(num_classes=self.num_classes)

            if os.path.exists(model_path):
                print(f"Loading model '{model_path}'...")
                checkpoint = torch.load(model_path, map_location=self.device)
                if args.poison_type == 'SSDT':
                    if 'netC' in checkpoint:
                        self.model.load_state_dict(checkpoint['netC'], strict=False)
                        print("SSDT model loaded successfully using 'netC' weights.")
                    else:
                        print("Warning: SSDT checkpoint does not contain 'netC'. Attempting full load.")
                        self.model.load_state_dict(checkpoint, strict=False)
                else:
                    self.model.load_state_dict(checkpoint)
                print("Model loaded.")
            else:
                print(f"Model '{model_path}' not found.")

        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()
        self.model.eval()

    def load_model(self):
        # if self.dataset == "cifar10":
        #     return ResNet18().to(self.device)
        # elif self.dataset == "gtsrb":
        #     return ResNet18(num_classes=43).to(self.device)
        # else:
        #     raise ValueError(f"Unknown dataset: {self.dataset}")

        return config.arch[self.dataset](num_classes=self.num_classes).to(self.device)

    def load_model_state(self):
        base_path = './checkpoints/'
        model_path = f"{base_path}{self.dataset}/SSDT/target_{self.target_class}/SSDT_{self.dataset}_ckpt.pth.tar"
        return torch.load(model_path, map_location=self.device, weights_only=False)

    def load_state(self, model, state_dict):
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        model.requires_grad_(False)
        return model

        



