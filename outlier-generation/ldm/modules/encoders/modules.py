import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import kornia

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)
import numpy as np
from copy import deepcopy
class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="/home1/gaoheng/.cache/huggingface/transformers", device="cuda", max_length=77):  # /home1/gaoheng/.cache/huggingface/transformers  openai/clip-vit-large-patch14
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.token_embedding = self.transformer.text_model.embeddings.token_embedding.weight

        self.new_dis = torch.distributions.MultivariateNormal(torch.zeros(768).cuda(), torch.eye(768).cuda())
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False


    def forward(self, text, class_index):
        if text[0] != '':
            if self.id_data != 'in100':
                fine_labels = [   # cifar100
                    'apples',  # id 0
                    'fish',
                    'baby',
                    'bear',
                    'beaver',
                    'bed',
                    'bee',
                    'beetle',
                    'bicycle',
                    'bottles',
                    'bowls',
                    'boy',
                    'bridge',
                    'bus',
                    'butterfly',
                    'camel',
                    'cans',
                    'castle',
                    'caterpillar',
                    'cattle',
                    'chair',
                    'chimp',
                    'clock',
                    'cloud',
                    'cockroach',
                    'couch',
                    'crab',
                    'crocodile',
                    'cups',
                    'dinosaur',
                    'dolphin',
                    'elephant',
                    'flatfish',
                    'forest',
                    'fox',
                    'girl',
                    'hamster',
                    'house',
                    'kangaroo',
                    'keyboard',
                    'lamp',
                    'mower',
                    'leopard',
                    'lion',
                    'lizard',
                    'lobster',
                    'man',
                    'maple',
                    'motorcycle',
                    'mountain',
                    'mouse',
                    'mushrooms',
                    'oak',
                    'oranges',
                    'orchids',
                    'otter',
                    'palm',
                    'pears',
                    'truck',
                    'pine',
                    'plain',
                    'plates',
                    'poppies',
                    'porcupine',
                    'possum',
                    'rabbit',
                    'raccoon',
                    'ray',
                    'road',
                    'rocket',
                    'roses',
                    'sea',
                    'seal',
                    'shark',
                    'shrew',
                    'skunk',
                    'skyscraper',
                    'snail',
                    'snake',
                    'spider',
                    'squirrel',
                    'streetcar',
                    'sunflowers',
                    'peppers',
                    'table',
                    'tank',
                    'telephone',
                    'television',
                    'tiger',
                    'tractor',
                    'train',
                    'trout',
                    'tulips',
                    'turtle',
                    'wardrobe',
                    'whale',
                    'willow',
                    'wolf',
                    'woman',
                    'worm',
                ]
            elif self.id_data=='cifar10':
                fine_labels = ['airplane',
                               'automobile',
                               'bird',
                               'cat',
                               'deer',
                               'dog',
                               'frog',
                               'horse',
                               'ship',
                               'truck']
            elif self.id_data=='in1k':
                fine_labels = ['tench', 'goldfish', 'great white shark', 'tiger shark',
                            'hammerhead shark', 'electric ray', 'stingray', 'rooster', 'hen',
                            'ostrich', 'brambling', 'goldfinch', 'house finch', 'junco',
                            'indigo bunting', 'American robin', 'bulbul', 'jay', 'magpie', 'chickadee',
                            'American dipper', 'kite (bird of prey)', 'bald eagle', 'vulture',
                            'great grey owl', 'fire salamander', 'smooth newt', 'newt',
                            'spotted salamander', 'axolotl', 'American bullfrog', 'tree frog',
                            'tailed frog', 'loggerhead sea turtle', 'leatherback sea turtle',
                            'mud turtle', 'terrapin', 'box turtle', 'banded gecko', 'green iguana',
                            'Carolina anole', 'desert grassland whiptail lizard', 'agama',
                            'frilled-necked lizard', 'alligator lizard', 'Gila monster',
                            'European green lizard', 'chameleon', 'Komodo dragon', 'Nile crocodile',
                            'American alligator', 'triceratops', 'worm snake', 'ring-necked snake',
                            'eastern hog-nosed snake', 'smooth green snake', 'kingsnake',
                            'garter snake', 'water snake', 'vine snake', 'night snake',
                            'boa constrictor', 'African rock python', 'Indian cobra', 'green mamba',
                            'sea snake', 'Saharan horned viper', 'eastern diamondback rattlesnake',
                            'sidewinder rattlesnake', 'trilobite', 'harvestman', 'scorpion',
                            'yellow garden spider', 'barn spider', 'European garden spider',
                            'southern black widow', 'tarantula', 'wolf spider', 'tick', 'centipede',
                            'black grouse', 'ptarmigan', 'ruffed grouse', 'prairie grouse', 'peafowl',
                            'quail', 'partridge', 'african grey parrot', 'macaw',
                            'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater', 'hornbill',
                            'hummingbird', 'jacamar', 'toucan', 'duck', 'red-breasted merganser',
                            'goose', 'black swan', 'tusker', 'echidna', 'platypus', 'wallaby', 'koala',
                            'wombat', 'jellyfish', 'sea anemone', 'brain coral', 'flatworm',
                            'nematode', 'conch', 'snail', 'slug', 'sea slug', 'chiton',
                            'chambered nautilus', 'Dungeness crab', 'rock crab', 'fiddler crab',
                            'red king crab', 'American lobster', 'spiny lobster', 'crayfish',
                            'hermit crab', 'isopod', 'white stork', 'black stork', 'spoonbill',
                            'flamingo', 'little blue heron', 'great egret', 'bittern bird',
                            'crane bird', 'limpkin', 'common gallinule', 'American coot', 'bustard',
                            'ruddy turnstone', 'dunlin', 'common redshank', 'dowitcher',
                            'oystercatcher', 'pelican', 'king penguin', 'albatross', 'grey whale',
                            'killer whale', 'dugong', 'sea lion', 'Chihuahua', 'Japanese Chin',
                            'Maltese', 'Pekingese', 'Shih Tzu', 'King Charles Spaniel', 'Papillon',
                            'toy terrier', 'Rhodesian Ridgeback', 'Afghan Hound', 'Basset Hound',
                            'Beagle', 'Bloodhound', 'Bluetick Coonhound', 'Black and Tan Coonhound',
                            'Treeing Walker Coonhound', 'English foxhound', 'Redbone Coonhound',
                            'borzoi', 'Irish Wolfhound', 'Italian Greyhound', 'Whippet',
                            'Ibizan Hound', 'Norwegian Elkhound', 'Otterhound', 'Saluki',
                            'Scottish Deerhound', 'Weimaraner', 'Staffordshire Bull Terrier',
                            'American Staffordshire Terrier', 'Bedlington Terrier', 'Border Terrier',
                            'Kerry Blue Terrier', 'Irish Terrier', 'Norfolk Terrier',
                            'Norwich Terrier', 'Yorkshire Terrier', 'Wire Fox Terrier',
                            'Lakeland Terrier', 'Sealyham Terrier', 'Airedale Terrier',
                            'Cairn Terrier', 'Australian Terrier', 'Dandie Dinmont Terrier',
                            'Boston Terrier', 'Miniature Schnauzer', 'Giant Schnauzer',
                            'Standard Schnauzer', 'Scottish Terrier', 'Tibetan Terrier',
                            'Australian Silky Terrier', 'Soft-coated Wheaten Terrier',
                            'West Highland White Terrier', 'Lhasa Apso', 'Flat-Coated Retriever',
                            'Curly-coated Retriever', 'Golden Retriever', 'Labrador Retriever',
                            'Chesapeake Bay Retriever', 'German Shorthaired Pointer', 'Vizsla',
                            'English Setter', 'Irish Setter', 'Gordon Setter', 'Brittany dog',
                            'Clumber Spaniel', 'English Springer Spaniel', 'Welsh Springer Spaniel',
                            'Cocker Spaniel', 'Sussex Spaniel', 'Irish Water Spaniel', 'Kuvasz',
                            'Schipperke', 'Groenendael dog', 'Malinois', 'Briard', 'Australian Kelpie',
                            'Komondor', 'Old English Sheepdog', 'Shetland Sheepdog', 'collie',
                            'Border Collie', 'Bouvier des Flandres dog', 'Rottweiler',
                            'German Shepherd Dog', 'Dobermann', 'Miniature Pinscher',
                            'Greater Swiss Mountain Dog', 'Bernese Mountain Dog',
                            'Appenzeller Sennenhund', 'Entlebucher Sennenhund', 'Boxer', 'Bullmastiff',
                            'Tibetan Mastiff', 'French Bulldog', 'Great Dane', 'St. Bernard', 'husky',
                            'Alaskan Malamute', 'Siberian Husky', 'Dalmatian', 'Affenpinscher',
                            'Basenji', 'pug', 'Leonberger', 'Newfoundland dog', 'Great Pyrenees dog',
                            'Samoyed', 'Pomeranian', 'Chow Chow', 'Keeshond', 'brussels griffon',
                            'Pembroke Welsh Corgi', 'Cardigan Welsh Corgi', 'Toy Poodle',
                            'Miniature Poodle', 'Standard Poodle',
                            'Mexican hairless dog (xoloitzcuintli)', 'grey wolf',
                            'Alaskan tundra wolf', 'red wolf or maned wolf', 'coyote', 'dingo',
                            'dhole', 'African wild dog', 'hyena', 'red fox', 'kit fox', 'Arctic fox',
                            'grey fox', 'tabby cat', 'tiger cat', 'Persian cat', 'Siamese cat',
                            'Egyptian Mau', 'cougar', 'lynx', 'leopard', 'snow leopard', 'jaguar',
                            'lion', 'tiger', 'cheetah', 'brown bear', 'American black bear',
                            'polar bear', 'sloth bear', 'mongoose', 'meerkat', 'tiger beetle',
                            'ladybug', 'ground beetle', 'longhorn beetle', 'leaf beetle',
                            'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant',
                            'grasshopper', 'cricket insect', 'stick insect', 'cockroach',
                            'praying mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly',
                            'damselfly', 'red admiral butterfly', 'ringlet butterfly',
                            'monarch butterfly', 'small white butterfly', 'sulphur butterfly',
                            'gossamer-winged butterfly', 'starfish', 'sea urchin', 'sea cucumber',
                            'cottontail rabbit', 'hare', 'Angora rabbit', 'hamster', 'porcupine',
                            'fox squirrel', 'marmot', 'beaver', 'guinea pig', 'common sorrel horse',
                            'zebra', 'pig', 'wild boar', 'warthog', 'hippopotamus', 'ox',
                            'water buffalo', 'bison', 'ram (adult male sheep)', 'bighorn sheep',
                            'Alpine ibex', 'hartebeest', 'impala (antelope)', 'gazelle',
                            'arabian camel', 'llama', 'weasel', 'mink', 'European polecat',
                            'black-footed ferret', 'otter', 'skunk', 'badger', 'armadillo',
                            'three-toed sloth', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon',
                            'siamang', 'guenon', 'patas monkey', 'baboon', 'macaque', 'langur',
                            'black-and-white colobus', 'proboscis monkey', 'marmoset',
                            'white-headed capuchin', 'howler monkey', 'titi monkey',
                            "Geoffroy's spider monkey", 'common squirrel monkey', 'ring-tailed lemur',
                            'indri', 'Asian elephant', 'African bush elephant', 'red panda',
                            'giant panda', 'snoek fish', 'eel', 'silver salmon', 'rock beauty fish',
                            'clownfish', 'sturgeon', 'gar fish', 'lionfish', 'pufferfish', 'abacus',
                            'abaya', 'academic gown', 'accordion', 'acoustic guitar',
                            'aircraft carrier', 'airliner', 'airship', 'altar', 'ambulance',
                            'amphibious vehicle', 'analog clock', 'apiary', 'apron', 'trash can',
                            'assault rifle', 'backpack', 'bakery', 'balance beam', 'balloon',
                            'ballpoint pen', 'Band-Aid', 'banjo', 'baluster / handrail', 'barbell',
                            'barber chair', 'barbershop', 'barn', 'barometer', 'barrel', 'wheelbarrow',
                            'baseball', 'basketball', 'bassinet', 'bassoon', 'swimming cap',
                            'bath towel', 'bathtub', 'station wagon', 'lighthouse', 'beaker',
                            'military hat (bearskin or shako)', 'beer bottle', 'beer glass',
                            'bell tower', 'baby bib', 'tandem bicycle', 'bikini', 'ring binder',
                            'binoculars', 'birdhouse', 'boathouse', 'bobsleigh', 'bolo tie',
                            'poke bonnet', 'bookcase', 'bookstore', 'bottle cap', 'hunting bow',
                            'bow tie', 'brass memorial plaque', 'bra', 'breakwater', 'breastplate',
                            'broom', 'bucket', 'buckle', 'bulletproof vest', 'high-speed train',
                            'butcher shop', 'taxicab', 'cauldron', 'candle', 'cannon', 'canoe',
                            'can opener', 'cardigan', 'car mirror', 'carousel', 'tool kit',
                            'cardboard box / carton', 'car wheel', 'automated teller machine',
                            'cassette', 'cassette player', 'castle', 'catamaran', 'CD player', 'cello',
                            'mobile phone', 'chain', 'chain-link fence', 'chain mail', 'chainsaw',
                            'storage chest', 'chiffonier', 'bell or wind chime', 'china cabinet',
                            'Christmas stocking', 'church', 'movie theater', 'cleaver',
                            'cliff dwelling', 'cloak', 'clogs', 'cocktail shaker', 'coffee mug',
                            'coffeemaker', 'spiral or coil', 'combination lock', 'computer keyboard',
                            'candy store', 'container ship', 'convertible', 'corkscrew', 'cornet',
                            'cowboy boot', 'cowboy hat', 'cradle', 'construction crane',
                            'crash helmet', 'crate', 'infant bed', 'Crock Pot', 'croquet ball',
                            'crutch', 'cuirass', 'dam', 'desk', 'desktop computer',
                            'rotary dial telephone', 'diaper', 'digital clock', 'digital watch',
                            'dining table', 'dishcloth', 'dishwasher', 'disc brake', 'dock',
                            'dog sled', 'dome', 'doormat', 'drilling rig', 'drum', 'drumstick',
                            'dumbbell', 'Dutch oven', 'electric fan', 'electric guitar',
                            'electric locomotive', 'entertainment center', 'envelope',
                            'espresso machine', 'face powder', 'feather boa', 'filing cabinet',
                            'fireboat', 'fire truck', 'fire screen', 'flagpole', 'flute',
                            'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen',
                            'four-poster bed', 'freight car', 'French horn', 'frying pan', 'fur coat',
                            'garbage truck', 'gas mask or respirator', 'gas pump', 'goblet', 'go-kart',
                            'golf ball', 'golf cart', 'gondola', 'gong', 'gown', 'grand piano',
                            'greenhouse', 'radiator grille', 'grocery store', 'guillotine',
                            'hair clip', 'hair spray', 'half-track', 'hammer', 'hamper', 'hair dryer',
                            'hand-held computer', 'handkerchief', 'hard disk drive', 'harmonica',
                            'harp', 'combine harvester', 'hatchet', 'holster', 'home theater',
                            'honeycomb', 'hook', 'hoop skirt', 'gymnastic horizontal bar',
                            'horse-drawn vehicle', 'hourglass', 'iPod', 'clothes iron',
                            'carved pumpkin', 'jeans', 'jeep', 'T-shirt', 'jigsaw puzzle', 'rickshaw',
                            'joystick', 'kimono', 'knee pad', 'knot', 'lab coat', 'ladle', 'lampshade',
                            'laptop computer', 'lawn mower', 'lens cap', 'letter opener', 'library',
                            'lifeboat', 'lighter', 'limousine', 'ocean liner', 'lipstick',
                            'slip-on shoe', 'lotion', 'music speaker', 'loupe magnifying glass',
                            'sawmill', 'magnetic compass', 'messenger bag', 'mailbox', 'tights',
                            'one-piece bathing suit', 'manhole cover', 'maraca', 'marimba', 'mask',
                            'matchstick', 'maypole', 'maze', 'measuring cup', 'medicine cabinet',
                            'megalith', 'microphone', 'microwave oven', 'military uniform', 'milk can',
                            'minibus', 'miniskirt', 'minivan', 'missile', 'mitten', 'mixing bowl',
                            'mobile home', 'ford model t', 'modem', 'monastery', 'monitor', 'moped',
                            'mortar and pestle', 'graduation cap', 'mosque', 'mosquito net', 'vespa',
                            'mountain bike', 'tent', 'computer mouse', 'mousetrap', 'moving van',
                            'muzzle', 'metal nail', 'neck brace', 'necklace', 'baby pacifier',
                            'notebook computer', 'obelisk', 'oboe', 'ocarina', 'odometer',
                            'oil filter', 'pipe organ', 'oscilloscope', 'overskirt', 'bullock cart',
                            'oxygen mask', 'product packet / packaging', 'paddle', 'paddle wheel',
                            'padlock', 'paintbrush', 'pajamas', 'palace', 'pan flute', 'paper towel',
                            'parachute', 'parallel bars', 'park bench', 'parking meter',
                            'railroad car', 'patio', 'payphone', 'pedestal', 'pencil case',
                            'pencil sharpener', 'perfume', 'Petri dish', 'photocopier', 'plectrum',
                            'Pickelhaube', 'picket fence', 'pickup truck', 'pier', 'piggy bank',
                            'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate ship',
                            'drink pitcher', 'block plane', 'planetarium', 'plastic bag', 'plate rack',
                            'farm plow', 'plunger', 'Polaroid camera', 'pole', 'police van', 'poncho',
                            'pool table', 'soda bottle', 'plant pot', "potter's wheel", 'power drill',
                            'prayer rug', 'printer', 'prison', 'missile', 'projector', 'hockey puck',
                            'punching bag', 'purse', 'quill', 'quilt', 'race car', 'racket',
                            'radiator', 'radio', 'radio telescope', 'rain barrel',
                            'recreational vehicle', 'fishing casting reel', 'reflex camera',
                            'refrigerator', 'remote control', 'restaurant', 'revolver', 'rifle',
                            'rocking chair', 'rotisserie', 'eraser', 'rugby ball',
                            'ruler measuring stick', 'sneaker', 'safe', 'safety pin', 'salt shaker',
                            'sandal', 'sarong', 'saxophone', 'scabbard', 'weighing scale',
                            'school bus', 'schooner', 'scoreboard', 'CRT monitor', 'screw',
                            'screwdriver', 'seat belt', 'sewing machine', 'shield', 'shoe store',
                            'shoji screen / room divider', 'shopping basket', 'shopping cart',
                            'shovel', 'shower cap', 'shower curtain', 'ski', 'balaclava ski mask',
                            'sleeping bag', 'slide rule', 'sliding door', 'slot machine', 'snorkel',
                            'snowmobile', 'snowplow', 'soap dispenser', 'soccer ball', 'sock',
                            'solar thermal collector', 'sombrero', 'soup bowl', 'keyboard space bar',
                            'space heater', 'space shuttle', 'spatula', 'motorboat', 'spider web',
                            'spindle', 'sports car', 'spotlight', 'stage', 'steam locomotive',
                            'through arch bridge', 'steel drum', 'stethoscope', 'scarf', 'stone wall',
                            'stopwatch', 'stove', 'strainer', 'tram', 'stretcher', 'couch', 'stupa',
                            'submarine', 'suit', 'sundial', 'sunglasses', 'sunglasses', 'sunscreen',
                            'suspension bridge', 'mop', 'sweatshirt', 'swim trunks / shorts', 'swing',
                            'electrical switch', 'syringe', 'table lamp', 'tank', 'tape player',
                            'teapot', 'teddy bear', 'television', 'tennis ball', 'thatched roof',
                            'front curtain', 'thimble', 'threshing machine', 'throne', 'tile roof',
                            'toaster', 'tobacco shop', 'toilet seat', 'torch', 'totem pole',
                            'tow truck', 'toy store', 'tractor', 'semi-trailer truck', 'tray',
                            'trench coat', 'tricycle', 'trimaran', 'tripod', 'triumphal arch',
                            'trolleybus', 'trombone', 'hot tub', 'turnstile', 'typewriter keyboard',
                            'umbrella', 'unicycle', 'upright piano', 'vacuum cleaner', 'vase',
                            'vaulted or arched ceiling', 'velvet fabric', 'vending machine',
                            'vestment', 'viaduct', 'violin', 'volleyball', 'waffle iron', 'wall clock',
                            'wallet', 'wardrobe', 'military aircraft', 'sink', 'washing machine',
                            'water bottle', 'water jug', 'water tower', 'whiskey jug', 'whistle',
                            'hair wig', 'window screen', 'window shade', 'Windsor tie', 'wine bottle',
                            'airplane wing', 'wok', 'wooden spoon', 'wool', 'split-rail fence',
                            'shipwreck', 'sailboat', 'yurt', 'website', 'comic book', 'crossword',
                            'traffic or street sign', 'traffic light', 'dust jacket', 'menu', 'plate',
                            'guacamole', 'consomme', 'hot pot', 'trifle', 'ice cream', 'popsicle',
                            'baguette', 'bagel', 'pretzel', 'cheeseburger', 'hot dog',
                            'mashed potatoes', 'cabbage', 'broccoli', 'cauliflower', 'zucchini',
                            'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber',
                            'artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith apple',
                            'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit',
                            'cherimoya (custard apple)', 'pomegranate', 'hay', 'carbonara',
                            'chocolate syrup', 'dough', 'meatloaf', 'pizza', 'pot pie', 'burrito',
                            'red wine', 'espresso', 'tea cup', 'eggnog', 'mountain', 'bubble', 'cliff',
                            'coral reef', 'geyser', 'lakeshore', 'promontory', 'sandbar', 'beach',
                            'valley', 'volcano', 'baseball player', 'bridegroom', 'scuba diver',
                            'rapeseed', 'daisy', "yellow lady's slipper", 'corn', 'acorn', 'rose hip',
                            'horse chestnut seed', 'coral fungus', 'agaric', 'gyromitra',
                            'stinkhorn mushroom', 'earth star fungus', 'hen of the woods mushroom',
                            'bolete', 'corn cob', 'toilet paper']
            else:
                fine_labels = ['stingray', 'hen', 'magpie', 'kite', 'vulture',
                   'agama',   'tick', 'quail', 'hummingbird', 'koala',
                   'jellyfish', 'snail', 'crawfish', 'flamingo', 'orca',
                   'chihuahua', 'coyote', 'tabby', 'leopard', 'lion',
                   'tiger','ladybug', 'fly' , 'ant', 'grasshopper',
                   'monarch', 'starfish', 'hare', 'hamster', 'beaver',
                   'zebra', 'pig', 'ox', 'impala',  'mink',
                   'otter', 'gorilla', 'panda', 'sturgeon', 'accordion',
                   'carrier', 'ambulance', 'apron', 'backpack', 'balloon',
                   'banjo','barn','baseball', 'basketball', 'beacon',
                   'binder', 'broom', 'candle', 'castle', 'chain',
                   'chest', 'church', 'cinema', 'cradle', 'dam',
                   'desk', 'dome', 'drum','envelope', 'forklift',
                   'fountain', 'gown', 'hammer','jean', 'jeep',
                   'knot', 'laptop', 'mower', 'library','lipstick',
                   'mask', 'maze', 'microphone','microwave','missile',
                    'nail', 'perfume','pillow','printer','purse',
                   'rifle', 'sandal', 'screw','stage','stove',
                   'swing','television','tractor','tripod','umbrella',
                    'violin','whistle','wreck', 'broccoli', 'strawberry'
                   ]
            
            fine_labels = ['tench', 'goldfish', 'great white shark', 'tiger shark',
                            'hammerhead shark', 'electric ray', 'stingray', 'rooster', 'hen',
                            'ostrich', 'brambling', 'goldfinch', 'house finch', 'junco',
                            'indigo bunting', 'American robin', 'bulbul', 'jay', 'magpie', 'chickadee',
                            'American dipper', 'kite (bird of prey)', 'bald eagle', 'vulture',
                            'great grey owl', 'fire salamander', 'smooth newt', 'newt',
                            'spotted salamander', 'axolotl', 'American bullfrog', 'tree frog',
                            'tailed frog', 'loggerhead sea turtle', 'leatherback sea turtle',
                            'mud turtle', 'terrapin', 'box turtle', 'banded gecko', 'green iguana',
                            'Carolina anole', 'desert grassland whiptail lizard', 'agama',
                            'frilled-necked lizard', 'alligator lizard', 'Gila monster',
                            'European green lizard', 'chameleon', 'Komodo dragon', 'Nile crocodile',
                            'American alligator', 'triceratops', 'worm snake', 'ring-necked snake',
                            'eastern hog-nosed snake', 'smooth green snake', 'kingsnake',
                            'garter snake', 'water snake', 'vine snake', 'night snake',
                            'boa constrictor', 'African rock python', 'Indian cobra', 'green mamba',
                            'sea snake', 'Saharan horned viper', 'eastern diamondback rattlesnake',
                            'sidewinder rattlesnake', 'trilobite', 'harvestman', 'scorpion',
                            'yellow garden spider', 'barn spider', 'European garden spider',
                            'southern black widow', 'tarantula', 'wolf spider', 'tick', 'centipede',
                            'black grouse', 'ptarmigan', 'ruffed grouse', 'prairie grouse', 'peafowl',
                            'quail', 'partridge', 'african grey parrot', 'macaw',
                            'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater', 'hornbill',
                            'hummingbird', 'jacamar', 'toucan', 'duck', 'red-breasted merganser',
                            'goose', 'black swan', 'tusker', 'echidna', 'platypus', 'wallaby', 'koala',
                            'wombat', 'jellyfish', 'sea anemone', 'brain coral', 'flatworm',
                            'nematode', 'conch', 'snail', 'slug', 'sea slug', 'chiton',
                            'chambered nautilus', 'Dungeness crab', 'rock crab', 'fiddler crab',
                            'red king crab', 'American lobster', 'spiny lobster', 'crayfish',
                            'hermit crab', 'isopod', 'white stork', 'black stork', 'spoonbill',
                            'flamingo', 'little blue heron', 'great egret', 'bittern bird',
                            'crane bird', 'limpkin', 'common gallinule', 'American coot', 'bustard',
                            'ruddy turnstone', 'dunlin', 'common redshank', 'dowitcher',
                            'oystercatcher', 'pelican', 'king penguin', 'albatross', 'grey whale',
                            'killer whale', 'dugong', 'sea lion', 'Chihuahua', 'Japanese Chin',
                            'Maltese', 'Pekingese', 'Shih Tzu', 'King Charles Spaniel', 'Papillon',
                            'toy terrier', 'Rhodesian Ridgeback', 'Afghan Hound', 'Basset Hound',
                            'Beagle', 'Bloodhound', 'Bluetick Coonhound', 'Black and Tan Coonhound',
                            'Treeing Walker Coonhound', 'English foxhound', 'Redbone Coonhound',
                            'borzoi', 'Irish Wolfhound', 'Italian Greyhound', 'Whippet',
                            'Ibizan Hound', 'Norwegian Elkhound', 'Otterhound', 'Saluki',
                            'Scottish Deerhound', 'Weimaraner', 'Staffordshire Bull Terrier',
                            'American Staffordshire Terrier', 'Bedlington Terrier', 'Border Terrier',
                            'Kerry Blue Terrier', 'Irish Terrier', 'Norfolk Terrier',
                            'Norwich Terrier', 'Yorkshire Terrier', 'Wire Fox Terrier',
                            'Lakeland Terrier', 'Sealyham Terrier', 'Airedale Terrier',
                            'Cairn Terrier', 'Australian Terrier', 'Dandie Dinmont Terrier',
                            'Boston Terrier', 'Miniature Schnauzer', 'Giant Schnauzer',
                            'Standard Schnauzer', 'Scottish Terrier', 'Tibetan Terrier',
                            'Australian Silky Terrier', 'Soft-coated Wheaten Terrier',
                            'West Highland White Terrier', 'Lhasa Apso', 'Flat-Coated Retriever',
                            'Curly-coated Retriever', 'Golden Retriever', 'Labrador Retriever',
                            'Chesapeake Bay Retriever', 'German Shorthaired Pointer', 'Vizsla',
                            'English Setter', 'Irish Setter', 'Gordon Setter', 'Brittany dog',
                            'Clumber Spaniel', 'English Springer Spaniel', 'Welsh Springer Spaniel',
                            'Cocker Spaniel', 'Sussex Spaniel', 'Irish Water Spaniel', 'Kuvasz',
                            'Schipperke', 'Groenendael dog', 'Malinois', 'Briard', 'Australian Kelpie',
                            'Komondor', 'Old English Sheepdog', 'Shetland Sheepdog', 'collie',
                            'Border Collie', 'Bouvier des Flandres dog', 'Rottweiler',
                            'German Shepherd Dog', 'Dobermann', 'Miniature Pinscher',
                            'Greater Swiss Mountain Dog', 'Bernese Mountain Dog',
                            'Appenzeller Sennenhund', 'Entlebucher Sennenhund', 'Boxer', 'Bullmastiff',
                            'Tibetan Mastiff', 'French Bulldog', 'Great Dane', 'St. Bernard', 'husky',
                            'Alaskan Malamute', 'Siberian Husky', 'Dalmatian', 'Affenpinscher',
                            'Basenji', 'pug', 'Leonberger', 'Newfoundland dog', 'Great Pyrenees dog',
                            'Samoyed', 'Pomeranian', 'Chow Chow', 'Keeshond', 'brussels griffon',
                            'Pembroke Welsh Corgi', 'Cardigan Welsh Corgi', 'Toy Poodle',
                            'Miniature Poodle', 'Standard Poodle',
                            'Mexican hairless dog (xoloitzcuintli)', 'grey wolf',
                            'Alaskan tundra wolf', 'red wolf or maned wolf', 'coyote', 'dingo',
                            'dhole', 'African wild dog', 'hyena', 'red fox', 'kit fox', 'Arctic fox',
                            'grey fox', 'tabby cat', 'tiger cat', 'Persian cat', 'Siamese cat',
                            'Egyptian Mau', 'cougar', 'lynx', 'leopard', 'snow leopard', 'jaguar',
                            'lion', 'tiger', 'cheetah', 'brown bear', 'American black bear',
                            'polar bear', 'sloth bear', 'mongoose', 'meerkat', 'tiger beetle',
                            'ladybug', 'ground beetle', 'longhorn beetle', 'leaf beetle',
                            'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant',
                            'grasshopper', 'cricket insect', 'stick insect', 'cockroach',
                            'praying mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly',
                            'damselfly', 'red admiral butterfly', 'ringlet butterfly',
                            'monarch butterfly', 'small white butterfly', 'sulphur butterfly',
                            'gossamer-winged butterfly', 'starfish', 'sea urchin', 'sea cucumber',
                            'cottontail rabbit', 'hare', 'Angora rabbit', 'hamster', 'porcupine',
                            'fox squirrel', 'marmot', 'beaver', 'guinea pig', 'common sorrel horse',
                            'zebra', 'pig', 'wild boar', 'warthog', 'hippopotamus', 'ox',
                            'water buffalo', 'bison', 'ram (adult male sheep)', 'bighorn sheep',
                            'Alpine ibex', 'hartebeest', 'impala (antelope)', 'gazelle',
                            'arabian camel', 'llama', 'weasel', 'mink', 'European polecat',
                            'black-footed ferret', 'otter', 'skunk', 'badger', 'armadillo',
                            'three-toed sloth', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon',
                            'siamang', 'guenon', 'patas monkey', 'baboon', 'macaque', 'langur',
                            'black-and-white colobus', 'proboscis monkey', 'marmoset',
                            'white-headed capuchin', 'howler monkey', 'titi monkey',
                            "Geoffroy's spider monkey", 'common squirrel monkey', 'ring-tailed lemur',
                            'indri', 'Asian elephant', 'African bush elephant', 'red panda',
                            'giant panda', 'snoek fish', 'eel', 'silver salmon', 'rock beauty fish',
                            'clownfish', 'sturgeon', 'gar fish', 'lionfish', 'pufferfish', 'abacus',
                            'abaya', 'academic gown', 'accordion', 'acoustic guitar',
                            'aircraft carrier', 'airliner', 'airship', 'altar', 'ambulance',
                            'amphibious vehicle', 'analog clock', 'apiary', 'apron', 'trash can',
                            'assault rifle', 'backpack', 'bakery', 'balance beam', 'balloon',
                            'ballpoint pen', 'Band-Aid', 'banjo', 'baluster / handrail', 'barbell',
                            'barber chair', 'barbershop', 'barn', 'barometer', 'barrel', 'wheelbarrow',
                            'baseball', 'basketball', 'bassinet', 'bassoon', 'swimming cap',
                            'bath towel', 'bathtub', 'station wagon', 'lighthouse', 'beaker',
                            'military hat (bearskin or shako)', 'beer bottle', 'beer glass',
                            'bell tower', 'baby bib', 'tandem bicycle', 'bikini', 'ring binder',
                            'binoculars', 'birdhouse', 'boathouse', 'bobsleigh', 'bolo tie',
                            'poke bonnet', 'bookcase', 'bookstore', 'bottle cap', 'hunting bow',
                            'bow tie', 'brass memorial plaque', 'bra', 'breakwater', 'breastplate',
                            'broom', 'bucket', 'buckle', 'bulletproof vest', 'high-speed train',
                            'butcher shop', 'taxicab', 'cauldron', 'candle', 'cannon', 'canoe',
                            'can opener', 'cardigan', 'car mirror', 'carousel', 'tool kit',
                            'cardboard box / carton', 'car wheel', 'automated teller machine',
                            'cassette', 'cassette player', 'castle', 'catamaran', 'CD player', 'cello',
                            'mobile phone', 'chain', 'chain-link fence', 'chain mail', 'chainsaw',
                            'storage chest', 'chiffonier', 'bell or wind chime', 'china cabinet',
                            'Christmas stocking', 'church', 'movie theater', 'cleaver',
                            'cliff dwelling', 'cloak', 'clogs', 'cocktail shaker', 'coffee mug',
                            'coffeemaker', 'spiral or coil', 'combination lock', 'computer keyboard',
                            'candy store', 'container ship', 'convertible', 'corkscrew', 'cornet',
                            'cowboy boot', 'cowboy hat', 'cradle', 'construction crane',
                            'crash helmet', 'crate', 'infant bed', 'Crock Pot', 'croquet ball',
                            'crutch', 'cuirass', 'dam', 'desk', 'desktop computer',
                            'rotary dial telephone', 'diaper', 'digital clock', 'digital watch',
                            'dining table', 'dishcloth', 'dishwasher', 'disc brake', 'dock',
                            'dog sled', 'dome', 'doormat', 'drilling rig', 'drum', 'drumstick',
                            'dumbbell', 'Dutch oven', 'electric fan', 'electric guitar',
                            'electric locomotive', 'entertainment center', 'envelope',
                            'espresso machine', 'face powder', 'feather boa', 'filing cabinet',
                            'fireboat', 'fire truck', 'fire screen', 'flagpole', 'flute',
                            'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen',
                            'four-poster bed', 'freight car', 'French horn', 'frying pan', 'fur coat',
                            'garbage truck', 'gas mask or respirator', 'gas pump', 'goblet', 'go-kart',
                            'golf ball', 'golf cart', 'gondola', 'gong', 'gown', 'grand piano',
                            'greenhouse', 'radiator grille', 'grocery store', 'guillotine',
                            'hair clip', 'hair spray', 'half-track', 'hammer', 'hamper', 'hair dryer',
                            'hand-held computer', 'handkerchief', 'hard disk drive', 'harmonica',
                            'harp', 'combine harvester', 'hatchet', 'holster', 'home theater',
                            'honeycomb', 'hook', 'hoop skirt', 'gymnastic horizontal bar',
                            'horse-drawn vehicle', 'hourglass', 'iPod', 'clothes iron',
                            'carved pumpkin', 'jeans', 'jeep', 'T-shirt', 'jigsaw puzzle', 'rickshaw',
                            'joystick', 'kimono', 'knee pad', 'knot', 'lab coat', 'ladle', 'lampshade',
                            'laptop computer', 'lawn mower', 'lens cap', 'letter opener', 'library',
                            'lifeboat', 'lighter', 'limousine', 'ocean liner', 'lipstick',
                            'slip-on shoe', 'lotion', 'music speaker', 'loupe magnifying glass',
                            'sawmill', 'magnetic compass', 'messenger bag', 'mailbox', 'tights',
                            'one-piece bathing suit', 'manhole cover', 'maraca', 'marimba', 'mask',
                            'matchstick', 'maypole', 'maze', 'measuring cup', 'medicine cabinet',
                            'megalith', 'microphone', 'microwave oven', 'military uniform', 'milk can',
                            'minibus', 'miniskirt', 'minivan', 'missile', 'mitten', 'mixing bowl',
                            'mobile home', 'ford model t', 'modem', 'monastery', 'monitor', 'moped',
                            'mortar and pestle', 'graduation cap', 'mosque', 'mosquito net', 'vespa',
                            'mountain bike', 'tent', 'computer mouse', 'mousetrap', 'moving van',
                            'muzzle', 'metal nail', 'neck brace', 'necklace', 'baby pacifier',
                            'notebook computer', 'obelisk', 'oboe', 'ocarina', 'odometer',
                            'oil filter', 'pipe organ', 'oscilloscope', 'overskirt', 'bullock cart',
                            'oxygen mask', 'product packet / packaging', 'paddle', 'paddle wheel',
                            'padlock', 'paintbrush', 'pajamas', 'palace', 'pan flute', 'paper towel',
                            'parachute', 'parallel bars', 'park bench', 'parking meter',
                            'railroad car', 'patio', 'payphone', 'pedestal', 'pencil case',
                            'pencil sharpener', 'perfume', 'Petri dish', 'photocopier', 'plectrum',
                            'Pickelhaube', 'picket fence', 'pickup truck', 'pier', 'piggy bank',
                            'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate ship',
                            'drink pitcher', 'block plane', 'planetarium', 'plastic bag', 'plate rack',
                            'farm plow', 'plunger', 'Polaroid camera', 'pole', 'police van', 'poncho',
                            'pool table', 'soda bottle', 'plant pot', "potter's wheel", 'power drill',
                            'prayer rug', 'printer', 'prison', 'missile', 'projector', 'hockey puck',
                            'punching bag', 'purse', 'quill', 'quilt', 'race car', 'racket',
                            'radiator', 'radio', 'radio telescope', 'rain barrel',
                            'recreational vehicle', 'fishing casting reel', 'reflex camera',
                            'refrigerator', 'remote control', 'restaurant', 'revolver', 'rifle',
                            'rocking chair', 'rotisserie', 'eraser', 'rugby ball',
                            'ruler measuring stick', 'sneaker', 'safe', 'safety pin', 'salt shaker',
                            'sandal', 'sarong', 'saxophone', 'scabbard', 'weighing scale',
                            'school bus', 'schooner', 'scoreboard', 'CRT monitor', 'screw',
                            'screwdriver', 'seat belt', 'sewing machine', 'shield', 'shoe store',
                            'shoji screen / room divider', 'shopping basket', 'shopping cart',
                            'shovel', 'shower cap', 'shower curtain', 'ski', 'balaclava ski mask',
                            'sleeping bag', 'slide rule', 'sliding door', 'slot machine', 'snorkel',
                            'snowmobile', 'snowplow', 'soap dispenser', 'soccer ball', 'sock',
                            'solar thermal collector', 'sombrero', 'soup bowl', 'keyboard space bar',
                            'space heater', 'space shuttle', 'spatula', 'motorboat', 'spider web',
                            'spindle', 'sports car', 'spotlight', 'stage', 'steam locomotive',
                            'through arch bridge', 'steel drum', 'stethoscope', 'scarf', 'stone wall',
                            'stopwatch', 'stove', 'strainer', 'tram', 'stretcher', 'couch', 'stupa',
                            'submarine', 'suit', 'sundial', 'sunglasses', 'sunglasses', 'sunscreen',
                            'suspension bridge', 'mop', 'sweatshirt', 'swim trunks / shorts', 'swing',
                            'electrical switch', 'syringe', 'table lamp', 'tank', 'tape player',
                            'teapot', 'teddy bear', 'television', 'tennis ball', 'thatched roof',
                            'front curtain', 'thimble', 'threshing machine', 'throne', 'tile roof',
                            'toaster', 'tobacco shop', 'toilet seat', 'torch', 'totem pole',
                            'tow truck', 'toy store', 'tractor', 'semi-trailer truck', 'tray',
                            'trench coat', 'tricycle', 'trimaran', 'tripod', 'triumphal arch',
                            'trolleybus', 'trombone', 'hot tub', 'turnstile', 'typewriter keyboard',
                            'umbrella', 'unicycle', 'upright piano', 'vacuum cleaner', 'vase',
                            'vaulted or arched ceiling', 'velvet fabric', 'vending machine',
                            'vestment', 'viaduct', 'violin', 'volleyball', 'waffle iron', 'wall clock',
                            'wallet', 'wardrobe', 'military aircraft', 'sink', 'washing machine',
                            'water bottle', 'water jug', 'water tower', 'whiskey jug', 'whistle',
                            'hair wig', 'window screen', 'window shade', 'Windsor tie', 'wine bottle',
                            'airplane wing', 'wok', 'wooden spoon', 'wool', 'split-rail fence',
                            'shipwreck', 'sailboat', 'yurt', 'website', 'comic book', 'crossword',
                            'traffic or street sign', 'traffic light', 'dust jacket', 'menu', 'plate',
                            'guacamole', 'consomme', 'hot pot', 'trifle', 'ice cream', 'popsicle',
                            'baguette', 'bagel', 'pretzel', 'cheeseburger', 'hot dog',
                            'mashed potatoes', 'cabbage', 'broccoli', 'cauliflower', 'zucchini',
                            'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber',
                            'artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith apple',
                            'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit',
                            'cherimoya (custard apple)', 'pomegranate', 'hay', 'carbonara',
                            'chocolate syrup', 'dough', 'meatloaf', 'pizza', 'pot pie', 'burrito',
                            'red wine', 'espresso', 'tea cup', 'eggnog', 'mountain', 'bubble', 'cliff',
                            'coral reef', 'geyser', 'lakeshore', 'promontory', 'sandbar', 'beach',
                            'valley', 'volcano', 'baseball player', 'bridegroom', 'scuba diver',
                            'rapeseed', 'daisy', "yellow lady's slipper", 'corn', 'acorn', 'rose hip',
                            'horse chestnut seed', 'coral fungus', 'agaric', 'gyromitra',
                            'stinkhorn mushroom', 'earth star fungus', 'hen of the woods mushroom',
                            'bolete', 'corn cob', 'toilet paper']
            tmp_token = self.tokenizer([fine_labels[class_index]], truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens = tmp_token["input_ids"].to(self.device)
            original_embed = deepcopy(self.transformer.text_model.embeddings.token_embedding.weight[tokens[0][1]])
            original_id = tokens[0][1]
            print(self.transformer.text_model.embeddings.token_embedding.weight[original_id][:10])
            if False:
                noise = 0.03 * self.new_dis.rsample(
                    (1,)).squeeze()
                self.transformer.text_model.embeddings.token_embedding.weight[original_id] = noise + original_embed
            else:
                outlier = self.outlier_embedding[class_index][np.random.choice(1000, 1)[0]]  # choice(10000, 1)
                self.transformer.text_model.embeddings.token_embedding.weight[original_id] = outlier.cuda()
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        # return to the intial embeddings for sampling next time.
        if text[0] != '':
            self.transformer.text_model.embeddings.token_embedding.weight[original_id] = original_embed
        return z

    def encode(self, text, class_index, opt):
        self.id_data = opt.id_data
        self.outlier_embedding = torch.from_numpy(
                np.load(opt.loaded_embedding))
        return self(text, class_index)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)
    # version="/home1/gaoheng/.cache/huggingface/transformers"
    # # 初始化 CLIPTokenizer 和 CLIPTextModel
    # tokenizer = CLIPTokenizer.from_pretrained(version)
    # text_model = CLIPTextModel.from_pretrained(version)

    # # 定义要编码的单词列表
    # fine_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # # 初始化保存编码结果的数组
    # encoded_texts = np.zeros((len(fine_labels), 768))

    # # 对每个单词进行编码并保存结果
    # for idx, word in enumerate(fine_labels):
    #     # 使用 tokenizer 对单词进行编码
    #     tokenized_input = tokenizer(word, return_tensors="pt")

    #     # 使用 CLIPTextModel 获取文本编码
    #     with torch.no_grad():
    #         text_encoding = text_model(**tokenized_input).last_hidden_state

    #     # 将文本编码保存到数组中
    #     encoded_texts[idx] = text_encoding.squeeze().numpy()

    # # 打印保存的数组
    # print(encoded_texts)
    
    
    