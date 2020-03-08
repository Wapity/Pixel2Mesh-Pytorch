# Put in pytorch/data/training_data/ShapeNetP2M/
import gdown
import subprocess

urls = [
    'https://drive.google.com/uc?id=1hyTn97fxTuUi49bk1mKzKesgBkzEaScK',
    'https://drive.google.com/uc?id=1Tc9s7gbfyouMGVoHatHtwr-QSCDe5bau'
]

urls = [url.replace('open', 'uc') for url in urls]
for id, url in enumerate(urls):
    output = 'training_data.00{}'.format(id + 1)
    print('Donwloading {} ...'.format(output))
    gdown.download(url, output, quiet=False)


commands = ['mkdir pytorch/data/training_data/ShapeNetP2M/',
            'mv training_data.001 pytorch/data/training_data/ShapeNetP2M/',
            'mv training_data.002 pytorch/data/training_data/ShapeNetP2M/',
            'unzip training_data.001', 'unzip training_data.002']
for command in commands:
    subprocess.check_call(command.split())
