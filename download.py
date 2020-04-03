import gdown
import subprocess

urls = [
    'https://drive.google.com/open?id=1dKM_6sR7SCdDQXX8dIQwtpaW3qRiA7pM',
    'https://drive.google.com/open?id=1hXq-kxEsDA0_DzcOFS0yzIvlKxwcgoGZ',
    'https://drive.google.com/uc?id=1R69tifYSd3SRe7357ppljmU7l_WX7Wxt',
    'https://drive.google.com/uc?id=1BzxmQlDDiXOClxkF6I2rskK8-dl2nJdV'
    'https://drive.google.com/open?id=1tA6Vc30CdTq7QnKqxUzejY1w8gy-uFDm'
]

urls = [url.replace('open', 'uc') for url in urls]
for id, url in enumerate(urls[:-1]):
    output = 'training_data.00{}'.format(id + 1)
    print('Donwloading {} ...'.format(output))
    gdown.download(url, output, quiet=False)

output = 'Test'
print('Donwloading {} ...'.format(output))
gdown.download(urls[-1], output, quiet=False)

commands = ['mkdir pytorch/data/training_data/ShapeNetP2M/',
            'mv training_data.001 pytorch/data/training_data/ShapeNetP2M/',
            'mv training_data.002 pytorch/data/training_data/ShapeNetP2M/',
            'mv training_data.003 pytorch/data/training_data/ShapeNetP2M/',
            'mv training_data.004 pytorch/data/training_data/ShapeNetP2M/',
            'mv Test pytorch/data/training_data/']
for command in commands:
    subprocess.check_call(command.split())
