import gdown
urls = [
    'https://drive.google.com/uc?id=1kjLMLd_55Ro7wDrKeFCL3mT0daQ7Aepf',
    'https://drive.google.com/uc?id=12XYrdIqZp4_jg2AJNZtCQvMC35KrnN4F',
    'https://drive.google.com/uc?id=10N5DUf8eacNVWK7txZt3ibbxsZ84K0B9',
    'https://drive.google.com/uc?id=149e9mcO79DEemy3ZBHfyWOGZBybspVwq'
]

urls = [url.replace('open', 'uc') for url in urls]
for id, url in enumerate(urls):
    output = 'training_data.7z.00{}'.format(id + 1)
    print('Donwloading {} ...'.format(output))
    gdown.download(url, output, quiet=False)

