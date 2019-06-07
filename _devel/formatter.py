"""
Usage: cat pytorch_vision_vgg.md | python formatter.py | notedown >pytorch_vision_vgg.ipynb
"""
import sys
import yaml

header = []
markdown = []
header_read = False
with open('/dev/stdin', 'r') as input, open('/dev/stdout', 'w') as output:
    for line in input:
        if line.startswith('---'):
            header_read = not header_read
            continue
        if header_read == True:
            header += [line]
        else:
            markdown += [line]

    header = yaml.load(''.join(header))

    images = []
    if header['featured_image_1'] != 'no-image':
        images.append(header['featured_image_1'])
    if header['featured_image_2'] != 'no-image':
        images.append(header['featured_image_2'])

    pre = []
    pre += ['# ' + header['title'] + '\n\n']
    pre += ['*Author: ' + header['author'] + '*' + '\n\n']
    pre += ['**' + header['summary'] + '**' + '\n\n']

    if len(images) == 2:
        pre += ['_ | _\n']
        pre += ['- | -\n']
        pre += ['![alt](https://pytorch.org/assets/images/{}) | '
                '![alt](https://pytorch.org/assets/images/{})\n\n'.format(*images)]
    elif len(images) == 1:
        pre += ['![alt](https://pytorch.org/assets/images/{})\n\n'.format(*images)]

    markdown = pre + markdown
    output.write(''.join(markdown))
