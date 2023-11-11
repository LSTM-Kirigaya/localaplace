import os
from time import time

import cv2
import localaplace as llp


image = cv2.imread('./images/ankle.png', cv2.IMREAD_UNCHANGED)

s = time()
result, layers = llp.local_laplace_filter(
    image, 0.2, 0.5, 0.8, 100, 
    num_workers=-1, 
    verbose=1, 
    return_layers=True
)

# save result to disk
if not os.path.exists('results'):
    os.makedirs('results')
cv2.imwrite('./results/ankle_enchanced.png', result)

# save inner layers to disk
for i, layer_image in enumerate(layers):
    cv2.imwrite(f'./results/layer_{i + 1}.png', layer_image)

cost_time = round(time() - s, 4)
print('cost time: {} s'.format(cost_time))