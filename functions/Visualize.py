import random
from PIL import Image
import torchvision.transforms as transforms

def plot_grid_samples(dataset, grid_size = [10,10]):
    """ Plots a grid of random samples with grid size = grid size"""
    
    x, y = grid_size[0], grid_size[1]
    
    random_indices = random.sample(range(len(dataset)), x*y)
    images = [dataset.data[i] for i in random_indices]

    w, h = images[0].shape
    grid = Image.new('RGB', size=(x*w, y*h))
    transform = transforms.ToPILImage()

    for i, img in enumerate(images):
        grid.paste(transform(img), box=(i%x*w, i//y*h))
    grid.show()




