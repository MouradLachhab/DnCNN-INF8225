from train import train_model
from models import DnCNN
from utils import weights_init_kaiming
# Possible preprocess

# Create model
model = DnCNN(channels=1)
model.apply(weights_init_kaiming)