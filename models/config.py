"""
Consists all the config for training
"""

# Default backbone value
backbone = "VGG16"

# Anchor scales for RPN
ANCHOR_SCALES = [2, 1, 0.5]

# Anchor ratios for RPN
ANCHOR_RATIOS = [0.5, 1, 2]


NUM_CLASS = 20 # For pascal VOC
# training parameters
TRAIN_BATCH_SIZE = 16
IMAGE_SIZE = (512, 512)

# inference parameters
INF_BATCH_SIZE = 16

