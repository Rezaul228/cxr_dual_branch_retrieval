"""
Advanced augmentation configuration for chest X-ray and report augmentation.
Contains configuration parameters for more aggressive and diverse augmentations.
"""

class AdvAugConfig:
    """Advanced configuration for medical image and text augmentation"""
    def __init__(self):
        # Image augmentation parameters - more aggressive
        self.translation_range = 0.2      # Maximum horizontal/vertical shift (fraction of image size)
        self.rotation_range = 15.0        # Maximum rotation in degrees
        self.brightness_range = 0.3       # Maximum brightness adjustment
        self.contrast_range = 0.3         # Maximum contrast adjustment
        self.noise_level = 0.03           # Maximum Gaussian noise level
        self.zoom_range = 0.2             # Maximum zoom factor (1±zoom_range)
        self.elastic_deform_alpha = 50    # Elastic deformation control parameter
        self.elastic_deform_sigma = 5     # Elastic deformation smoothness parameter
        self.use_elastic_deform = True    # Whether to apply elastic deformations
        
        # Text augmentation parameters - more diverse
        self.synonym_replacement_prob = 0.4    # Probability of replacing medical terms
        self.sentence_restructure_prob = 0.4   # Probability of restructuring sentences
        self.terminology_style_prob = 0.3      # Probability of changing terminology style
        self.finding_order_prob = 0.3          # Probability of reordering findings
        self.certainty_modifier_prob = 0.3     # Probability of modifying certainty language
        
        # General parameters
        self.random_seed = 42              # For reproducibility
        self.num_augmentations = 8         # Augmentations per original sample
        self.visualize_samples = 3         # Number of samples to visualize
        self.augmentation_weighting = 0.8  # Weight for augmented samples vs original (1.0 = equal)
        
    def to_dict(self):
        """Convert configuration to dictionary for saving"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config_dict):
        """Create configuration from dictionary"""
        config = cls()
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config
