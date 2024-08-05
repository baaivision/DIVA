from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class EmptyClass(PretrainedConfig):
    def __init__(self):
        pass
class SDConfig(PretrainedConfig):

    def __init__(self,
                    sd_version = '2-1',
                    override_total_steps = -1,
                    freeze_class_embeds = True,
                    freeze_vae = False,
                    use_flash = False,
                    adapt_only_classifier = True,
                    adapt_topk = -1,
                    loss = 'mse',
                    actual_bs = 16,
                    mean = [0.485, 0.456, 0.406],
                    std = [0.229, 0.224, 0.225],
                    use_same_noise_among_timesteps = False,
                    random_timestep_per_iteration = True, 
                    rand_timestep_equal_int = False,
                    weight_decay = 0,
                    train_steps = 1,
                    accum_iter = 1,
                    optimizer = 'sgd',
                    optimizer_momentum = 0.9,
                    pred_noise_batch_size = 1,
                    output_dir = './outputs/First_Start',
                    visual_pattern = None,
                    clip_image_size = 224,
                    metaclip_version = 1
    ):
        super().__init__()
        self.model = EmptyClass()
        self.model.sd_version = sd_version
        self.model.override_total_steps = override_total_steps
        self.model.freeze_class_embeds = freeze_class_embeds
        self.model.freeze_vae = freeze_vae
        self.model.use_flash = use_flash
        self.model.adapt_only_classifier = adapt_only_classifier
        self.tta = EmptyClass()
        self.tta.gradient_descent = EmptyClass()
        self.tta.adapt_topk = adapt_topk
        self.tta.loss = loss
        self.tta.use_same_noise_among_timesteps = use_same_noise_among_timesteps
        self.tta.random_timestep_per_iteration = random_timestep_per_iteration 
        self.tta.rand_timestep_equal_int = rand_timestep_equal_int
        self.tta.gradient_descent.weight_decay = weight_decay
        self.tta.gradient_descent.train_steps = train_steps
        self.tta.gradient_descent.accum_iter = accum_iter
        self.tta.gradient_descent.optimizer = optimizer
        self.tta.gradient_descent.optimizer_momentum = optimizer_momentum
        self.input = EmptyClass()
        self.input.batch_size = pred_noise_batch_size
        self.input.mean = mean
        self.input.std = std
        self.output_dir = output_dir
        self.actual_bs = actual_bs
        self.visual_pattern = visual_pattern
        self.clip_image_size = clip_image_size
        self.metaclip_version = metaclip_version

if __name__ =='__main__':
    SDConfig()