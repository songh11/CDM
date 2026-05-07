import os
from pathlib import Path
from tkinter import TRUE
import ml_collections

# ==================== Constants ====================

# Project root: parent directory of the `config/` package, i.e. the repo root.
# All default paths below are anchored here so the project is portable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_LOG_DIR = str(PROJECT_ROOT / "logs")
BASE_DATASET_PATH = str(PROJECT_ROOT / "dataset")

VALID_DATASETS = ["pickscore", "ocr", "geneval", "ocr_small", "laion", "t2i_2M", "text_image_pair", "i2i_pair", "mixed_200k", "mixed_400k"]
VALID_BASE_MODELS = ["sd3", "longcat"]


# ==================== Utility Functions ====================

def get_num_gpus():
    """Get total GPU count from environment variables (nnodes * nproc_per_node)."""
    num_nodes = int(os.getenv('NNODES', '1'))
    num_procs_per_node = int(os.getenv('NPROC_PER_NODE', '1'))
    return num_nodes * num_procs_per_node


def get_config(name="sd3"):
    """Get config by name."""
    return globals()[name]()


def _get_training_params():
    """
    Get hierarchical training parameters.
    
    Args:
        student_use_lora: If True, use LoRA training params for student; otherwise full fine-tuning params.
    
    Returns:
        ml_collections.ConfigDict with hierarchical training parameters.
    """
    config = ml_collections.ConfigDict()
    
    # ==================== Reward Function Configuration ====================
    config.reward_fn = {
        # "ocr": 1.0,
        # "aesthetic": 1.0,
        # "clipscore": 1.0,
        # "pickscore": 1.0,
        # "hpsv2": 1.0,
    }
    
    # ==================== Common Training Configuration ====================
    config.batch_size = 16  # Unified batch size for sampling, student training, and fake teacher training
    config.gradient_accumulation_steps = 1
    config.max_grad_norm = 1.0
    # Common optimizer settings (shared by student and fake teacher)
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.999
    config.adam_weight_decay = 1e-4
    config.adam_epsilon = 1e-8
    
    # ==================== Global Memory Optimization ====================
    config.vae_slicing = True
    config.enable_xformers = True
    
    # ==================== Sigma Shift ====================
    # When True, applies the scheduler's timestep shifting (shift/mu/dynamic_shifting)
    # to training sigma sampling, ensuring consistency with inference sigma distributions.
    # Automatically reads shift parameters from the scheduler config.
    
    # ==================== Global Sigma Truncation ====================
    # All sigma sampling methods clamp output to [sigma_min, sigma_max] to avoid
    # numerical instability at extreme values (near 0 or 1).
    config.sigma_min = 0.02
    config.sigma_max = 0.98

    # ==================== Shared Latent Indices ====================
    config.share_latent_indices = False
    
    # ==================== Student Model Configuration ====================
    config.student = ml_collections.ConfigDict()
    config.student.use_lora = False
    config.student.lora_rank = 64
    config.student.lora_alpha = 128
    config.student.lora_include_ff = True  # Whether to include FFN linear layers in LoRA target modules
    config.student.lora_path = None
    config.student.warmup_steps = 100
    config.student.warmup_ratio = 0.05
    config.student.update_ratio = 5  # Student trains every N fake teacher steps (e.g., 5 means 1 student update per 5 FT updates)
    
    # Training parameters
    config.student.ema = True
    config.student.ema_decay = 0.999
    config.student.lr_scheduler_type = "constant"
    # Student memory optimization
    config.student.use_8bit_adam = False
    config.student.gradient_checkpointing = True
    
    config.student.learning_rate = 2e-5  # LoRA learning rate
    config.student.full_learning_rate = 5e-6  # Full fine-tuning learning rate
    
    # ==================== Fake Teacher Model Configuration ====================
    config.fake_teacher = ml_collections.ConfigDict()
    config.fake_teacher.enabled = True

    config.fake_teacher.use_lora = False
    config.fake_teacher.lora_rank = 64
    config.fake_teacher.lora_alpha = 128
    config.fake_teacher.lora_include_ff = True  # Whether to include FFN linear layers in LoRA target modules
    config.fake_teacher.learning_rate = 2e-5  # LoRA learning rate
    config.fake_teacher.full_learning_rate = 5e-6  # Full fine-tuning learning rate
    config.fake_teacher.gradient_checkpointing = True
    config.fake_teacher.use_8bit_adam = False
    config.fake_teacher.x0_source = "x0_next" # "x0_final" or "x0_current" or "x0_next"
    
    # Fake Teacher ← Student EMA Merge Configuration
    # After each FT gradient update, merge student parameters into FT via EMA:
    #   θ_ft = decay * θ_ft + (1 - decay) * θ_student
    config.fake_teacher.ema_from_student = True       # Whether to enable FT←Student EMA merge
    config.fake_teacher.ema_from_student_decay = 0.999  # EMA decay (higher = more FT retention)
    config.fake_teacher.ema_from_student_interval = 1   # Merge every N FT steps
    
    # Fake Teacher LR Scheduler Configuration
    config.fake_teacher.lr_scheduler_type = "constant"  # "constant", "constant_with_warmup", "linear", "cosine"
    config.fake_teacher.warmup_steps = 100  # Number of warmup steps (also used for DM loss warmup)
    config.fake_teacher.warmup_ratio = 0.0  # Warmup ratio (used when warmup_steps=0)
    
    # Sigma sampling for fake teacher
    config.fake_teacher.sigma_sampling_method = "uniform"
    config.fake_teacher.logit_mean = 0.0
    config.fake_teacher.logit_std = 1.0
    
    # ==================== Loss Configuration ====================
    config.loss = ml_collections.ConfigDict()
    
    # Fake Teacher's Diffusion Loss
    config.loss.diffusion = ml_collections.ConfigDict()
    config.loss.diffusion.weight = 1.0
    config.loss.diffusion.enabled = True
    
    # ==================== Three-way Loss Configuration ====================
    # Each loss branch has independent sigma sampling and can be enabled/disabled separately.
    
    # --- CFG: CFG loss with on_trajectory student forward ---
    # Student uses sampling trajectory xt + stop_sigma, teacher gets re-noised x0_pred
    config.loss.cfg = ml_collections.ConfigDict()
    config.loss.cfg.enabled = True
    config.loss.cfg.weight = 1.0
    config.loss.cfg.guidance_scale = 7.0  # Teacher CFG guidance scale for distillation
    config.loss.cfg.norm_clip_min = 0.1  # Minimum clamp for norm factor to prevent gradient explosion
    # "uniform", "logit_normal", "uniform_capped": U(0, stop_sigma)
    config.loss.cfg.teacher_sigma_method = "uniform_capped"
    config.loss.cfg.teacher_sigma_logit_mean = -1.0
    config.loss.cfg.teacher_sigma_logit_std = 1.0
    
    # --- DDM: DM loss with on_trajectory student forward ---
    # Student uses sampling trajectory xt + stop_sigma, teacher gets re-noised x0_pred
    config.loss.ddm = ml_collections.ConfigDict()
    config.loss.ddm.enabled = True
    config.loss.ddm.weight = 1.0
    config.loss.ddm.norm_clip_min = 0.1  # Minimum clamp for norm factor to prevent gradient explosion
    # "uniform", "logit_normal", "uniform_capped": U(0, stop_sigma)
    config.loss.ddm.teacher_sigma_method = "uniform"
    config.loss.ddm.teacher_sigma_logit_mean = 0.0
    config.loss.ddm.teacher_sigma_logit_std = 1.0
    # Teacher reuse student xt: when True, teacher directly reuses student's xt and sigma
    # instead of independently sampling sigma and constructing a new xt.
    # This ensures teacher evaluates at the exact same point as student, providing
    # stronger gradient signal but potentially introducing high-frequency noise coupling.
    # --- CDM: DM loss with fresh_noise student forward ---
    # Student uses independently noised x0_base, teacher reuses student's xt
    config.loss.cdm = ml_collections.ConfigDict()
    config.loss.cdm.enabled = False
    config.loss.cdm.weight = 1.0
    config.loss.cdm.norm_clip_min = 0.1  # Minimum clamp for norm factor to prevent gradient explosion
    # x0 base for constructing student/teacher xt. Decoupled: student and teacher can use different sources.
    config.loss.cdm.student_x0_base = "x0_final"  # "x0_final" or "x0_next"
    # teacher_x0_base options (must be explicitly set, no fallback):
    #   "x0_final"     -> use the final denoised latent from cached sampling trajectory
    #   "x0_next"      -> per-sample random x0 from cached trajectory (all_x0[1:])
    #   "student_pred" -> use student's online x0 prediction at this step
    #                     (cdm_student_xt - sigma_s * cdm_student_prediction).detach();
    #                     incompatible with teacher_xt_mode="share_student".
    config.loss.cdm.teacher_x0_base = "x0_final"  # "x0_final" | "x0_next" | "student_pred"
    config.loss.cdm.student_sigma_method = "uniform"  # "uniform", "logit_normal"
    config.loss.cdm.student_sigma_logit_mean = 0.0
    config.loss.cdm.student_sigma_logit_std = 1.0
    # Teacher sigma sampling (independent from student, used when teacher_xt_mode="fresh_independent")
    config.loss.cdm.teacher_sigma_method = "uniform"  # "uniform", "logit_normal"
    config.loss.cdm.teacher_sigma_logit_mean = 0.0
    config.loss.cdm.teacher_sigma_logit_std = 1.0
    # Student xt construction mode (decides how (xt, t) for the student forward is built):
    #   "fresh":          xt = (1 - σ_s) * x0_base + σ_s * ε_fresh  (σ_s independently sampled, ε independent)
    #   "trajectory_eps": xt = traj_xt + (σ_s - σ_traj) * traj_v
    #                     (σ_s independently sampled, noise direction reused from sampling trajectory;
    #                      preserves noise structure consistency between training and inference)
    #   "trajectory_xt":  xt = traj_xt, σ_s = σ_traj
    #                     (directly use the on-trajectory point; student sees exactly the (xt, t)
    #                      it would encounter at inference. student_sigma_method and related
    #                      params are ignored in this mode.)
    config.loss.cdm.student_xt_mode = "trajectory_eps"  # "fresh" | "trajectory_eps" | "trajectory_xt"
    # Teacher xt construction mode (decides how (xt, t) for the real-teacher forward is built):
    #   "share_student":    teacher reuses student's (xt, t) exactly.
    #   "fresh_shared_t":   xt = (1 - σ_s) * x0_base + σ_s * ε_fresh, t = student's t
    #                       (decouples teacher noise from student to avoid shared high-frequency artifacts,
    #                        but still evaluates teacher at the student's sigma).
    #   "fresh_independent": xt = (1 - σ_t) * x0_base + σ_t * ε_fresh, σ_t independently resampled
    #                       (teacher and student see different (xt, t); strongest decoupling).
    config.loss.cdm.teacher_xt_mode = "fresh_independent"  # "share_student" | "fresh_shared_t" | "fresh_independent"

    return config


def _get_sampling_config():
    """
    Get default sampling configuration.
    
    Args:
        num_epochs: Total number of training epochs (for DynaRS transition steps).
        gradient_steps_per_epoch: Number of gradient steps per epoch.
    """
    config = ml_collections.ConfigDict()
    
    # Basic sampling settings
    config.num_steps = 4
    config.guidance_scale = 1.0  # Student sampling guidance scale (usually 1.0 for distilled models)
    config.global_std = True
    config.deterministic = True  # Whether to use deterministic sampling
    
    # Sampling adapter selection
    config.sampling_adapter = "default"  # "old" or "default"
    
    # Number of groups for batch calculation
    config.num_image_per_prompt = 1 # 一个prompt会有多少张图像
    # 一个epoch采样的batch数目，一个epoch总的sample数量 = num_batches_per_epoch * train.batch_size * worldsize
    config.num_batches_per_epoch = 1
    
    # ==================== Random Sigma Schedule ====================
    # When enabled, randomizes the sigma schedule each epoch for ODE discretization
    # generalization. Instead of using a fixed custom_sigmas, each epoch samples a
    # random number of steps and random decreasing sigma values.
    # This forces the student to learn a v-prediction that generalizes across different
    # ODE discretization schemes, similar to data augmentation for the denoising schedule.
    config.random_sigma_schedule = False
    config.random_sigma_num_steps_min = 1  # Minimum number of denoising steps
    config.random_sigma_num_steps_max = 28   # Maximum number of denoising steps
    config.random_sigma_min = 0.25          # Minimum sigma value (avoid near-zero instability)
    config.random_sigma_max = 0.95          # Maximum sigma value for inner points (first sigma is always 1.0)
    
    return config


def _get_evaluation_config():
    """Get evaluation configuration."""
    config = ml_collections.ConfigDict()
    
    # Basic evaluation settings
    config.deterministic = True
    config.student_num_steps = 4  # Student inference steps for evaluation
    config.student_guidance_scale = 1.0
    config.skip_first_eval = False

    
    config.generate_teacher = True  # Whether to generate teacher images for comparison
    config.teacher_num_steps = 50  # Teacher inference steps
    config.teacher_guidance_scale = 4.5  # CFG guidance scale for evaluation

    # Distillation gap evaluation
    config.distillation_enabled = True
    config.eval_batch_size = 32
    config.max_eval_samples = 32 * get_num_gpus()
    
    # Image logging settings
    config.num_log_images = 8  # Number of images to log
    config.save_images = True  # Whether to save evaluation images to local disk
    # When set to a non-empty path, evaluation images will be saved under
    # "<image_save_root>/<run_name>/eval_images/step_<N>/<prefix>/" instead of
    # the experiment log directory. Leave empty to keep the legacy behavior
    # (saved next to the tensorboard log dir).
    config.image_save_root = f"{BASE_LOG_DIR}/eval_images"

    # ==================== Fake Teacher Visualization ====================
    config.fake_teacher_viz = ml_collections.ConfigDict()
    config.fake_teacher_viz.enabled = True  # Whether to enable fake teacher visualization
    config.fake_teacher_viz.num_steps = 50  # Number of inference steps for fake teacher
    # Note: fake teacher viz guidance_scale defaults to 1.0 (no CFG)
    config.fake_teacher_viz.num_images = 8 * get_num_gpus()      # 总共生成的图片数
    config.fake_teacher_viz.num_log_images = 8   # 每次 log 的图片数  

    # ==================== Evaluation Reward Function ====================
    config.reward_fn = {
        "aesthetic": 1.0,
        "clipscore": 1.0,
        "pickscore": 1.0,
        "hpsv2": 1.0,
    }

    return config

# ==================== Auto Run Name Generator ====================

def _generate_run_name(config):
    """Auto-generate run_name from config values, ensuring consistency with actual settings."""
    parts = []

    # Base model and resolution
    # resolution = f"{config.height}" if config.height == config.width else f"{config.height}x{config.width}"
    parts.append(f"{config.base_model} {config.seed} {config.num_epochs} {config.sample.random_sigma_num_steps_min} {config.sample.random_sigma_num_steps_max}")

    # Dataset (extract folder name from full path)
    parts.append(f"ft {config.train.fake_teacher.x0_source} random {config.sample.random_sigma_schedule} {config.sample.random_sigma_min} {config.sample.random_sigma_max}")

    # Loss branches
    cfg_enabled = config.train.loss.cfg.enabled
    ddm_enabled = config.train.loss.ddm.enabled
    cdm_enabled = config.train.loss.cdm.enabled
    parts.append(f"cfg {cfg_enabled} sigma {config.train.loss.cfg.teacher_sigma_method}")
    parts.append(f"ddm {ddm_enabled}")
    parts.append(f"cdm {cdm_enabled} stu_xt {config.train.loss.cdm.student_xt_mode} tea_xt {config.train.loss.cdm.teacher_xt_mode} {config.train.loss.cdm.teacher_x0_base} {config.train.loss.cdm.student_sigma_method}")

    # Student and fake teacher mode + learning rate
    stu_mode = "lora" if config.train.student.use_lora else "full"
    ft_mode = "lora" if config.train.fake_teacher.use_lora else "full"
    stu_lr = config.train.student.learning_rate if config.train.student.use_lora else config.train.student.full_learning_rate
    ft_lr = config.train.fake_teacher.learning_rate if config.train.fake_teacher.use_lora else config.train.fake_teacher.full_learning_rate
    if stu_lr == ft_lr:
        parts.append(f"stu {stu_mode} ratio {config.train.student.update_ratio} ft {ft_mode} {stu_lr}")
    else:
        parts.append(f"stu {stu_mode} {stu_lr} ft {ft_mode} {ft_lr}")


    return ", ".join(parts)

# ==================== Main Configuration Generator ====================

def _create_unified_config(
    base_model="sd3",
    sample_dataset="mixed_200k",
    eval_dataset="mixed_200k",
    experiment_name="",
):
    # Validate inputs
    assert base_model in VALID_BASE_MODELS, f"base_model must be one of {VALID_BASE_MODELS}"
    assert sample_dataset in VALID_DATASETS, f"sample_dataset must be one of {VALID_DATASETS}"
    assert eval_dataset in VALID_DATASETS, f"eval_dataset must be one of {VALID_DATASETS}"
    
    config = ml_collections.ConfigDict()
    
    # ==================== General Settings ====================
    config.seed = 42
    config.deterministic = False  # If True, enable full deterministic mode for reproducibility (slower)
    config.base_model = base_model
    
    # ==================== Paths ====================
    # Unified output directory: logs and checkpoints are stored together under
    config.output_dir = os.path.join(BASE_LOG_DIR, "experiments", base_model, experiment_name)
    
    # ==================== Logging ====================
    config.logger_type = "tensorboard"  # "wandb", "tensorboard", or "both"
    config.wandb_project = "distill-rl"
    config.wandb_entity = None
    # Leave empty: log in via `wandb login` instead of hard-coding the key here.
    config.wandb_api_key = ""
    
    # ==================== Training Schedule ====================
    config.num_epochs = 8001
    config.save_freq = 1000
    config.eval_freq = 400
    config.log_image_resolution = 1024  # Resolution for logged images (default: 512)
    
    # ==================== Precision & Performance ====================
    config.mixed_precision = "bf16"
    config.allow_tf32 = True
    config.attn_implementation = "flash_attention_2"
    
    # ==================== Checkpoint ====================
    config.resume_from = ""
    
    # ==================== Dataset ====================
    config.sample_dataset = os.path.join(BASE_DATASET_PATH, sample_dataset)
    config.eval_dataset = os.path.join(BASE_DATASET_PATH, eval_dataset)
    if base_model == "sd3":
        config.height = 1024
        config.width = 1024
    elif base_model == "longcat":
        config.height = 1024
        config.width = 1024
    else:
        config.height = 1024
        config.width = 1024
    config.max_sample_samples = 2000000
    # Base path for resolving image file paths (only used by TextImagePairDataset).
    # Empty string means fallback to the dataset directory itself.
    config.image_base_path = str(
        PROJECT_ROOT / "dataset" / "FreedomIntelligence_ShareGPT-4o-Image" / "text_to_image"
    )
    
    # Custom sigmas for denoising schedule (used by both training sampling and eval).
    # When set, these sigmas override the default uniform schedule from num_steps.
    # Example: [1.0, 0.75, 0.5, 0.25] for 4-step distillation with custom noise levels.
    # Set to None to use the default schedule derived from num_steps.
    config.custom_sigmas = [1.0, 0.75, 0.5, 0.25] # [1.0000, 0.8959, 0.7371, 0.6022]
    # ==================== Pretrained Model ====================
    config.pretrained = ml_collections.ConfigDict()
    if base_model == "longcat":
        config.pretrained.model = "meituan-longcat/LongCat-Image"
    else:
        config.pretrained.model = "stabilityai/stable-diffusion-3-medium-diffusers"

    config.pretrained.revision = ""
    
    # ==================== Prompt ====================
    _prompt_fn_map = {"geneval": "geneval", "text_image_pair": "text_image_pair", "i2i_pair": "image_edit"}
    config.prompt_fn = _prompt_fn_map.get(sample_dataset, "general_ocr")
    config.eval_prompt_fn = _prompt_fn_map.get(eval_dataset, "general_ocr")
    config.prompt_fn_kwargs = {}
    # ==================== Training Configuration (Hierarchical) ====================
    config.train = _get_training_params()
    
    # ==================== Sampling Configuration ====================
    config.sample = _get_sampling_config()

    # ==================== Evaluation Configuration ====================
    config.eval = _get_evaluation_config()
    
    # ==================== Post-Training Evaluation Configuration ====================
    # These settings are used by submit.py --auto_eval to run full benchmark
    # evaluation after training completes (prepare pipeline → generate → score).
    config.post_eval = ml_collections.ConfigDict()
    # Empty string means inherit CUDA_VISIBLE_DEVICES from the launch environment.
    config.post_eval.gpus = ""
    # 0 means auto-detect from CUDA_VISIBLE_DEVICES (or fall back to NPROC_PER_NODE).
    config.post_eval.nproc = 0
    config.post_eval.metrics = "imagereward clipscore pickscore hpsv2 hpsv3 aesthetic ocr fid dpgbench"
    config.post_eval.batch_size = 4
    config.post_eval.scoring_batch_size = 2
    config.post_eval.max_eval_samples = -1
    config.post_eval.diversity_num_prompts = 128
    config.post_eval.num_steps = 4
    config.post_eval.guidance_scale = 1.0
    config.post_eval.save_images = True
    config.post_eval.use_cache = False
    config.post_eval.no_ema = False
    config.post_eval.gen_env = "/mnt/fast-backup/liutao/envs/nft4"
    config.post_eval.eval_env = "/mnt/fast-backup/liutao/envs/cdm_eval_flash"
    # Output directories for pipeline and evaluation results.
    # Empty string means use default: logs/pipelines/<output_name>/ and logs/evaluations/<output_name>/
    config.post_eval.pipeline_output_root = str(PROJECT_ROOT / "logs" / "pipelines")
    config.post_eval.eval_output_root = str(PROJECT_ROOT / "logs" / "evaluations")
    # Number of characters from experiment basename to use as output directory name
    config.post_eval.output_name_length = 40
    
    return config


# ==================== Pre-defined Configurations ====================

def sd3():
    
    config = _create_unified_config(
        base_model="sd3",
        sample_dataset="mixed_200k",
        eval_dataset="mixed_200k",
    )
    config.eval.eval_batch_size = 16
    config.eval.max_eval_samples = 16 * get_num_gpus()
    config.eval.fake_teacher_viz.num_images = 16 * get_num_gpus()     # 总共生成的图片数

    config.train.student.full_learning_rate = 1e-5
    config.train.student.learning_rate = 1e-5
    config.train.fake_teacher.full_learning_rate = 5e-6
    config.train.fake_teacher.learning_rate = 5e-6


    config.train.student.use_lora = False
    config.train.fake_teacher.use_lora = False
    # config.eval.skip_first_eval = True
    config.seed = 42

    config.train.loss.cfg.enabled = True
    config.train.loss.ddm.enabled = True
    config.train.loss.cdm.enabled = True
    config.train.share_latent_indices = True

    config.train.loss.cfg.teacher_sigma_method = "uniform_capped" # "uniform" "uniform_capped" "logit_normal"
    config.train.loss.ddm.teacher_sigma_method = "uniform" # "uniform" "uniform_capped" "logit_normal"

    config.train.loss.cdm.student_sigma_method = "uniform" # "uniform" "uniform_capped" "logit_normal"
    config.train.loss.cdm.student_xt_mode = "trajectory_eps" # trajectory_xt "trajectory_eps"
    config.train.loss.cdm.teacher_xt_mode = "fresh_independent"
    config.train.loss.cdm.teacher_x0_base = "student_pred"  # "x0_final" | "x0_next" | "student_pred"

    config.sample.random_sigma_schedule = True

    config.deterministic = True
    config.train.batch_size = 16
    config.train.enable_xformers = False
    config.train.student.update_ratio = 2
    config.num_epochs = 4001
    config.save_freq = 2000
    config.eval_freq = 200


    config.run_name = "test"

    return config

def longcat():
    config = _create_unified_config(
        base_model="longcat",
        sample_dataset="mixed_200k",
        eval_dataset="mixed_200k",
    )

    config.eval.eval_batch_size = 8
    config.eval.max_eval_samples = 8 * get_num_gpus()
    config.eval.fake_teacher_viz.num_images = 8 * get_num_gpus()

    config.train.student.full_learning_rate = 1e-5
    config.train.student.learning_rate = 1e-5
    config.train.fake_teacher.full_learning_rate = 5e-6
    config.train.fake_teacher.learning_rate = 5e-6


    config.train.student.use_lora = False
    config.train.fake_teacher.use_lora = False
    # config.eval.skip_first_eval = True
    config.seed = 42

    config.train.loss.cfg.enabled = True
    config.train.loss.ddm.enabled = True
    config.train.loss.cdm.enabled = True
    config.train.share_latent_indices = True

    config.train.loss.cfg.teacher_sigma_method = "uniform_capped" # "uniform" "uniform_capped" "logit_normal"
    config.train.loss.ddm.teacher_sigma_method = "uniform" # "uniform" "uniform_capped" "logit_normal"

    config.train.loss.cdm.student_sigma_method = "uniform" # "uniform" "uniform_capped" "logit_normal"
    config.train.loss.cdm.student_xt_mode = "trajectory_eps" # trajectory_xt "trajectory_eps"
    config.train.loss.cdm.teacher_xt_mode = "fresh_independent"
    config.train.loss.cdm.teacher_x0_base = "student_pred"  # "x0_final" | "x0_next" | "student_pred"

    config.sample.random_sigma_schedule = True


    config.train.batch_size = 8
    config.deterministic = True
    config.train.student.update_ratio = 2 
    config.num_epochs = 2001
    config.save_freq = 1000
    config.eval_freq = 100


    # LongCat prompt rewrite: whether to enable Qwen2.5-VL prompt rewrite during evaluation and training sampling.
    config.eval.enable_prompt_rewrite = False

    config.run_name = _generate_run_name(config)

    return config