# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified logging module supporting both TensorBoard and Weights & Biases (wandb).

Usage:
    from flow_grpo.utils.logger import Logger
    
    # Create logger (defaults to wandb)
    logger = Logger(
        log_dir="/path/to/logs",
        run_name="my_run",
        config=config_dict,
        use_wandb=True,  # Default
        use_tensorboard=False,
        project="flow-grpo",
    )
    
    # Log scalars
    logger.add_scalar("train/loss", 0.5, step=100)
    
    # Log images
    logger.add_image("train/sample", image_tensor, step=100)
    
    # Log text
    logger.add_text("train/prompt", "A cat sitting on a mat", step=100)
    
    # Close logger
    logger.close()
"""

import os
import logging
from typing import Optional, Dict, Any, Union
import numpy as np

import torch

logger = logging.getLogger(__name__)


def chown_recursive(path: str, username: str = "yilun.lt"):
    """
    递归修改目录及其所有内容的所有者为指定用户。
    这样可以确保在分布式任务中以 root 身份创建的文件可以被普通用户删除。
    
    Args:
        path: 要修改的文件或目录路径
        username: 目标用户名
    """
    try:
        pw = pwd.getpwnam(username)
        uid, gid = pw.pw_uid, pw.pw_gid
        
        if os.path.isfile(path):
            os.chown(path, uid, gid)
        elif os.path.isdir(path):
            os.chown(path, uid, gid)
            for root, dirs, files in os.walk(path):
                for d in dirs:
                    try:
                        os.chown(os.path.join(root, d), uid, gid)
                    except (PermissionError, OSError) as e:
                        logger.debug(f"Cannot chown directory {os.path.join(root, d)}: {e}")
                for f in files:
                    try:
                        os.chown(os.path.join(root, f), uid, gid)
                    except (PermissionError, OSError) as e:
                        logger.debug(f"Cannot chown file {os.path.join(root, f)}: {e}")
        logger.info(f"Changed ownership of {path} to {username}")
    except KeyError:
        logger.warning(f"User {username} not found, skipping chown")
    except PermissionError:
        logger.warning(f"Permission denied when trying to chown {path}")
    except Exception as e:
        logger.warning(f"Failed to chown {path}: {e}")


class Logger:
    """Unified logger supporting TensorBoard and wandb."""
    
    def __init__(
        self,
        log_dir: str,
        run_name: str,
        config: Optional[Dict[str, Any]] = None,
        use_wandb: bool = True,
        use_tensorboard: bool = False,
        project: str = "flow-grpo",
        entity: Optional[str] = None,
        api_key: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
    ):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory for saving logs
            run_name: Name of the run
            config: Configuration dictionary to log
            use_wandb: Whether to use wandb (default: True)
            use_tensorboard: Whether to use TensorBoard (default: False)
            project: wandb project name
            entity: wandb entity (team or username)
            api_key: wandb API key (optional, can also use WANDB_API_KEY env var)
            tags: List of tags for wandb
            notes: Notes for wandb run
        """
        self.log_dir = log_dir
        self.run_name = run_name
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        self._wandb_run = None
        self._tb_writer = None
        
        os.makedirs(log_dir, exist_ok=True)
        
        if use_wandb:
            self._init_wandb(
                project=project,
                entity=entity,
                api_key=api_key,
                config=config,
                tags=tags,
                notes=notes,
            )
        
        if use_tensorboard:
            self._init_tensorboard()
        
        if config is not None:
            self._log_config(config)
    
    def _init_wandb(
        self,
        project: str,
        entity: Optional[str],
        api_key: Optional[str],
        config: Optional[Dict[str, Any]],
        tags: Optional[list],
        notes: Optional[str],
    ):
        """Initialize wandb."""
        try:
            import wandb
            
            # Login with API key if provided
            if api_key:
                logger.info("Logging in to wandb with provided API key...")
                wandb.login(key=api_key, relogin=True)
            
            logger.info(f"Initializing wandb with project={project}, entity={entity}, run_name={self.run_name}")
            
            self._wandb_run = wandb.init(
                project=project,
                entity=entity,
                name=self.run_name,
                config=config,
                dir=self.log_dir,
                tags=tags,
                notes=notes,
                reinit=True,
            )
            logger.info(f"wandb initialized successfully: {wandb.run.url}")
        except ImportError:
            logger.warning("wandb not installed. Please install with: pip install wandb")
            logger.warning("Falling back to TensorBoard only.")
            self.use_wandb = False
            self.use_tensorboard = True
            self._init_tensorboard()
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            logger.warning("Falling back to TensorBoard. To use wandb, please run: wandb login")
            self.use_wandb = False
            self.use_tensorboard = True
            self._init_tensorboard()
    
    def _init_tensorboard(self):
        """Initialize TensorBoard."""
        from torch.utils.tensorboard import SummaryWriter
        self._tb_writer = SummaryWriter(log_dir=self.log_dir)
        print(f"[Logger] TensorBoard initialized, logging to {self.log_dir}")
        logger.info(f"TensorBoard initialized, logging to {self.log_dir}")
    
    def _log_config(self, config: Dict[str, Any]):
        """Log configuration."""
        import json
        
        if self.use_tensorboard and self._tb_writer is not None:
            config_str = f"```json\n{json.dumps(config, indent=2, default=str)}\n```"
            self._tb_writer.add_text("config", config_str, 0)
    
    def add_scalar(
        self,
        tag: str,
        value: Union[float, int, torch.Tensor],
        step: int,
    ):
        """
        Log a scalar value.
        
        Args:
            tag: Name of the metric (e.g., "train/loss")
            value: Scalar value to log
            step: Global step
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            wandb.log({tag: value}, step=step)
        
        if self.use_tensorboard and self._tb_writer is not None:
            self._tb_writer.add_scalar(tag, value, step)
    
    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, Union[float, int, torch.Tensor]],
        step: int,
    ):
        """
        Log multiple scalars under a main tag.
        
        Args:
            main_tag: Main tag prefix
            tag_scalar_dict: Dictionary of tag -> value
            step: Global step
        """
        for tag, value in tag_scalar_dict.items():
            full_tag = f"{main_tag}/{tag}" if main_tag else tag
            self.add_scalar(full_tag, value, step)
    
    def add_image(
        self,
        tag: str,
        img_tensor: torch.Tensor,
        step: int,
        caption: Optional[str] = None,
    ):
        """
        Log an image.
        
        Args:
            tag: Name of the image
            img_tensor: Image tensor (C, H, W) with values in [0, 1] or [0, 255]
            step: Global step
            caption: Optional caption for the image
        """
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            
            if img_tensor.dim() == 3:
                img_np = img_tensor.cpu().numpy()
                if img_np.shape[0] in [1, 3, 4]:
                    img_np = np.transpose(img_np, (1, 2, 0))
                # Check if image is in [0, 1] range (float images typically have max <= 1.0 and are float type)
                # Use both max value and dtype to determine if scaling is needed
                if img_np.max() <= 1.0 and img_np.dtype in [np.float32, np.float64, np.float16]:
                    img_np = (img_np * 255).astype(np.uint8)
                elif img_np.dtype != np.uint8:
                    img_np = img_np.astype(np.uint8)
                
                wandb_image = wandb.Image(img_np, caption=caption)
                wandb.log({tag: wandb_image}, step=step)
        
        if self.use_tensorboard and self._tb_writer is not None:
            self._tb_writer.add_image(tag, img_tensor, step)

    def add_image_table(
        self,
        tag: str,
        img_tensors: list,
        step: int,
        captions: Optional[list] = None,
        prompts: Optional[list] = None,
        rewards: Optional[list] = None,
        resize_to: int = 512,
    ):
        """
        Log multiple images as a W&B Table for synchronized viewing across steps.
        
        All images are saved in a single row with each image as a separate column,
        allowing easy comparison across all images at the same step.
        
        Args:
            tag: Name of the table (e.g., "train_images")
            img_tensors: List of image tensors, each (C, H, W) with values in [0, 1]
            step: Global step
            captions: Optional list of captions for each image
            prompts: Optional list of prompts for each image
            rewards: Optional list of reward values for each image
            resize_to: Target size to resize images (default: 512, images will be resized to resize_to x resize_to)
        """
        # Resize all images to target size (512x512 by default)
        if resize_to is not None and len(img_tensors) > 0:
            import torch.nn.functional as F
            resized_tensors = []
            for img_tensor in img_tensors:
                if img_tensor.dim() == 3:
                    # Add batch dimension for interpolate: (C, H, W) -> (1, C, H, W)
                    img_4d = img_tensor.unsqueeze(0)
                    resized = F.interpolate(img_4d, size=(resize_to, resize_to), mode='bilinear', align_corners=False)
                    resized_tensors.append(resized.squeeze(0))  # Remove batch dimension
                else:
                    resized_tensors.append(img_tensor)
            img_tensors = resized_tensors
        
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            
            # Build columns: each image gets its own column (image_0, image_1, ...)
            columns = []
            for i in range(len(img_tensors)):
                columns.append(f"image_{i}")
            
            # Add optional metadata columns at the end
            if prompts is not None:
                for i in range(len(img_tensors)):
                    columns.append(f"prompt_{i}")
            if rewards is not None:
                for i in range(len(img_tensors)):
                    columns.append(f"reward_{i}")
            if captions is not None:
                for i in range(len(img_tensors)):
                    columns.append(f"caption_{i}")
            
            table = wandb.Table(columns=columns)
            
            # Build single row with all images as columns
            row = []
            
            # Add all images
            for i, img_tensor in enumerate(img_tensors):
                if img_tensor.dim() == 3:
                    img_np = img_tensor.cpu().numpy()
                    if img_np.shape[0] in [1, 3, 4]:
                        img_np = np.transpose(img_np, (1, 2, 0))
                    # Check if image is in [0, 1] range (float images typically have max <= 1.0 and are float type)
                    if img_np.max() <= 1.0 and img_np.dtype in [np.float32, np.float64, np.float16]:
                        img_np = (img_np * 255).astype(np.uint8)
                    elif img_np.dtype != np.uint8:
                        img_np = img_np.astype(np.uint8)
                    
                    wandb_image = wandb.Image(img_np)
                    row.append(wandb_image)
                else:
                    row.append(None)
            
            # Add optional metadata for each image
            if prompts is not None:
                for i in range(len(img_tensors)):
                    row.append(prompts[i] if i < len(prompts) else "")
            if rewards is not None:
                for i in range(len(img_tensors)):
                    row.append(rewards[i] if i < len(rewards) else 0.0)
            if captions is not None:
                for i in range(len(img_tensors)):
                    row.append(captions[i] if i < len(captions) else "")
            
            table.add_data(*row)
            
            wandb.log({tag: table}, step=step)
        
        if self.use_tensorboard and self._tb_writer is not None:
            # Stack all images into a single tensor and display as a grid
            # This allows synchronized viewing across steps in TensorBoard
            if len(img_tensors) > 0:
                from torchvision.utils import make_grid
                
                # Ensure all tensors have the same shape and are on CPU
                processed_tensors = []
                for img_tensor in img_tensors:
                    if img_tensor.dim() == 3:
                        processed_tensors.append(img_tensor.cpu().float())
                
                if processed_tensors:
                    # Stack into (N, C, H, W) format
                    stacked = torch.stack(processed_tensors, dim=0)
                    # Create grid with labels
                    # nrow controls how many images per row
                    nrow = min(8, len(processed_tensors))  # Max 8 images per row
                    grid = make_grid(stacked, nrow=nrow, padding=4, normalize=False)
                    self._tb_writer.add_image(tag, grid, step)
                    
                    # Build text info for each image (combine prompts, rewards, captions)
                    text_lines = []
                    num_images = len(processed_tensors)
                    for i in range(num_images):
                        parts = [f"**[{i}]**"]
                        # Add prompt
                        if prompts is not None and i < len(prompts) and prompts[i]:
                            parts.append(f"**Prompt:** {prompts[i]}")
                        # Add reward
                        if rewards is not None and i < len(rewards):
                            parts.append(f"**Reward:** {rewards[i]:.4f}")
                        # Add caption (e.g., Student/Teacher label)
                        if captions is not None and i < len(captions) and captions[i]:
                            parts.append(f"**Type:** {captions[i]}")
                        if len(parts) > 1:  # Has content beyond just the index
                            text_lines.append("  \n".join(parts))  # Use Markdown line break
                    
                    if text_lines:
                        # Use double newline + horizontal rule for clear separation between images
                        full_text = "\n\n---\n\n".join(text_lines)
                        self._tb_writer.add_text(f"{tag}_info", full_text, step)
    
    def add_images(
        self,
        tag: str,
        img_tensors: torch.Tensor,
        step: int,
        captions: Optional[list] = None,
    ):
        """
        Log multiple images.
        
        Args:
            tag: Name prefix for images
            img_tensors: Batch of image tensors (N, C, H, W)
            step: Global step
            captions: Optional list of captions
        """
        for i, img in enumerate(img_tensors):
            caption = captions[i] if captions and i < len(captions) else None
            self.add_image(f"{tag}/{i}", img, step, caption=caption)
    
    def add_text(
        self,
        tag: str,
        text: str,
        step: int,
    ):
        """
        Log text.
        
        Args:
            tag: Name of the text entry
            text: Text content
            step: Global step
        """
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            wandb.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=step)
        
        if self.use_tensorboard and self._tb_writer is not None:
            self._tb_writer.add_text(tag, text, step)
    
    def add_histogram(
        self,
        tag: str,
        values: Union[torch.Tensor, np.ndarray],
        step: int,
    ):
        """
        Log a histogram.
        
        Args:
            tag: Name of the histogram
            values: Values to create histogram from
            step: Global step
        """
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            wandb.log({tag: wandb.Histogram(values)}, step=step)
        
        if self.use_tensorboard and self._tb_writer is not None:
            self._tb_writer.add_histogram(tag, values, step)
    
    def log_dict(
        self,
        metrics: Dict[str, Any],
        step: int,
        prefix: str = "",
    ):
        """
        Log a dictionary of metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Global step
            prefix: Optional prefix for all metric names
        """
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            if isinstance(value, (int, float)) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                self.add_scalar(tag, value, step)
    
    def watch_model(
        self,
        model: torch.nn.Module,
        log_freq: int = 1000,
    ):
        """
        Watch model gradients and parameters (wandb only).
        
        Args:
            model: PyTorch model to watch
            log_freq: Frequency of logging
        """
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            wandb.watch(model, log_freq=log_freq)
    
    def save_artifact(
        self,
        name: str,
        artifact_type: str,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save an artifact (wandb only).
        
        Args:
            name: Name of the artifact
            artifact_type: Type of artifact (e.g., "model", "dataset")
            path: Path to the artifact file or directory
            metadata: Optional metadata dictionary
        """
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
            if os.path.isdir(path):
                artifact.add_dir(path)
            else:
                artifact.add_file(path)
            wandb.log_artifact(artifact)
    
    def close(self):
        """Close the logger and finish logging."""
        if self.use_wandb and self._wandb_run is not None:
            import wandb
            wandb.finish()
            logger.info("wandb run finished")
        
        if self.use_tensorboard and self._tb_writer is not None:
            self._tb_writer.close()
            logger.info(f"TensorBoard logs saved to {self.log_dir}")
        
        # Log final status
        active_loggers = []
        if self.use_wandb and self._wandb_run is not None:
            active_loggers.append("wandb")
        if self.use_tensorboard and self._tb_writer is not None:
            active_loggers.append("tensorboard")
        if active_loggers:
            logger.info(f"Logging completed with: {', '.join(active_loggers)}")
    
    @property
    def tb_writer(self):
        """Get the TensorBoard writer (for backward compatibility)."""
        return self._tb_writer
    
    @property
    def wandb_run(self):
        """Get the wandb run object."""
        return self._wandb_run


def create_logger(
    log_dir: str,
    run_name: str,
    config: Optional[Dict[str, Any]] = None,
    logger_type: str = "wandb",
    project: str = "flow-grpo",
    entity: Optional[str] = None,
    api_key: Optional[str] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
) -> Logger:
    """
    Factory function to create a logger.
    
    Args:
        log_dir: Directory for saving logs
        run_name: Name of the run
        config: Configuration dictionary
        logger_type: Type of logger ("wandb", "tensorboard", or "both")
        project: wandb project name
        entity: wandb entity
        api_key: wandb API key (get from https://wandb.ai/authorize)
        tags: Tags for wandb
        notes: Notes for wandb
    
    Returns:
        Logger instance
    """
    use_wandb = logger_type in ["wandb", "both"]
    use_tensorboard = logger_type in ["tensorboard", "both"]
    
    return Logger(
        log_dir=log_dir,
        run_name=run_name,
        config=config,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
        project=project,
        entity=entity,
        api_key=api_key,
        tags=tags,
        notes=notes,
    )


def log_fake_teacher_metrics(exp_logger, ft_step, ft_diffusion_loss, ft_sigma,
                              ft_diffusion_info, ft_grad_stats, ft_lr, ft_batch_size):
    """Log all Fake Teacher metrics under the 'fake_teacher/' namespace.

    Args:
        exp_logger: Logger instance (from create_logger).
        ft_step: Current fake teacher training step.
        ft_diffusion_loss: Diffusion loss tensor.
        ft_sigma: Sigma tensor used for this step.
        ft_diffusion_info: Dict of additional diffusion info metrics.
        ft_grad_stats: Dict with optional 'grad_norm' key.
        ft_lr: Current learning rate.
        ft_batch_size: Batch size used for this step.
    """
    import torch

    exp_logger.add_scalar("fake_teacher/step", ft_step, ft_step)
    exp_logger.add_scalar("fake_teacher/diffusion_loss", ft_diffusion_loss.item(), ft_step)
    exp_logger.add_scalar("fake_teacher/learning_rate", ft_lr, ft_step)
    exp_logger.add_scalar("fake_teacher/batch_size", ft_batch_size, ft_step)
    exp_logger.add_scalar("fake_teacher/sigma_mean", ft_sigma.mean().item(), ft_step)
    exp_logger.add_scalar("fake_teacher/grad_norm", ft_grad_stats.get("grad_norm", 0.0), ft_step)
    for k, v in ft_diffusion_info.items():
        exp_logger.add_scalar(f"fake_teacher/{k}", v.item() if isinstance(v, torch.Tensor) else v, ft_step)
