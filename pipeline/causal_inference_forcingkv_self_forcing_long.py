from typing import List, Optional
import time

import torch

from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation

from .causal_inference_forcingkv import (
    CausalInferencePipeline_ForcingKV,
    _append_fps_record,
)


class CausalInferencePipeline_ForcingKV_Self_Forcing_Long(CausalInferencePipeline_ForcingKV):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__(
            args=args,
            device=device,
            generator=generator,
            text_encoder=text_encoder,
            vae=vae,
        )

        self.rollout_latent_frames = int(getattr(args, "rollout_latent_frames", 21))
        self.num_rollout = int(getattr(args, "num_rollout", 1))
        self.num_overlap_frames = int(getattr(args, "num_overlap_frames", self.num_frame_per_block))

        if self.rollout_latent_frames <= 0:
            raise ValueError("rollout_latent_frames must be positive")
        if self.num_rollout <= 0:
            raise ValueError("num_rollout must be positive")
        if self.num_overlap_frames <= 0:
            raise ValueError("num_overlap_frames must be positive")
        if self.num_overlap_frames >= self.rollout_latent_frames:
            raise ValueError("num_overlap_frames must be smaller than rollout_latent_frames")
        if self.rollout_latent_frames % self.num_frame_per_block != 0:
            raise ValueError(
                "rollout_latent_frames must be divisible by num_frame_per_block"
            )
        if self.num_overlap_frames % self.num_frame_per_block != 0:
            raise ValueError(
                "num_overlap_frames must be divisible by num_frame_per_block"
            )

        print(
            "Long forcingkv self forcing inference with "
            f"rollout_latent_frames={self.rollout_latent_frames}, "
            f"num_rollout={self.num_rollout}, "
            f"num_overlap_frames={self.num_overlap_frames}"
        )

    def _overlap_pixel_frames(self) -> int:
        return 4 * (self.num_overlap_frames - 1) + 1

    def _prepare_start_latents(
        self,
        video: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_overlap_frames == 1:
            boundary_frame = video[:, -1:, :, :, :]
        else:
            overlap_pixel_frames = self._overlap_pixel_frames()
            boundary_frame = video[:, -overlap_pixel_frames:-(overlap_pixel_frames - 1), :, :, :]

        boundary_frame = boundary_frame * 2.0 - 1.0
        boundary_frame = boundary_frame.permute(0, 2, 1, 3, 4).to(
            device=latents.device,
            dtype=latents.dtype,
        )
        start_frame = self.vae.encode_to_latent(boundary_frame).to(
            device=latents.device,
            dtype=latents.dtype,
        )

        if self.num_overlap_frames == 1:
            return start_frame

        return torch.cat([start_frame, latents[:, -(self.num_overlap_frames - 1):]], dim=1)

    def _infer_chunk(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        low_memory: bool = False,
    ):
        batch_size, num_frames, num_channels, height, width = noise.shape
        self.frame_seq_length = height * width // 4

        if num_frames % self.num_frame_per_block != 0:
            raise ValueError("Chunk noise length must be divisible by num_frame_per_block")

        num_blocks = num_frames // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        total_output_frames = num_frames + num_input_frames

        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output_device = torch.device("cpu") if low_memory else noise.device
        output = torch.zeros(
            [batch_size, total_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype,
        )

        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
        else:
            kv_cache_size = total_output_frames * self.frame_seq_length

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size,
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        self._set_all_modules_max_attention_size(self.local_attn_size)

        if initial_latent is not None:
            if num_input_frames % self.num_frame_per_block != 0:
                raise ValueError(
                    "initial_latent length must be divisible by num_frame_per_block"
                )
            timestep_zero = torch.zeros(
                [batch_size, self.num_frame_per_block],
                device=noise.device,
                dtype=torch.int64,
            )
            num_input_blocks = num_input_frames // self.num_frame_per_block
            for _ in range(num_input_blocks):
                current_ref_latents = initial_latent[
                    :, current_start_frame:current_start_frame + self.num_frame_per_block
                ]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents.to(output.device)
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep_zero,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        all_num_frames = [self.num_frame_per_block] * num_blocks
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :,
                current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames,
            ]

            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64,
                ) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames],
                            device=noise.device,
                            dtype=torch.long,
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            current_start_frame += current_num_frames

        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        return video, output.to(noise.device)

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        num_output_frames: int = 84,
        sample_idx: Optional[int] = None,
    ) -> torch.Tensor:
        target_latent_frames = int(num_output_frames)
        if target_latent_frames <= 0:
            raise ValueError("num_output_frames must be positive")
        if target_latent_frames % self.num_frame_per_block != 0:
            raise ValueError(
                "num_output_frames must be divisible by num_frame_per_block for forcingkv_self_forcing_long"
            )

        unique_latent_per_rollout = self.rollout_latent_frames - self.num_overlap_frames
        max_latent_capacity = self.rollout_latent_frames + max(self.num_rollout - 1, 0) * unique_latent_per_rollout
        if max_latent_capacity < target_latent_frames:
            raise ValueError(
                "Requested num_output_frames exceeds the configured long rollout capacity: "
                f"{target_latent_frames} > {max_latent_capacity}. "
                "Increase num_rollout or rollout_latent_frames."
            )

        batch_size, first_noise_frames, num_channels, height, width = noise.shape
        required_first_rollout_frames = min(self.rollout_latent_frames, target_latent_frames)
        if first_noise_frames < required_first_rollout_frames:
            raise ValueError(
                "The provided noise tensor is too short for the first rollout: "
                f"{first_noise_frames} < {required_first_rollout_frames}."
            )

        if profile:
            torch.cuda.synchronize(noise.device)
            start_time = time.perf_counter()

        remaining_latent_frames = target_latent_frames
        rollout_index = 0
        start_latents = None
        all_video = []
        all_latents = []

        while remaining_latent_frames > 0:
            if rollout_index >= self.num_rollout:
                raise RuntimeError(
                    "Ran out of configured rollouts before reaching the requested output length."
                )

            if rollout_index == 0:
                current_new_latent_frames = min(self.rollout_latent_frames, remaining_latent_frames)
                current_noise = noise[:, :current_new_latent_frames].contiguous()
            else:
                current_new_latent_frames = min(unique_latent_per_rollout, remaining_latent_frames)
                current_noise = torch.randn(
                    [batch_size, current_new_latent_frames, num_channels, height, width],
                    device=noise.device,
                    dtype=noise.dtype,
                )

            if current_new_latent_frames % self.num_frame_per_block != 0:
                raise ValueError(
                    "Each rollout must generate a multiple of num_frame_per_block latent frames. "
                    f"Got {current_new_latent_frames} with num_frame_per_block={self.num_frame_per_block}."
                )

            current_video, current_latents = self._infer_chunk(
                noise=current_noise,
                text_prompts=text_prompts,
                initial_latent=start_latents,
                low_memory=low_memory,
            )

            is_last_rollout = remaining_latent_frames == current_new_latent_frames
            if is_last_rollout:
                all_video.append(current_video)
            else:
                all_video.append(current_video[:, :-self._overlap_pixel_frames(), :, :, :])
                start_latents = self._prepare_start_latents(current_video, current_latents)

            if rollout_index == 0:
                all_latents.append(current_latents)
            else:
                all_latents.append(current_latents[:, self.num_overlap_frames:])

            remaining_latent_frames -= current_new_latent_frames
            rollout_index += 1

        video = torch.cat(all_video, dim=1)
        latents = torch.cat(all_latents, dim=1)
        if latents.shape[1] != target_latent_frames:
            raise RuntimeError(
                "Long forcingkv assembled an unexpected latent length: "
                f"{latents.shape[1]} != {target_latent_frames}"
            )

        if profile:
            torch.cuda.synchronize(noise.device)
            elapsed = time.perf_counter() - start_time
            elapsed_ms = elapsed * 1000.0
            latent_fps = target_latent_frames / max(elapsed, 1e-8)
            pixel_fps = video.shape[1] / max(elapsed, 1e-8)
            print("Long forcingkv profiling results:")
            print(f"  - Rollouts used: {rollout_index}")
            print(f"  - Total time: {elapsed_ms:.2f} ms")
            print(f"  - Latent FPS: {latent_fps:.2f}")
            print(f"  - Pixel FPS: {pixel_fps:.2f}")
            _append_fps_record(self.args.output_folder, sample_idx, pixel_fps)

        if return_latents:
            return video, latents
        return video
