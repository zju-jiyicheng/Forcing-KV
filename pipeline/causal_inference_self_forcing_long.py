from typing import List, Optional
import time

import torch

from .causal_inference_selfforcing import (
    CausalInferencePipeline_Self_Forcing,
    _append_fps_record,
)


class CausalInferencePipeline_Self_Forcing_Long(CausalInferencePipeline_Self_Forcing):
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
            "Long self forcing inference with "
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

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        num_output_frames: int = 84,
        sample_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if initial_latent is not None:
            raise NotImplementedError(
                "self_forcing_long currently supports text-to-video only; "
                "initial_latent is not supported."
            )

        target_latent_frames = int(num_output_frames)
        if target_latent_frames <= 0:
            raise ValueError("num_output_frames must be positive")
        if target_latent_frames % self.num_frame_per_block != 0:
            raise ValueError(
                "num_output_frames must be divisible by num_frame_per_block for self_forcing_long"
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

            current_video, current_latents = super().inference(
                noise=current_noise,
                text_prompts=text_prompts,
                initial_latent=start_latents,
                return_latents=True,
                profile=False,
                low_memory=low_memory,
                num_output_frames=current_new_latent_frames,
                sample_idx=sample_idx,
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
                "Long self forcing assembled an unexpected latent length: "
                f"{latents.shape[1]} != {target_latent_frames}"
            )

        if profile:
            torch.cuda.synchronize(noise.device)
            elapsed = time.perf_counter() - start_time
            elapsed_ms = elapsed * 1000.0
            latent_fps = target_latent_frames / max(elapsed, 1e-8)
            pixel_fps = video.shape[1] / max(elapsed, 1e-8)
            print("Long self forcing profiling results:")
            print(f"  - Rollouts used: {rollout_index}")
            print(f"  - Total time: {elapsed_ms:.2f} ms")
            print(f"  - Latent FPS: {latent_fps:.2f}")
            print(f"  - Pixel FPS: {pixel_fps:.2f}")
            _append_fps_record(self.args.output_folder, sample_idx, pixel_fps)

        if return_latents:
            return video, latents
        return video
