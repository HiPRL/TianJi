# -*- coding: utf-8 -*-
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.clip_action import ClipAction
from gym.wrappers.filter_observation import FilterObservation
from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers.frame_stack import FrameStack, LazyFrames
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.monitor import Monitor
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers.rescale_action import RescaleAction
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.time_aware_observation import TimeAwareObservation
from gym.wrappers.time_limit import TimeLimit
from gym.wrappers.transform_observation import TransformObservation
from gym.wrappers.transform_reward import TransformReward

# don't support wrapper
# from gym.wrappers.normalize import NormalizeObservation, NormalizeReward
# from gym.wrappers.record_video import RecordVideo, capped_cubic_video_schedule

# import atari_wrappers models
from env.gym_env.atari_wrappers import *


__all__ = ['wrapper_table']



_gym_atari_wrappers = [wrap_deepmind, MonitorEnv, NoopResetEnv, FireResetEnv, EpisodicLifeEnv,
                       MaxAndSkipEnv, ClipRewardEnv, ClipActionsWrapper, WarpFrame, ScaledFloatFrame, FrameStackOrder]

_gym_wrappers = [AtariOriginalReward, AtariPreprocessing, ClipAction, FilterObservation, FlattenObservation, FrameStack,
                 LazyFrames, GrayScaleObservation, Monitor, PixelObservationWrapper, RecordEpisodeStatistics,
                 RescaleAction, ResizeObservation, TimeAwareObservation, TimeLimit, TransformObservation, TransformReward]

wrapper_table = {}
for item in _gym_wrappers + _gym_atari_wrappers:
    wrapper_table[item.__name__] = item