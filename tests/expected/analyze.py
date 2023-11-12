"""
Tests expected values
"""

from src.allin1 import AnalysisResult
from src.allin1.typings import Segment

from pathlib import PosixPath


expected_short = AnalysisResult(
        path=PosixPath("/home/allin1_user/tests/data/test-chunk.wav"),
        bpm=162,
        beats=[
            0.34,
            0.73,
            1.09,
            1.47,
            1.85,
            2.25,
            2.61,
            3.02,
            3.41,
            3.78,
            4.15,
            4.54,
            4.93,
            5.31,
            5.68,
            6.07,
            6.43,
            6.83,
            7.19,
            7.58,
            7.96,
            8.34,
            8.7,
            9.08,
            9.45,
            9.83,
            10.19,
            10.58,
            10.95,
            11.34,
            11.69,
            12.08,
            12.44,
            12.84,
            13.2,
            13.59,
            13.97,
            14.36,
            14.73,
            15.1,
            15.46,
            15.84,
            16.21,
            16.59,
            16.94,
            17.33,
            17.7,
            18.09,
            18.48,
            18.85,
            19.22,
            19.58,
            19.94,
            20.33,
            20.7,
            21.09,
            21.46,
            21.83,
            22.19,
            22.58,
            22.93,
            23.31,
            23.67,
            24.05,
            24.4,
            24.78,
            25.14,
            25.52,
            25.88,
            26.27,
            26.64,
            27.01,
            27.36,
            27.74,
            28.11,
            28.49,
            28.84,
            29.21,
            29.58,
            29.96,
            30.33,
            30.68,
            31.04,
            31.43,
            31.79,
            32.16,
            32.52,
            32.91,
            33.26,
            33.64,
            34.01,
            34.39,
            34.74,
            35.12,
        ],
        downbeats=[
            0.34,
            1.85,
            3.41,
            4.93,
            6.43,
            7.96,
            9.45,
            10.95,
            12.44,
            13.97,
            15.46,
            16.94,
            18.48,
            19.94,
            21.46,
            22.93,
            24.4,
            25.88,
            27.36,
            28.84,
            30.33,
            31.79,
            33.26,
            34.74,
        ],
        beat_positions=[
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            4,
            1,
            2,
        ],
        segments=[
            Segment(start=0.0, end=0.33, label="start"),
            Segment(start=0.33, end=18.47, label="intro"),
            Segment(start=18.47, end=35.49, label="verse"),
        ],
        activations=None,
        embeddings=None,
    )