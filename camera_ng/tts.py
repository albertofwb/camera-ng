#!/usr/bin/env python3
"""
TTS 模块 - XiaoxiaoTTS
封装 ~/.local/bin/xiaoxiao-tts 命令，供业务逻辑调用
"""

import os
import subprocess
import time
from pathlib import Path


class XiaoxiaoTTS:
    """晓晓 TTS 命令行封装"""

    COMMON_PROMPTS = {
        "收到": "ack.ogg",
        "开始录像": "record_start.ogg",
        "停止录像": "record_stop.ogg",
        "找到目标，开始录像": "target_found_record_start.ogg",
        "丢失目标，停止录像": "target_lost_record_stop.ogg",
        "目标捕获": "target_captured.ogg",
        "目标丢失": "target_lost.ogg",
        "校准中": "calibrating.ogg",
        "校准完成": "calibration_done.ogg",
    }

    def __init__(
        self,
        command_path: str = "~/.local/bin/xiaoxiao-tts",
        media_dir: str = "",
    ):
        self.command_path = os.path.expanduser(command_path)
        if media_dir:
            self.media_dir = Path(media_dir).expanduser()
        else:
            self.media_dir = Path(__file__).resolve().parent.parent / "media" / "tts"

    def is_available(self) -> bool:
        return os.path.isfile(self.command_path) and os.access(self.command_path, os.X_OK)

    def get_cached_prompt_path(self, text: str) -> str:
        file_name = self.COMMON_PROMPTS.get(text)
        if not file_name:
            return ""
        return str(self.media_dir / file_name)

    def pregenerate_common_prompts(self, overwrite: bool = False) -> tuple[int, int]:
        """预生成常用提示音到 media/tts，返回 (成功数, 跳过数)"""
        self.media_dir.mkdir(parents=True, exist_ok=True)
        created = 0
        skipped = 0

        for text, file_name in self.COMMON_PROMPTS.items():
            out_path = self.media_dir / file_name
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            self.synthesize(text, str(out_path))
            created += 1

        return created, skipped

    def _play_file(self, audio_path: str) -> bool:
        players = [
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", audio_path],
            ["mpv", "--no-video", "--really-quiet", audio_path],
            ["paplay", audio_path],
        ]
        for cmd in players:
            try:
                result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if result.returncode == 0:
                    return True
            except FileNotFoundError:
                continue
            except Exception:
                continue
        return False

    def synthesize(self, text: str, output_path: str = "") -> str:
        """将文本合成为 OGG 语音并返回音频路径"""
        if not text:
            raise ValueError("text 不能为空")

        if not output_path:
            output_path = f"/tmp/xiaoxiao-{int(time.time() * 1000)}.ogg"

        if not self.is_available():
            raise RuntimeError(f"TTS 命令不可用: {self.command_path}")

        cmd = [self.command_path, text, output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"TTS 合成失败: {stderr}")

        out = (result.stdout or "").strip().splitlines()
        if out:
            return out[-1].strip()
        return output_path

    def playback(self, text: str) -> bool:
        """在本机播放中文语音（优先使用预生成本地音频）"""
        if not text:
            return False

        cached_path = self.get_cached_prompt_path(text)
        if cached_path and os.path.isfile(cached_path):
            if self._play_file(cached_path):
                return True

        cmd = [
            "edge-playback",
            "--voice",
            "zh-CN-XiaoxiaoNeural",
            "--text",
            text,
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
