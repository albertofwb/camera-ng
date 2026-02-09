#!/usr/bin/env python3
"""
TTS 模块 - XiaoxiaoTTS
封装 ~/.local/bin/xiaoxiao-tts 命令，供业务逻辑调用
"""

import os
import subprocess
import time


class XiaoxiaoTTS:
    """晓晓 TTS 命令行封装"""

    def __init__(self, command_path: str = "~/.local/bin/xiaoxiao-tts"):
        self.command_path = os.path.expanduser(command_path)

    def is_available(self) -> bool:
        return os.path.isfile(self.command_path) and os.access(self.command_path, os.X_OK)

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
        """在本机播放中文语音（edge-playback）"""
        if not text:
            return False

        # 播放前强制取消静音并设置音量到 60%（尽力而为）
        volume_cmds = [
            ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"],
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "60%"],
            ["amixer", "-D", "pulse", "sset", "Master", "60%", "unmute"],
            ["amixer", "sset", "Master", "60%", "unmute"],
        ]
        for volume_cmd in volume_cmds:
            try:
                subprocess.run(volume_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

        cmd = [
            "edge-playback",
            "--voice",
            "zh-CN-XiaoxiaoNeural",
            "--text",
            text,
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
