import asyncio
import base64
import json
import os
import re
import uuid
import wave

import aiohttp

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register


@register("AstrBot_Plugins_GeminiTTS", "长安某", "Gemini文本转语音工具", "1.0.0")
class GeminiTTSGenerator(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        self.api_keys = [
            key.strip() for key in self.config.get("gemini_api_keys", []) if key.strip()
        ]

        api_base_url_from_config = self.config.get("api_base_url")
        if api_base_url_from_config and api_base_url_from_config.strip():
            self.api_base_url = api_base_url_from_config.strip()
        else:
            self.api_base_url = "https://generativelanguage.googleapis.com"
            if api_base_url_from_config is not None:
                logger.info("配置中的 api_base_url 为空, 将使用默认官方地址")

        self.model_id = self.config.get("model_id", "gemini-1.5-flash-preview-tts")
        full_voice_name = self.config.get("default_voice", "Kore (女声 - 坚定自信)")
        self.default_voice = full_voice_name.split(" ")[0]
        default_prompt = (
            "你是一位顶级的专业配音总监你的任务是分析一段包含**对话**和**括号/星号内的动作或情绪描述**的文本，让最终输出的语音更自然、更具角色感\n\n"
            "**核心任务**:\n1.  仔细阅读并理解所有内容，特别是括号/星号内的描述，它们是决定语气、情感和语速的关键\n2.  **即使没有括号描述**，也要根据对话内容本身，**主动推断**出最合适的角色情绪和语气（例如，抱怨的、开心的、疑惑的）\n3.  **仅提取**出实际需要朗读的对话部分，**必须丢弃**所有括号/星号及其中的描述内容\n4.  为你提取的对话，生成一条**丰富且具体**的英文TTS指令指令应包含**情绪、风格、语气、语速**等多个维度\n5.  你的输出**必须**严格遵循格式：`[英文指令，描述风格、情绪和语速]: [提取出的对话文本]`\n\n--- \n**你可以使用的风格提示词示例 (请灵活运用和组合)**:\n*   **性格风格**: `tsundere` (傲娇), `energetic and cheerful` (元气), `lazy and sleepy` (慵懒), `gentle and soft` (温柔), `yandere` (病娇), `calm and indifferent` (冷淡).\n*   **情绪描述**: `annoyed` (恼怒的), `surprised` (惊讶的), `shy` (害羞的), `pouting` (撅着嘴的), `happy` (开心的), `sad` (悲伤的).\n*   **语气语速**: `in a high-pitched voice` (高音调), `in a soft whisper` (轻声细语), `at a faster pace` (语速加快), `with a rising intonation` (用上扬的语调), `drawn-out and impatient` (拖长声音且不耐烦).\n--- \n\n**输出示例**:\n输入: `(声音拔高，带着一丝难以置信的慌乱) 减…减半？！你这杂鱼在胡说什么？！`\n输出: `Read this in a high-pitched, panicked, and unbelieving tsundere voice: 减…减半？！你这杂鱼在胡说什么？！`\n\n输入: `哼，谁要你管了……（小声嘟囔）……笨蛋`\n输出: `Read this in an annoyed, tsundere tone, muttering the last word softly: 哼，谁要你管了……笨蛋`\n\n**请只输出最终的完整指令字符串**，不要包含任何额外解释、介绍或Markdown标记"
        )
        self.dubbing_director_prompt = self.config.get(
            "dubbing_director_prompt", default_prompt
        )

        self.current_key_index = 0
        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.plugin_dir, "conf.json")
        self.tts_settings = {}
        self._load_plugin_config()

        self.save_dir = os.path.join(self.plugin_dir, "temp_audio")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if not self.api_keys:
            logger.error("未配置任何 Gemini API 密钥，请在插件配置中添加")
        else:
            logger.info(f"已加载 {len(self.api_keys)} 个 Gemini API 密钥")

    def _load_plugin_config(self):
        default_config = {"tts_settings": {}}
        if not os.path.exists(self.config_path):
            self.tts_settings = default_config["tts_settings"]
            self._save_plugin_config()
        else:
            with open(self.config_path, "r", encoding="utf-8") as f:
                try:
                    plugin_conf = json.load(f)
                    self.tts_settings = plugin_conf.get("tts_settings", {})
                except (json.JSONDecodeError, AttributeError):
                    logger.warning(
                        f"配置文件 {self.config_path} 解析失败，将使用默认配置重置"
                    )
                    self.tts_settings = default_config["tts_settings"]
                    self._save_plugin_config()
        logger.info(f"会话配置已加载: {len(self.tts_settings)} 个会话有独立设置")

    def _save_plugin_config(self):
        config_data = {"tts_settings": self.tts_settings}
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
        logger.info("配置已保存")

    def _extract_dialogue(self, text: str) -> str:
        pattern = re.compile(r"\(.*?\)|（.*?）|\*.*?\*|【.*?】|\[.*?\]")
        return re.sub(pattern, "", text).strip()

    def _get_current_api_key(self):
        if not self.api_keys:
            return None
        return self.api_keys[self.current_key_index]

    def _switch_next_api_key(self):
        if not self.api_keys:
            return
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)

    async def _generate_tts_with_retry(self, text, voice_name=None):
        voice_name = voice_name or self.default_voice
        if not self.api_keys:
            logger.error("没有可用的API Key，无法生成语音")
            return None
        logger.info(f"准备为文本生成语音: '{text}'")
        max_attempts = len(self.api_keys)
        self.current_key_index = 0
        for attempt in range(max_attempts):
            current_key = self._get_current_api_key()
            if not current_key:
                break
            logger.debug(f"正在尝试使用 API Key 索引 {self.current_key_index} ...")
            try:
                audio_data = await self._generate_tts_manually(
                    text, voice_name, current_key
                )
                logger.info(
                    f"使用 API Key 索引 {self.current_key_index} 成功生成语音数据"
                )
                return audio_data
            except Exception as e:
                logger.warning(
                    f"使用API Key索引 {self.current_key_index} 失败: {e}正在尝试下一个Key..."
                )
                self._switch_next_api_key()
        logger.error(f"所有({max_attempts}个)API Key都无法成功生成语音，文本为: {text}")
        return None

    async def _generate_tts_manually(self, text, voice_name, api_key):
        base_url = self.api_base_url.rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            base_url = f"https://{base_url}"
        endpoint = (
            f"{base_url}/v1beta/models/{self.model_id}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice_name}}
                },
            },
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=endpoint, json=payload, headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                data = await response.json()
        audio_data_b64 = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("inlineData", {})
            .get("data")
        )
        if not audio_data_b64:
            raise Exception("API响应中未找到音频数据 (inline_data.data 为空)")
        return base64.b64decode(audio_data_b64)

    def _save_wav_file(self, audio_data, filename):
        save_path = os.path.join(self.save_dir, filename)
        logger.debug(f"准备将音频数据保存到: {save_path}")
        try:
            with wave.open(save_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_data)
            logger.info(f"音频文件成功保存于: {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"保存WAV文件到 {save_path} 时发生错误: {e}")
            return None

    @filter.command("tts_auto", alias={"gtts", "语音模式"})
    async def toggle_auto_tts(self, event: AstrMessageEvent, switch: str):
        group_id = event.message_obj.group_id
        session_id = (
            f"group_{group_id}" if group_id else f"user_{event.get_sender_id()}"
        )

        if switch.lower() == "on":
            self.tts_settings[session_id] = True
            yield event.plain_result("本会话的自动语音功能已开启")
        elif switch.lower() == "off":
            self.tts_settings[session_id] = False
            yield event.plain_result("本会话的自动语音功能已关闭")
        else:
            yield event.plain_result("指令错误，请输入 /tts_auto on 或 /tts_auto off")
            return
        self._save_plugin_config()

    @filter.on_llm_request()
    async def mark_llm_event(self, event: AstrMessageEvent, req: ProviderRequest):
        event._is_llm_response_pending_for_tts = True

    @filter.on_decorating_result()
    async def on_llm_reply_to_tts(self, event: AstrMessageEvent):
        try:
            group_id = event.message_obj.group_id
            session_id = (
                f"group_{group_id}" if group_id else f"user_{event.get_sender_id()}"
            )

            is_tts_enabled = self.tts_settings.get(session_id, False)

            if not (
                is_tts_enabled
                and getattr(event, "_is_llm_response_pending_for_tts", False)
            ):
                return

            result = event.get_result()
            chain = result.chain if result else []

            if len(chain) == 1 and isinstance(chain[0], Comp.Plain):
                original_text = chain[0].text
                if not original_text.strip():
                    return

                logger.info(f"会话 {session_id} 触发自动语音流程")
                fallback_text = self._extract_dialogue(original_text)
                text_for_tts = fallback_text

                llm_provider = self.context.get_using_provider()
                if not llm_provider:
                    logger.warning("无法获取LLM提供商，将使用清理后的文本生成默认语音")
                else:
                    try:
                        system_prompt = self.dubbing_director_prompt
                        user_prompt = f"请为以下内容生成配音指令: {original_text}"

                        logger.info("正在调用'配音总监'LLM以生成TTS指令...")
                        response = await llm_provider.text_chat(
                            prompt=user_prompt,
                            system_prompt=system_prompt,
                        )

                        if (
                            response
                            and response.role == "assistant"
                            and response.completion_text
                        ):
                            processed_text = response.completion_text.strip()
                            if re.match(r"^.+:\s*.+", processed_text):
                                text_for_tts = processed_text
                                logger.info(f"LLM成功生成TTS指令: {text_for_tts}")
                            else:
                                logger.warning(
                                    f"LLM响应格式不符合预期: '{processed_text}'，将使用清理后的文本"
                                )
                        else:
                            logger.warning("LLM未能生成有效指令，将使用清理后的文本")

                    except Exception as e:
                        logger.error(
                            f"调用LLM生成TTS指令时发生错误: {e}，将使用清理后的文本"
                        )

                if not text_for_tts:
                    logger.info("文本经处理后为空，不生成语音")
                    return

                save_path = None
                try:
                    audio_data = await self._generate_tts_with_retry(text_for_tts)
                    if audio_data:
                        file_name = f"{uuid.uuid4()}.wav"
                        save_path = self._save_wav_file(audio_data, file_name)
                        if save_path:
                            logger.info("语音已生成准备构建 Comp.Record 对象")
                            record_component = Comp.Record(file=save_path)
                            chain.append(record_component)
                            logger.info("Comp.Record 已成功添加到消息链，等待框架发送")
                        else:
                            logger.error(
                                "音频数据已生成，但保存为WAV文件时失败无法发送语音"
                            )
                    else:
                        logger.warning("语音生成失败，将只发送纯文本消息")
                finally:
                    if save_path:
                        asyncio.create_task(self._cleanup_temp_file(save_path))
        finally:
            if hasattr(event, "_is_llm_response_pending_for_tts"):
                delattr(event, "_is_llm_response_pending_for_tts")

    async def _cleanup_temp_file(self, path: str, delay: int = 10):
        await asyncio.sleep(delay)
        if os.path.exists(path):
            try:
                os.remove(path)
                logger.debug(f"已清理临时音频文件: {path}")
            except OSError as e:
                logger.warning(f"清理临时音频文件 {path} 失败: {e}")

    async def terminate(self):
        if os.path.exists(self.save_dir):
            logger.info("正在清理所有剩余的临时音频文件...")
            for filename in os.listdir(self.save_dir):
                file_path = os.path.join(self.save_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.warning(f"停用插件时清理文件 {file_path} 失败: {e}")
        logger.info("已停用")
