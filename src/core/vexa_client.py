"""Cliente da API Vexa para captura e transcricao de reunioes."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests

log = logging.getLogger(__name__)

VEXA_DEFAULT_URL = "https://api.vexa.ai"


@dataclass
class VexaSegment:
    """Segmento de transcricao do Vexa com identificacao de falante."""
    start_time: float
    end_time: float
    text: str
    speaker: str  # nome real do participante
    language: str = ""


@dataclass
class VexaTranscript:
    """Transcricao completa retornada pelo Vexa."""
    meeting_id: str
    platform: str
    start_time: datetime | None
    end_time: datetime | None
    segments: list[VexaSegment]
    recordings: list[dict]


class VexaClient:
    """Wrapper para a API REST do Vexa."""

    def __init__(self, api_key: str, base_url: str = VEXA_DEFAULT_URL):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    @property
    def _headers(self) -> dict:
        return {"X-API-Key": self._api_key}

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    # --- Bot management ---

    def create_bot(
        self,
        platform: str,
        meeting_id: str,
        *,
        language: str = "pt",
        passcode: str = "",
        bot_name: str = "",
        bot_image: str = "",
        recording_enabled: bool = True,
        transcribe_enabled: bool = True,
    ) -> dict:
        """Envia bot para entrar em uma reuniao.

        Returns:
            Dados do bot criado (id, status, etc).
        """
        payload = {
            "platform": platform,
            "native_meeting_id": meeting_id,
            "language": language,
            "recording_enabled": recording_enabled,
            "transcribe_enabled": transcribe_enabled,
        }
        if passcode:
            payload["passcode"] = passcode
        if bot_name:
            payload["bot_name"] = bot_name
        if bot_image:
            payload["bot_image"] = bot_image

        log.info("Criando bot Vexa: %s/%s", platform, meeting_id)
        resp = requests.post(
            self._url("/bots"),
            json=payload,
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        log.info("Bot criado: %s (status=%s)", data.get("id"), data.get("status"))
        return data

    def stop_bot(self, platform: str, meeting_id: str) -> None:
        """Para o bot e encerra a gravacao."""
        log.info("Parando bot Vexa: %s/%s", platform, meeting_id)
        resp = requests.delete(
            self._url(f"/bots/{platform}/{meeting_id}"),
            headers=self._headers,
            timeout=30,
        )
        resp.raise_for_status()
        log.info("Bot parado com sucesso")

    def get_bot_status(self) -> list[dict]:
        """Lista todos os bots ativos."""
        resp = requests.get(
            self._url("/bots/status"),
            headers=self._headers,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    # --- Transcripts ---

    def get_transcript(self, platform: str, meeting_id: str) -> VexaTranscript:
        """Busca a transcricao completa de uma reuniao."""
        log.info("Buscando transcricao Vexa: %s/%s", platform, meeting_id)
        resp = requests.get(
            self._url(f"/transcripts/{platform}/{meeting_id}"),
            headers=self._headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        segments = []
        for seg in data.get("segments", []):
            segments.append(VexaSegment(
                start_time=seg.get("start_time", 0.0),
                end_time=seg.get("end_time", 0.0),
                text=seg.get("text", ""),
                speaker=seg.get("speaker", "Desconhecido"),
                language=seg.get("language", ""),
            ))

        start_time = None
        end_time = None
        if data.get("start_time"):
            try:
                start_time = datetime.fromisoformat(
                    data["start_time"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass
        if data.get("end_time"):
            try:
                end_time = datetime.fromisoformat(
                    data["end_time"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        log.info(
            "Transcricao recebida: %d segmentos, %d participantes",
            len(segments),
            len({s.speaker for s in segments}),
        )
        return VexaTranscript(
            meeting_id=meeting_id,
            platform=platform,
            start_time=start_time,
            end_time=end_time,
            segments=segments,
            recordings=data.get("recordings", []),
        )

    # --- Recordings ---

    def list_recordings(self) -> list[dict]:
        """Lista todas as gravacoes disponivel."""
        resp = requests.get(
            self._url("/recordings"),
            headers=self._headers,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def download_recording(
        self,
        recording_id: str,
        media_file_id: str,
        output_path: Path,
    ) -> Path:
        """Baixa um arquivo de audio do Vexa.

        Returns:
            Caminho do arquivo baixado.
        """
        log.info("Baixando gravacao: %s/%s", recording_id, media_file_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        resp = requests.get(
            self._url(f"/recordings/{recording_id}/media/{media_file_id}/raw"),
            headers=self._headers,
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        log.info("Gravacao salva: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
        return output_path

    # --- Serialization helpers ---

    @staticmethod
    def segments_to_json(segments: list[VexaSegment]) -> str:
        """Serializa segmentos para JSON (para persistir na task queue)."""
        return json.dumps([
            {
                "start_time": s.start_time,
                "end_time": s.end_time,
                "text": s.text,
                "speaker": s.speaker,
                "language": s.language,
            }
            for s in segments
        ], ensure_ascii=False)

    @staticmethod
    def segments_from_json(data: str) -> list[VexaSegment]:
        """Desserializa segmentos de JSON."""
        items = json.loads(data)
        return [
            VexaSegment(
                start_time=s["start_time"],
                end_time=s["end_time"],
                text=s["text"],
                speaker=s["speaker"],
                language=s.get("language", ""),
            )
            for s in items
        ]

    def is_available(self) -> bool:
        """Verifica se a API key esta configurada e valida."""
        if not self._api_key:
            return False
        try:
            resp = requests.get(
                self._url("/bots/status"),
                headers=self._headers,
                timeout=10,
            )
            return resp.status_code in (200, 401, 403)
        except requests.RequestException:
            return False
