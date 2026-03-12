import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config.constants import VOICE_PROFILES_DIR

log = logging.getLogger(__name__)

@dataclass
class VoiceProfile:
    name: str
    embedding: list[float]  # speaker embedding vector

class SpeakerIdentifier:
    def __init__(self, profiles_dir: Path | None = None):
        self._profiles_dir = profiles_dir or VOICE_PROFILES_DIR
        self._profiles_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, VoiceProfile] = {}
        self._load_profiles()

    def _load_profiles(self):
        """Load saved voice profiles from disk."""
        profiles_file = self._profiles_dir / "profiles.json"
        if profiles_file.exists():
            data = json.loads(profiles_file.read_text(encoding="utf-8"))
            for name, embedding in data.items():
                self._profiles[name] = VoiceProfile(name=name, embedding=embedding)

    def save_profiles(self):
        """Save voice profiles to disk."""
        profiles_file = self._profiles_dir / "profiles.json"
        data = {p.name: p.embedding for p in self._profiles.values()}
        profiles_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def register_voice(self, name: str, audio_path: Path):
        """Extract embedding from audio and register as a known voice."""
        # Use pyannote's embedding model to extract speaker embedding
        # Save to profiles
        try:
            from pyannote.audio import Inference
            inference = Inference("pyannote/embedding", use_auth_token=os.environ.get("HF_TOKEN"))
            embedding = inference(str(audio_path))
            # Average over time to get a single vector
            avg_embedding = np.mean(embedding.data, axis=0).tolist()
            self._profiles[name] = VoiceProfile(name=name, embedding=avg_embedding)
            self.save_profiles()
            log.info(f"Voz registrada: {name}")
        except Exception as e:
            log.error(f"Erro ao registrar voz: {e}")

    def identify_by_voice(self, speaker_embedding: list[float], threshold: float = 0.7) -> str | None:
        """Try to match an embedding against known profiles using cosine similarity."""
        if not self._profiles:
            return None

        query = np.array(speaker_embedding)
        best_name = None
        best_score = -1.0

        for profile in self._profiles.values():
            ref = np.array(profile.embedding)
            # Cosine similarity
            sim = np.dot(query, ref) / (np.linalg.norm(query) * np.linalg.norm(ref) + 1e-8)
            if sim > best_score:
                best_score = sim
                best_name = profile.name

        if best_score >= threshold:
            return best_name
        return None

    def identify_by_context(self, segments: list) -> dict[str, str]:
        """Scan transcript text for name mentions and try to associate with speakers.

        Returns mapping: {"SPEAKER_00": "Joao", "SPEAKER_01": "Maria", ...}
        """
        # Patterns that suggest someone is addressing a speaker:
        # "Fulano, o que voce acha?"
        # "Obrigado, Maria"
        # "Bom dia Joao"
        # "Eu sou o Pedro"

        name_pattern = re.compile(
            r'(?:'
            r'(?:obrigad[oa],?\s+)'      # "Obrigado, Nome"
            r'|(?:bom dia,?\s+)'          # "Bom dia Nome"
            r'|(?:boa tarde,?\s+)'        # "Boa tarde Nome"
            r'|(?:boa noite,?\s+)'        # "Boa noite Nome"
            r'|(?:valeu,?\s+)'            # "Valeu Nome"
            r'|(?:ne,?\s+)'              # "ne Nome"
            r'|(?:eu sou (?:o |a )?)'     # "Eu sou o Nome"
            r'|(?:meu nome e\s+)'         # "Meu nome e Nome"
            r')'
            r'([A-Z\u00c0-\u00dc][a-z\u00e0-\u00fc]+)',  # Capture name (capitalized)
            re.IGNORECASE
        )

        # Also detect "Nome, ..." at start of sentence (vocative)
        vocative_pattern = re.compile(
            r'^([A-Z\u00c0-\u00dc][a-z\u00e0-\u00fc]+),\s',
        )

        # Collect name mentions with their associated speaker and timing
        speaker_name_votes = {}  # {speaker_label: {name: count}}

        for seg in segments:
            speaker = getattr(seg, 'speaker', None)
            text = getattr(seg, 'text', '')
            if not speaker or not text:
                continue

            # Check for name mentions addressing someone ELSE
            # If speaker A says "Obrigado, Joao", then the PREVIOUS speaker is likely Joao
            for match in name_pattern.finditer(text):
                name = match.group(1)
                # This name is likely NOT the current speaker, but someone they're talking to
                # We'll collect as a general vote
                if name not in speaker_name_votes:
                    speaker_name_votes[name] = {}

            # Check vocative at start
            voc_match = vocative_pattern.match(text)
            if voc_match:
                name = voc_match.group(1)
                if name not in speaker_name_votes:
                    speaker_name_votes[name] = {}

            # Check "Eu sou o/a Nome" - this IS the current speaker
            self_intro = re.search(r'eu sou (?:o |a )?([A-Z\u00c0-\u00dc][a-z\u00e0-\u00fc]+)', text, re.IGNORECASE)
            if self_intro:
                name = self_intro.group(1)
                if speaker not in speaker_name_votes:
                    speaker_name_votes[speaker] = {}
                speaker_name_votes[speaker] = {name: speaker_name_votes.get(speaker, {}).get(name, 0) + 10}

        # Build final mapping from most confident associations
        mapping = {}
        used_names = set()

        for speaker, names in sorted(speaker_name_votes.items(), key=lambda x: max(x[1].values()) if x[1] else 0, reverse=True):
            if not names:
                continue
            best_name = max(names, key=names.get)
            if best_name not in used_names:
                mapping[speaker] = best_name
                used_names.add(best_name)

        return mapping
