# CLAUDE.md — Gravação e Transcrição de Reuniões

## Projeto

App desktop Windows (System Tray) para gravação automática e transcrição de reuniões. Substituto do Tactiq. Grava dual-track (mic + loopback), transcreve com Qwen3-ASR ou faster-whisper, identifica falantes, e salva em Markdown.

## Comandos

```bash
# Rodar o app
python -m src.main

# Baixar modelos ML
python setup_models.py

# Instalar dependências
pip install -e .

# Testes
pytest tests/
```

## Arquitetura

- **Entry point**: `src/main.py` → `src/tray/tray_app.py` (System Tray)
- **Pipeline**: `src/core/pipeline.py` orquestra o fluxo: gravação → transcrição → diarização → merge → Markdown
- **Gravação**: `src/core/audio_capture.py` — dual-track via PyAudioWPatch (WASAPI Loopback + Microfone)
- **Transcrição**: interface `TranscriberBase` com 3 implementações (qwen, whisper, groq)
- **Diarização**: `src/core/diarizer.py` — pyannote-audio
- **Config**: `src/config/settings.py` — dataclass persistida em JSON (`data/config.json`)
- **Constantes**: `src/config/constants.py` — caminhos, sample rates, etc.

## Convenções

- Python 3.11+, imports com `from src.config/core/utils/... import ...`
- Logging: `log = logging.getLogger(__name__)` em cada módulo
- Texto da UI em português
- Nomes de arquivo em snake_case
- Constantes em `src/config/constants.py`
- Configurações do usuário via `Settings` dataclass
- Transcrições salvas em `H:\Meu Drive\Tactiq Transcription\YYYYMMDD-HHmm.md`
- Áudios temporários em `recordings/`, deletados após transcrição confirmada

## Saída

Transcrições em Markdown com:
- Cabeçalho: data, início, fim, duração, participantes, motor
- Corpo: `**[HH:MM:SS] Nome:**` seguido do texto
- Footer: motor + diarização usados
