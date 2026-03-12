# Gravação e Transcrição de Reuniões

App desktop Windows para gravação automática e transcrição de reuniões do Google Meet, Teams, Zoom e qualquer áudio do sistema. Substituto do Tactiq.

## Funcionalidades

- **System Tray** com ícone colorido (cinza/vermelho/amarelo) e menu completo
- **Gravação dual-track**: microfone (sua voz) + loopback (áudio dos outros) em faixas separadas
- **Detecção automática do Google Meet**: monitora processo do Chrome Web App e detecta sons de join/leave
- **Transcrição local** com Qwen3-ASR-0.6B (melhor qualidade PT-BR) ou faster-whisper large-v3 (mais rápido)
- **Fallback gratuito** via Groq API (8h/dia grátis)
- **Identificação de falantes** em 5 camadas:
  1. Dual-track (eu vs. outros — 100% certeza)
  2. pyannote-audio (diferenciar os outros entre si)
  3. Voice enrollment (mapear vozes a nomes conhecidos)
  4. Inferência por contexto (detectar nomes na conversa)
  5. Detecção visual do Meet (screenshot + OCR)
- **Saída em Markdown** com timestamps, speakers e metadados
- **Log geral** de todas as gravações com início/fim/arquivos

## Requisitos

- Windows 10/11
- Python 3.11+
- GPU NVIDIA com CUDA (opcional, melhora velocidade)
- Tesseract OCR (para detecção visual do Meet)

### Hardware

| Cenário | RAM | VRAM GPU | Disco |
|---------|-----|----------|-------|
| GPU (recomendado) | 16 GB | 4 GB+ | 5 GB |
| CPU-only | 16 GB | — | 5 GB |

## Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/caiobuni/gravacao-transcricao-de-meet-teams-zoom-e-afins.git
cd gravacao-transcricao-de-meet-teams-zoom-e-afins

# 2. Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate

# 3. Instalar PyTorch (com CUDA se tiver GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
# Ou sem GPU:
pip install torch torchaudio

# 4. Instalar o projeto
pip install -e .

# 5. Configurar variáveis de ambiente
copy .env.example .env
# Editar .env com seu HF_TOKEN e GROQ_API_KEY

# 6. Baixar modelos
python setup_models.py
```

## Uso

```bash
# Iniciar o app (aparece no System Tray)
python -m src.main
```

### Menu do System Tray

| Item | Ação |
|------|------|
| Iniciar Gravação | Começa gravação manual |
| Parar Gravação | Para e inicia transcrição |
| Última transcrição | Abre o .md mais recente |
| Abrir pasta de transcrições | Abre `H:\Meu Drive\Tactiq Transcription\` |
| Abrir log | Abre o log de gravações |
| Configurações | Abre janela de configurações |
| Sair | Encerra o app |

### Detecção automática do Meet

O app monitora se o Chrome Web App do Google Meet está rodando. Quando detectado, ativa o "modo escuta" e grava automaticamente ao detectar o som de início de reunião. Para ao detectar o som de saída.

Para preparar os samples de som, grave os tons de join/leave do Meet e salve em:
- `assets/meet_join_sound.wav`
- `assets/meet_leave_sound.wav`

## Saída

Transcrições salvas em `H:\Meu Drive\Tactiq Transcription\YYYYMMDD-HHmm.md`:

```markdown
# Transcrição de Reunião

- **Data:** 2026-03-12
- **Início:** 14:30
- **Duração:** 01:23:45
- **Participantes:** Caio (eu), João, Maria

---

## Transcrição

**[00:00:05] Caio (eu):**
Bom dia pessoal, vamos começar.

**[00:00:12] João:**
Bom dia! Tenho atualizações do projeto.
```

## Configuração

O app salva configurações em `data/config.json`:

| Configuração | Padrão | Descrição |
|-------------|--------|-----------|
| `user_name` | "Eu" | Seu nome nas transcrições |
| `transcription_engine` | "qwen" | Motor: qwen, whisper, groq |
| `language` | "pt" | Idioma das reuniões |
| `auto_detect_meet` | true | Detectar Meet automaticamente |
| `auto_delete_audio` | true | Deletar WAV após transcrição |
| `transcription_output_dir` | `H:\Meu Drive\Tactiq Transcription` | Pasta de saída |

## Estrutura do Projeto

```
├── pyproject.toml              # Dependências e config
├── setup_models.py             # Download de modelos ML
├── .env.example                # Template de variáveis
│
├── src/
│   ├── main.py                 # Entry point
│   ├── config/
│   │   ├── constants.py        # Caminhos e constantes
│   │   └── settings.py         # Config persistente (JSON)
│   ├── core/
│   │   ├── audio_capture.py    # Gravação dual-track (loopback + mic)
│   │   ├── audio_preprocessing.py # Resample 16kHz mono
│   │   ├── transcriber_base.py # Interface abstrata
│   │   ├── transcriber_qwen.py # Qwen3-ASR-0.6B (primário)
│   │   ├── transcriber_whisper.py # faster-whisper (alternativo)
│   │   ├── transcriber_groq.py # Groq API (fallback gratuito)
│   │   ├── diarizer.py         # pyannote speaker diarization
│   │   ├── aligner.py          # Merge dual-track + diarização
│   │   ├── speaker_identifier.py # Voice enrollment + contexto
│   │   ├── meet_visual.py      # Screenshot + OCR do Meet
│   │   ├── meet_detector.py    # Detectar sons join/leave
│   │   ├── process_monitor.py  # Detectar Chrome Web App
│   │   ├── formatter.py        # Gerar SRT + Markdown
│   │   └── pipeline.py         # Orquestração completa
│   ├── tray/
│   │   └── tray_app.py         # System Tray (pystray)
│   ├── gui/
│   │   └── settings_window.py  # Configurações (CustomTkinter)
│   └── utils/
│       ├── logger.py           # Log de gravações
│       ├── gpu_check.py        # Detecção CUDA
│       └── audio_utils.py      # Helpers de áudio
│
├── assets/                     # Sons de referência do Meet
├── models/                     # Modelos ML (gitignored)
├── data/                       # Config + voice profiles (gitignored)
├── recordings/                 # Gravações temporárias (gitignored)
└── log/                        # Logs (gitignored)
```

## Motores de Transcrição

| Motor | WER PT-BR | Velocidade CPU | Velocidade GPU | Custo |
|-------|-----------|---------------|---------------|-------|
| Qwen3-ASR-0.6B | ~3.9% | 2-6x realtime | 50-100x realtime | Gratuito (local) |
| faster-whisper large-v3 | ~7-8% | 8-15x realtime | 40-60x realtime | Gratuito (local) |
| Groq API (large-v3) | ~7-8% | Instantâneo | — | Gratuito (8h/dia) |

## Licença

MIT
