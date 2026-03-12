"""Utilitário para gravar os sons de join/leave do Google Meet.

Uso:
  1. Abra um Google Meet no navegador
  2. Execute este script
  3. Escolha "join" ou "leave"
  4. O script grava 5 segundos do áudio do sistema
  5. Entre ou saia do Meet durante a gravação
  6. O áudio é salvo em assets/meet_join_sound.wav ou assets/meet_leave_sound.wav
"""

import sys
import time
import wave
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DURATION_SECONDS = 5
SAMPLE_RATE = 16000


def find_loopback_device(pa: pyaudio.PyAudio) -> dict:
    """Encontra o dispositivo WASAPI loopback padrão."""
    wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_output = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

    if default_output.get("isLoopbackDevice"):
        return default_output

    for loopback in pa.get_loopback_device_info_generator():
        if default_output["name"] in loopback["name"]:
            return loopback

    raise RuntimeError("Nenhum dispositivo WASAPI loopback encontrado.")


def record_system_audio(output_path: Path, duration: float = DURATION_SECONDS):
    """Grava o áudio do sistema (loopback) por `duration` segundos."""
    pa = pyaudio.PyAudio()
    try:
        device = find_loopback_device(pa)
        device_rate = int(device["defaultSampleRate"])
        channels = device["maxInputChannels"]

        print(f"Dispositivo: {device['name']}")
        print(f"Sample rate: {device_rate} Hz, Canais: {channels}")
        print(f"Gravando por {duration} segundos...")
        print(">>> FAÇA A AÇÃO NO MEET AGORA! <<<\n")

        frames = []

        def callback(in_data, frame_count, time_info, status):
            frames.append(in_data)
            return (in_data, pyaudio.paContinue)

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=device_rate,
            frames_per_buffer=1024,
            input=True,
            input_device_index=device["index"],
            stream_callback=callback,
        )

        stream.start_stream()
        for i in range(duration):
            time.sleep(1)
            print(f"  {duration - i - 1} segundos restantes...")
        stream.stop_stream()
        stream.close()

        # Converter para mono 16kHz
        raw = b"".join(frames)
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

        # Stereo → mono
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)

        # Resample para 16kHz
        if device_rate != SAMPLE_RATE:
            from scipy import signal
            num_samples = int(len(data) * SAMPLE_RATE / device_rate)
            data = signal.resample(data, num_samples)

        # Normalizar
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val

        # Salvar como WAV 16-bit
        data_int16 = (data * 32767).astype(np.int16)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(data_int16.tobytes())

        print(f"\nSalvo em: {output_path}")
        print(f"Duração: {len(data_int16) / SAMPLE_RATE:.1f}s")

    finally:
        pa.terminate()


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Gravador de Sons do Google Meet")
    print("=" * 50)
    print()
    print("Escolha o que gravar:")
    print("  1. Som de JOIN (alguém entrou na reunião)")
    print("  2. Som de LEAVE (alguém saiu da reunião)")
    print("  3. Ambos (join primeiro, depois leave)")
    print()

    choice = input("Opção (1/2/3): ").strip()

    if choice in ("1", "3"):
        print("\n--- Gravando som de JOIN ---")
        print("Prepare-se: entre em um Meet nos próximos 5 segundos.")
        input("Pressione ENTER quando estiver pronto...")
        record_system_audio(ASSETS_DIR / "meet_join_sound.wav")

    if choice in ("2", "3"):
        if choice == "3":
            print("\n--- Agora vamos gravar o som de LEAVE ---")
            input("Pressione ENTER quando estiver pronto para sair do Meet...")
        else:
            print("\n--- Gravando som de LEAVE ---")
            print("Prepare-se: saia de um Meet nos próximos 5 segundos.")
            input("Pressione ENTER quando estiver pronto...")
        record_system_audio(ASSETS_DIR / "meet_leave_sound.wav")

    if choice not in ("1", "2", "3"):
        print("Opção inválida.")
        sys.exit(1)

    print("\nPronto! Os sons foram salvos em assets/")


if __name__ == "__main__":
    main()
