import io
import json
import math
import os
import zipfile
from pathlib import PurePosixPath
import wave

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from scipy.signal import butter, lfilter, resample


MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
METADATA_NAMES = {"mxai.ini", "character.txt", "character.yaml", "readme.txt"}

st.set_page_config(
    page_title="MVX VoiceBank Creator",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #121212; }
    h1, h2, h3, p, label { color: #ffffff; }
    .stButton > button, .stDownloadButton > button {
        background-color: #39FF14;
        color: #000000;
        font-weight: bold;
        border: none;
        min-height: 42px;
        width: 100%;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #2EE610;
        color: #000000;
    }
    .stTextArea textarea {
        background-color: #1A1A1A;
        color: #39FF14;
        font-family: Consolas, monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _safe_zip_path(filename):
    normalized = filename.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
        or (path.parts and path.parts[0].endswith(":"))
    ):
        raise ValueError(f"Caminho inválido dentro do ZIP: {filename}")
    return path


def _validate_archive(zip_bytes):
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError(f"O arquivo enviado não é um ZIP válido: {exc}") from exc

    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if not infos:
            raise ValueError("O arquivo ZIP está vazio.")

        total_size = sum(info.file_size for info in infos)
        if total_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise ValueError("O tamanho descompactado do banco excede o limite de 1 GiB.")

        paths = {}
        for info in infos:
            path = _safe_zip_path(info.filename)
            key = str(path).casefold()
            if key in paths:
                raise ValueError(f"O ZIP contém caminho duplicado: {info.filename}")
            paths[key] = info.filename

        corrupt_member = archive.testzip()
        if corrupt_member:
            raise ValueError(f"Arquivo corrompido dentro do ZIP: {corrupt_member}")

        oto_files = [
            info.filename
            for info in infos
            if PurePosixPath(info.filename.replace("\\", "/")).name.casefold() == "oto.ini"
        ]
        if len(oto_files) != 1:
            raise ValueError(f"Esperado exatamente um arquivo oto.ini; encontrados: {len(oto_files)}.")

        audio_files = [
            info.filename
            for info in infos
            if PurePosixPath(info.filename.replace("\\", "/")).suffix.casefold() == ".wav"
        ]
        if not audio_files:
            raise ValueError("O pacote não contém nenhuma amostra WAV.")

        oto_bytes = archive.read(oto_files[0])
        return oto_files[0], oto_bytes, audio_files


def interpretar_oto_ini(conteudo):
    dados = {}
    for line_number, line in enumerate(conteudo.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith((";", "#")) or "=" not in line:
            continue
        wav_name, raw_values = line.split("=", 1)
        values = [value.strip() for value in raw_values.split(",")]
        if not wav_name.strip() or len(values) < 6:
            continue
        try:
            numbers = [float(value) for value in values[1:6]]
        except ValueError as exc:
            raise ValueError(f"Valor numérico inválido no oto.ini, linha {line_number}.") from exc

        key = wav_name.strip().replace("\\", "/").casefold()
        dados[key] = {
            "alias": values[0],
            "offset": numbers[0],
            "consonant": numbers[1],
            "cutoff": numbers[2],
            "preutterance": numbers[3],
            "overlap": numbers[4],
        }

    if not dados:
        raise ValueError("oto.ini não contém nenhuma entrada válida.")
    return dados


def _oto_for_audio(dados_oto, wav_path):
    normalized = wav_path.replace("\\", "/").casefold()
    if normalized in dados_oto:
        return dados_oto[normalized]
    basename = PurePosixPath(normalized).name
    matches = [value for key, value in dados_oto.items() if PurePosixPath(key).name == basename]
    if len(matches) == 1:
        return matches[0]
    return {}


def _decode_pcm(audio_bytes, sample_width, channels):
    if sample_width == 1:
        audio = (np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif sample_width == 2:
        audio = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768
    elif sample_width == 3:
        if len(audio_bytes) % 3:
            raise ValueError("Dados PCM de 24 bits incompletos.")
        raw = np.frombuffer(audio_bytes, dtype=np.uint8).reshape(-1, 3)
        values = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        audio = values.astype(np.float32) / 8388608
    elif sample_width == 4:
        audio = np.frombuffer(audio_bytes, dtype="<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError(f"PCM de {sample_width} bytes por amostra não é suportado.")

    if channels < 1:
        raise ValueError("O WAV informa uma quantidade inválida de canais.")
    if channels > 1:
        if audio.size % channels:
            raise ValueError("Dados PCM incompletos para o número de canais.")
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


def processar_audio(audio_bytes_in, dados_oto, nome_wav, p_pitch, p_air, p_formant, p_gender, p_growl, p_naturalness):
    try:
        with wave.open(io.BytesIO(audio_bytes_in), "rb") as wav:
            if wav.getcomptype() != "NONE":
                raise ValueError("O WAV precisa usar PCM sem compressão.")
            sample_rate = wav.getframerate()
            if sample_rate <= 0:
                raise ValueError("O WAV informa uma frequência de amostragem inválida.")
            audio = _decode_pcm(wav.readframes(wav.getnframes()), wav.getsampwidth(), wav.getnchannels())
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"Não foi possível ler o WAV: {exc}") from exc

    if not audio.size:
        raise ValueError("A amostra WAV está vazia.")

    oto = _oto_for_audio(dados_oto, nome_wav)
    try:
        consonant_ms = max(0.0, float(oto.get("consonant", 0.0)))
    except (TypeError, ValueError):
        consonant_ms = 0.0
    split = min(int(consonant_ms * sample_rate / 1000), len(audio))
    consonant = audio[:split]
    vowel = audio[split:].copy()

    pitch_factor = 2 ** (p_pitch / 12.0)
    if len(vowel) > 1 and not math.isclose(pitch_factor, 1.0, rel_tol=1e-6):
        vowel = resample(vowel, max(1, round(len(vowel) / pitch_factor))).astype(np.float32)

    formant_factor = p_formant * p_gender
    if len(vowel) > 1 and not math.isclose(formant_factor, 1.0, rel_tol=1e-6):
        spectrum = np.fft.rfft(vowel)
        old_bins = np.arange(len(spectrum))
        new_bins = np.clip(old_bins * formant_factor, 0, len(spectrum) - 1)
        real = np.interp(new_bins, old_bins, spectrum.real)
        imag = np.interp(new_bins, old_bins, spectrum.imag)
        vowel = np.fft.irfft(real + 1j * imag, n=len(vowel)).astype(np.float32)

    if vowel.size and p_growl > 0:
        peak = max(float(np.max(np.abs(vowel))), 1e-8)
        times = np.arange(len(vowel)) / sample_rate
        modulation = np.sin(2 * math.pi * 60 * times) * p_growl * 0.12
        saturated = np.arctan((vowel / peak) * (1 + p_growl * 2.5)) / (math.pi / 2)
        vowel = (vowel * (1 - p_growl * 0.25) + saturated * p_growl * 0.25) * (1 + modulation)

    if vowel.size and p_air > 0:
        cutoff = min(3800.0, sample_rate * 0.45)
        if cutoff > 0:
            b, a = butter(4, cutoff / (sample_rate / 2), btype="high")
            breath = lfilter(b, a, np.random.normal(0, 0.02, len(vowel)))
            envelope = np.abs(vowel)
            breath *= envelope / max(float(np.max(envelope)), 1e-8)
            vowel = vowel * (1 - p_air * 0.2) + breath * p_air

    if vowel.size and p_naturalness > 0:
        times = np.arange(len(vowel)) / sample_rate
        vibrato = 1 + np.sin(2 * math.pi * 5.6 * times) * p_naturalness * 0.00015
        vowel *= vibrato

    rendered = np.concatenate((consonant, vowel))
    peak = float(np.max(np.abs(rendered))) if rendered.size else 0.0
    if peak > 0.98:
        rendered *= 0.98 / peak
    pcm = np.clip(rendered * 32767, -32768, 32767).astype("<i2")

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


def _validated_image(uploaded_file, label, max_size=None):
    if uploaded_file is None:
        return None, None
    content = uploaded_file.getvalue()
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError(f"{label}: a imagem excede o limite de 20 MiB.")
    try:
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
            image_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"{label}: arquivo de imagem inválido.") from exc
    if image_format not in {"PNG", "JPEG"}:
        raise ValueError(f"{label}: use uma imagem PNG ou JPEG.")
    if max_size and (width > max_size or height > max_size):
        raise ValueError(f"{label}: dimensão máxima {max_size}x{max_size}; imagem {width}x{height}.")
    extension = ".png" if image_format == "PNG" else ".jpg"
    return content, extension


def _metadata_line(value):
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def criar_pacote(zip_bytes, nome, autor, icon_bytes, portrait_bytes, readme, oto_data, audio_files, settings):
    name = _metadata_line(nome)
    author = _metadata_line(autor)
    if not name or not author:
        raise ValueError("Informe o nome do personagem e o autor.")

    mxai = [
        "[MxAi_Configuration]",
        "EngineCompatible=MaximizeVoiceSynthesizerMobile",
        "GlobalType=MaxiVloid",
        f"GlobalPitchShift={settings['pitch']:.2f}",
        f"GlobalAirRatio={settings['air']:.2f}",
        f"GlobalFormantShift={settings['formant']:.2f}",
        f"GlobalGenderFactor={settings['gender']:.2f}",
        f"GlobalGrowlAmount={settings['growl']:.2f}",
        f"GlobalNaturalness={settings['naturalness']:.2f}",
        "",
        "[Amostras_Calibradas]",
    ]
    for wav_path in audio_files:
        basename = PurePosixPath(wav_path.replace("\\", "/")).name
        oto = _oto_for_audio(oto_data, wav_path)
        mxai.append(
            f"{basename}=Pitch={settings['pitch']:.2f},Air={settings['air']:.2f},"
            f"Formant={settings['formant']:.2f},Gender={settings['gender']:.2f},"
            f"Growl={settings['growl']:.2f},Naturalness={settings['naturalness']:.2f},"
            f"ProtectConsonant={oto.get('consonant', 0)}"
        )

    character_txt = [
        "type=MaxiVloid",
        f"name={name}",
        "image=portrait.png",
        f"author={author}",
        "icon=icon.png",
        "version=Maximize_Voice_2.0",
    ]
    character_yaml = [
        "type: MaxiVloid",
        f"name: {json.dumps(name, ensure_ascii=False)}",
        f"author: {json.dumps(author, ensure_ascii=False)}",
        "image: portrait.png",
        "icon: icon.png",
        "voice_engine: MaximizeVoiceSynthesizerMobile",
    ]
    replacement_names = METADATA_NAMES | {"icon.png", "portrait.png"}

    result = io.BytesIO()
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as source:
            with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as output:
                for item in source.infolist():
                    if item.is_dir():
                        continue
                    basename = PurePosixPath(item.filename.replace("\\", "/")).name.casefold()
                    if basename not in replacement_names:
                        output.writestr(item, source.read(item.filename))
                output.writestr("MxAi.ini", "\n".join(mxai))
                output.writestr("character.txt", "\n".join(character_txt))
                output.writestr("character.yaml", "\n".join(character_yaml))
                if readme.strip():
                    output.writestr("README.txt", readme.strip())
                output.writestr("icon.png", icon_bytes)
                output.writestr("portrait.png", portrait_bytes)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError(f"Não foi possível montar o pacote ZIP: {exc}") from exc
    return result.getvalue()


def _initialize_state():
    defaults = {
        "stage": 1,
        "zip_bytes": None,
        "zip_name": "",
        "oto_path": "",
        "oto_bytes": b"",
        "oto_data": {},
        "audio_files": [],
        "preview_bytes": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_import():
    for key in (
        "zip_bytes", "zip_name", "oto_path", "oto_bytes", "oto_data",
        "audio_files", "preview_bytes",
    ):
        st.session_state[key] = None if key in {"zip_bytes", "preview_bytes"} else (
            {} if key == "oto_data" else [] if key == "audio_files" else b"" if key == "oto_bytes" else ""
        )
    st.session_state.stage = 1


def _import_panel():
    st.subheader("1. Importar banco de voz")
    uploaded = st.file_uploader(
        "Selecione o ZIP com oto.ini e amostras WAV",
        type=["zip"],
        key="voicebank_zip",
    )
    if uploaded is None:
        return
    if st.button("Validar pacote e continuar", key="validate_zip"):
        try:
            zip_bytes = uploaded.getvalue()
            oto_path, oto_bytes, audio_files = _validate_archive(zip_bytes)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state.zip_bytes = zip_bytes
        st.session_state.zip_name = uploaded.name
        st.session_state.oto_path = oto_path
        st.session_state.oto_bytes = oto_bytes
        st.session_state.audio_files = audio_files
        st.session_state.stage = 2
        st.rerun()


def _encoding_panel():
    st.subheader("2. Validar codificação")
    encoding = st.selectbox("Codificação do oto.ini", ["UTF-8", "Shift-JIS"], key="oto_encoding")
    codec = "shift_jis" if encoding == "Shift-JIS" else "utf-8-sig"
    try:
        text = st.session_state.oto_bytes.decode(codec)
    except UnicodeDecodeError as exc:
        st.error(f"O oto.ini não pode ser decodificado como {encoding}: {exc}")
        return
    st.text_area("Prévia de oto.ini", "\n".join(text.splitlines()[:12]), height=220, disabled=True)
    left, right = st.columns(2)
    if left.button("Voltar à importação"):
        _reset_import()
        st.rerun()
    if right.button("Confirmar codificação e avançar", type="primary"):
        try:
            st.session_state.oto_data = interpretar_oto_ini(text)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state.stage = 3
        st.rerun()


def _settings_panel():
    with st.expander("Ajustes de processamento", expanded=True):
        return {
            "pitch": st.slider("Ajuste de nota musical (semitons)", -6.0, 6.0, 0.0, 0.5),
            "air": st.slider("Ar na voz", 0.0, 1.0, 0.0, 0.01),
            "formant": st.slider("Timbre do corpo vocal", 0.85, 1.15, 1.0, 0.01),
            "gender": st.slider("Fator de trato vocal", 0.8, 1.2, 1.0, 0.01),
            "growl": st.slider("Growl", 0.0, 1.0, 0.0, 0.01),
            "naturalness": st.slider("Vibrato", 0.0, 1.0, 1.0, 0.01),
        }


def _creator_panel(settings):
    st.subheader("3. Identidade e exportação")
    with st.form("voicebank_identity"):
        name = st.text_input("Nome do personagem", value="MaxiVloid_Character", max_chars=120)
        author = st.text_input("Autor / desenvolvedor", value="MVX_Developer", max_chars=120)
        icon_upload = st.file_uploader(
            "Ícone obrigatório (PNG/JPEG, até 1280x1280)",
            type=["png", "jpg", "jpeg"],
            key="icon_upload",
        )
        portrait_upload = st.file_uploader(
            "Ilustração obrigatória (PNG/JPEG)",
            type=["png", "jpg", "jpeg"],
            key="portrait_upload",
        )
        readme = st.text_area("README.txt (opcional)", height=120, max_chars=20000)
        preview_sample = st.selectbox(
            "Amostra para prévia",
            st.session_state.audio_files,
            key="preview_sample",
        )
        make_preview = st.form_submit_button("Gerar prévia de áudio")
        compile_package = st.form_submit_button("Compilar pacote MaxiVloid")

    if make_preview or compile_package:
        if not name.strip() or not author.strip():
            st.error("Informe o nome do personagem e o autor.")
            return

        if make_preview:
            try:
                with zipfile.ZipFile(io.BytesIO(st.session_state.zip_bytes), "r") as archive:
                    raw_audio = archive.read(preview_sample)
                st.session_state.preview_bytes = processar_audio(
                    raw_audio,
                    st.session_state.oto_data,
                    preview_sample,
                    settings["pitch"],
                    settings["air"],
                    settings["formant"],
                    settings["gender"],
                    settings["growl"],
                    settings["naturalness"],
                )
            except (KeyError, OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
                st.error(f"Não foi possível processar o áudio: {exc}")
                return
        elif compile_package:
            try:
                icon_bytes, _ = _validated_image(icon_upload, "Ícone", max_size=1280)
                portrait_bytes, _ = _validated_image(portrait_upload, "Ilustração")
                if icon_bytes is None or portrait_bytes is None:
                    raise ValueError("Envie o ícone e a ilustração obrigatórios.")
                result = criar_pacote(
                    st.session_state.zip_bytes,
                    name,
                    author,
                    icon_bytes,
                    portrait_bytes,
                    readme,
                    st.session_state.oto_data,
                    st.session_state.audio_files,
                    settings,
                )
            except ValueError as exc:
                st.error(str(exc))
                return
            st.download_button(
                "Baixar banco de voz compilado",
                data=result,
                file_name="banco_maxivloid_final.zip",
                mime="application/zip",
            )

    if st.session_state.preview_bytes:
        st.audio(st.session_state.preview_bytes, format="audio/wav")


def main():
    _initialize_state()
    st.title("MVX VoiceBank Creator")
    st.caption("Importe um banco de voz, confira oto.ini e exporte o pacote MaxiVloid.")

    with st.sidebar:
        st.header("Fluxo")
        st.write(f"Etapa {st.session_state.stage} de 3")
        if st.session_state.zip_name:
            st.caption(f"Pacote: {st.session_state.zip_name}")
        if st.button("Reiniciar"):
            _reset_import()
            st.rerun()

    import_col, process_col = st.columns([1, 1])
    with import_col:
        if st.session_state.stage == 1:
            _import_panel()
        elif st.session_state.stage == 2:
            _encoding_panel()
        else:
            st.success(
                f"Pacote validado: {len(st.session_state.audio_files)} WAV(s), "
                f"{len(st.session_state.oto_data)} entrada(s) oto.ini."
            )
    with process_col:
        if st.session_state.stage == 3:
            settings = _settings_panel()
            _creator_panel(settings)
        else:
            st.info("Conclua a importação e a validação do oto.ini para habilitar processamento.")


if __name__ == "__main__":
    main()
