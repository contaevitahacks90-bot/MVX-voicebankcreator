import io
import math
import os
import wave
import zipfile

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from scipy.ndimage import zoom
from scipy.signal import butter, lfilter


st.set_page_config(
    page_title="MVX VoiceBank Creator",
    page_icon="mxveditor.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #121212; }
    h1, h2, h3, p, label, .stSlider { color: #ffffff !important; }
    .stButton > button {
        background-color: #39FF14 !important;
        color: #000000 !important;
        font-weight: bold !important;
        border: none !important;
        width: 100% !important;
        height: 45px !important;
    }
    .stButton > button:hover { background-color: #2EE610 !important; color: #000000 !important; }
    .import-btn > div > button { background-color: #FFF000 !important; color: #000000 !important; }
    .import-btn > div > button:hover { background-color: #CCFF00 !important; }
    .stTextArea > div > div > textarea {
        background-color: #1A1A1A !important;
        color: #39FF14 !important;
        font-family: 'Consolas', monospace !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


SESSION_KEYS = (
    "tela",
    "zip_bytes",
    "linhas_oto_cruas",
    "nome_arquivo_zip",
    "dados_oto",
    "lista_audios",
    "audio_selecionado",
    "encoding",
)

for key in SESSION_KEYS:
    if key not in st.session_state:
        default = 1 if key == "tela" else None
        if key == "dados_oto":
            default = {}
        elif key == "lista_audios":
            default = []
        elif key == "audio_selecionado":
            default = ""
        elif key == "encoding":
            default = "UTF-8"
        st.session_state[key] = default


def interpretar_oto_ini(conteudo: str) -> dict[str, dict[str, str]]:
    dados: dict[str, dict[str, str]] = {}
    for linha in conteudo.splitlines():
        linha = linha.strip()
        if not linha or linha.startswith(";") or "=" not in linha:
            continue

        nome_wav, dados_linha = [parte.strip() for parte in linha.split("=", 1)]
        parametros = [parametro.strip() for parametro in dados_linha.split(",")]
        if not nome_wav:
            continue

        alias = parametros[0] if parametros and parametros[0] else ""
        consonant = parametros[1] if len(parametros) > 1 and parametros[1] else "0"
        dados[nome_wav] = {"alias": alias, "consonant": consonant}

    return dados


def _normalizar_audio(audio_data: np.ndarray, sampwidth: int) -> np.ndarray:
    if sampwidth == 1:
        return (audio_data.astype(np.float32) - 128.0) * 256.0
    if sampwidth == 2:
        return audio_data.astype(np.float32)
    if sampwidth == 3:
        samples = audio_data.reshape(-1, 3)
        return (
            samples.astype(np.int32)
            .dot(np.array([1, 256, 65536], dtype=np.int32))
            .astype(np.float32)
        )
    if sampwidth == 4:
        return audio_data.view(np.int32).astype(np.float32)
    raise ValueError(f"Formato de áudio não suportado: {sampwidth} bytes por amostra")


def processar_audio(
    audio_bytes_in: bytes,
    dados_oto: dict[str, dict[str, str]],
    nome_wav_puro: str,
    p_pitch: float,
    p_air: float,
    p_formant: float,
    p_gender: float,
    p_growl: float,
    p_naturalness: float,
) -> bytes:
    try:
        with wave.open(io.BytesIO(audio_bytes_in), "rb") as w_in:
            params = w_in.getparams()
            frames = w_in.readframes(params.nframes)
    except wave.Error as exc:
        raise ValueError("O arquivo de áudio não é um WAV válido.") from exc

    sample_count = len(frames) // params.sampwidth
    audio_data = np.frombuffer(frames, dtype=np.uint8 if params.sampwidth == 1 else np.dtype(f"<i{params.sampwidth}")).astype(np.float32)
    audio_data = _normalizar_audio(audio_data, params.sampwidth)

    try:
        consonante_ms = float(dados_oto.get(nome_wav_puro, {}).get("consonant", "0"))
    except ValueError:
        consonante_ms = 0.0

    ponto_corte = min(int((consonante_ms / 1000.0) * params.framerate), len(audio_data))
    consonante_chunk = audio_data[:ponto_corte]
    vogal_chunk = audio_data[ponto_corte:]

    if len(vogal_chunk) == 0:
        audio_final = audio_data
    else:
        if p_naturalness > 0.05:
            freq_vibrato = 5.6 + np.random.normal(0.0, 0.4)
            amplitude_vibrato = 0.022 * p_naturalness
            tempo = np.arange(len(vogal_chunk), dtype=np.float32) / params.framerate
            pontos_vibrato = np.sin(2.0 * math.pi * freq_vibrato * tempo) * amplitude_vibrato
            indices_organicos = np.clip(
                np.arange(len(vogal_chunk), dtype=np.float32)
                + pontos_vibrato * params.framerate * 0.01,
                0,
                len(vogal_chunk) - 1,
            ).astype(np.int32)
            vogal_chunk = vogal_chunk[indices_organicos]

        pitch_factor = 2 ** (p_pitch / 12.0)
        if pitch_factor != 1.0:
            vogal_chunk = zoom(vogal_chunk, 1.0 / pitch_factor)

        if p_formant != 1.0 or p_gender != 1.0:
            fator_total = p_formant * p_gender
            janela, passo = 1024, 256
            janela_hann = np.hanning(janela)
            espectros = []
            max_inicio = max(0, len(vogal_chunk) - janela)
            for inicio in range(0, max_inicio + 1, passo):
                segmento = vogal_chunk[inicio : inicio + janela]
                if len(segmento) < janela:
                    segmento = np.pad(segmento, (0, janela - len(segmento)))
                espectro = np.fft.rfft(segmento * janela_hann)
                indices = np.clip(np.arange(len(espectro)) * fator_total, 0, len(espectro) - 1).astype(np.int32)
                espectro_modificado = espectro[indices]
                espectro_modificado = np.pad(
                    espectro_modificado,
                    (0, len(espectro) - len(espectro_modificado)),
                ) if len(espectro_modificado) < len(espectro) else espectro_modificado[: len(espectro)]
                espectros.append(np.fft.irfft(espectro_modificado))

            if espectros:
                recon = np.zeros(len(vogal_chunk), dtype=np.float32)
                for indice, bloco in enumerate(espectros):
                    inicio = indice * passo
                    fim = min(inicio + janela, len(recon))
                    recon[inicio:fim] += bloco[: fim - inicio]
                vogal_chunk = recon

        if p_growl > 0.0:
            tempo_g = np.arange(len(vogal_chunk), dtype=np.float32) / params.framerate
            mod_ventricular = np.sin(2.0 * math.pi * 60.0 * tempo_g) * (p_growl * 0.28)
            pico_maximo = np.max(np.abs(vogal_chunk)) + 1e-5
            sinal_saturado = np.arctan(vogal_chunk / pico_maximo * (1.0 + p_growl * 2.5)) * pico_maximo
            vogal_chunk = (
                vogal_chunk * (1.0 - p_growl * 0.4)
                + sinal_saturado * p_growl * 0.5
                + vogal_chunk * mod_ventricular
            )

        if p_air > 0.0:
            ruido_base = np.random.normal(0.0, 1000.0, len(vogal_chunk)).astype(np.float32)
            nyquist = 0.5 * params.framerate
            b, a = butter(4, 3800.0 / nyquist, btype="high")
            sopro_puro = lfilter(b, a, ruido_base)
            envelope_vocal = np.abs(vogal_chunk)
            pico_envelope = np.max(envelope_vocal) + 1e-5
            sopro_modulado = sopro_puro * (envelope_vocal / pico_envelope)
            vogal_chunk = vogal_chunk * (1.0 - p_air * 0.3) + sopro_modulado * p_air * 1.6

        pico_onda = np.max(np.abs(vogal_chunk))
        teto_maximo = 28000.0
        if pico_onda > teto_maximo:
            vogal_chunk = vogal_chunk * (teto_maximo / (pico_onda + 1e-5))

        audio_final = np.clip(
            np.concatenate((consonante_chunk, vogal_chunk)),
            -32768.0,
            32767.0,
        ).astype(np.int16)

    out_mem = io.BytesIO()
    with wave.open(out_mem, "wb") as w_out:
        w_out.setnchannels(params.nchannels)
        w_out.setsampwidth(2)
        w_out.setframerate(params.framerate)
        w_out.writeframes(audio_final.tobytes())
    return out_mem.getvalue()


def _mostrar_tela_importacao():
    st.subheader("Painel de Importação")
    uploaded_file = st.file_uploader(
        "Selecione o arquivo Zip contendo o banco de voz e o arquivo 'oto.ini'",
        type=["zip"],
    )

    if uploaded_file is None:
        return

    zip_bytes = uploaded_file.read()
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
            nomes = z.namelist()
            oto_nome = next(
                (name for name in nomes if os.path.basename(name).lower() == "oto.ini"),
                None,
            )
            if oto_nome is None:
                st.error("Arquivo 'oto.ini' não foi encontrado dentro do arquivo ZIP fornecido.")
                return

            st.session_state.linhas_oto_cruas = z.read(oto_nome)
            st.session_state.lista_audios = sorted(
                name for name in nomes
                if not name.endswith("/") and os.path.basename(name).lower().endswith(".wav")
            )
    except zipfile.BadZipFile:
        st.error("O arquivo enviado não é um ZIP válido.")
        return

    st.session_state.zip_bytes = zip_bytes
    st.session_state.nome_arquivo_zip = uploaded_file.name
    st.session_state.audio_selecionado = ""
    st.session_state.tela = 2
    st.rerun()


def _mostrar_tela_validacao():
    st.subheader("Validação de Codificação do Pacote")
    if st.session_state.linhas_oto_cruas is None:
        st.session_state.tela = 1
        st.rerun()

    enc_opcao = st.selectbox("Selecione o formato de encoding do texto:", ["UTF-8", "Shift-JIS"])
    st.session_state.encoding = enc_opcao
    encoding_alvo = "shift_jis" if enc_opcao == "Shift-JIS" else "utf-8"
    texto_decodificado = st.session_state.linhas_oto_cruas.decode(encoding_alvo, errors="ignore")
    linhas_preview = "\n".join(texto_decodificado.splitlines()[:12])
    st.text_area(
        "Estrutura interna detectada no arquivo 'oto.ini':",
        value=linhas_preview,
        height=220,
        disabled=True,
    )

    if st.button("CONFIRMAR CODIFICAÇÃO E AVANÇAR ➔"):
        st.session_state.dados_oto = interpretar_oto_ini(
            st.session_state.linhas_oto_cruas.decode(encoding_alvo, errors="ignore")
        )
        st.session_state.tela = 3
        st.rerun()


def _mostrar_tela_processamento():
    st.subheader("Identidade Digital do Banco de Voz")
    st.markdown("Formato de Metadados: MaxiVloid", unsafe_allow_html=True)

    nome_ch = st.text_input("Nome do Character:", value="MaxiVloid_Character")
    autor_ch = st.text_input("Autor / Desenvolvedor:", value="MVX_Developer")
    uploaded_icon = st.file_uploader(
        "Carregar Imagem de Ícone (máximo 1280x1280):",
        type=["png", "jpg", "jpeg"],
    )
    uploaded_ilusr = st.file_uploader(
        "Carregar Ilustração Completa do Avatar (.png/.jpg):",
        type=["png", "jpg", "jpeg"],
    )
    readme_text = st.text_area(
        "Conteúdo complementar para o README.txt (opcional):",
        height=150,
    )

    icone_bytes = None
    icone_nome = ""
    if uploaded_icon is not None:
        try:
            with Image.open(uploaded_icon) as img_ico:
                if img_ico.width > 1280 or img_ico.height > 1280:
                    st.error(
                        f"Erro: o ícone deve ter no máximo 1280x1280 pixels; "
                        f"a imagem possui {img_ico.width}x{img_ico.height}."
                    )
                else:
                    uploaded_icon.seek(0)
                    icone_bytes = uploaded_icon.getvalue()
                    icone_nome = uploaded_icon.name
        except UnidentifiedImageError:
            st.error("A imagem do ícone não é válida.")

    ilusr_bytes = None
    ilusr_nome = ""
    if uploaded_ilusr is not None:
        try:
            with Image.open(uploaded_ilusr) as img_ilusr:
                if img_ilusr.width > 1280 or img_ilusr.height > 1280:
                    st.error(
                        f"Erro: a ilustração deve ter no máximo 1280x1280 pixels; "
                        f"a imagem possui {img_ilusr.width}x{img_ilusr.height}."
                    )
                else:
                    uploaded_ilusr.seek(0)
                    ilusr_bytes = uploaded_ilusr.getvalue()
                    ilusr_nome = uploaded_ilusr.name
        except UnidentifiedImageError:
            st.error("A ilustração não é uma imagem válida.")

    with col_dir:
        st.subheader("Painel de Processamento")
        p_pitch = st.slider("Ajuste de Nota Musical (Semitons):", -6.0, 6.0, 0.0, 0.5)
        p_air = st.slider("Ar na Voz Dinâmico:", 0.0, 1.0, 0.0, 0.01)
        p_formant = st.slider("Timbre do Corpo Vocal:", 0.85, 1.15, 1.00, 0.01)
        p_gender = st.slider("Transição de Gênero Humano:", 0.80, 1.20, 1.00, 0.01)
        p_growl = st.slider("Drive Vocal Ventricular:", 0.0, 1.0, 0.0, 0.01)
        p_naturalness = st.slider("Naturalidade Emocional da Onda:", 0.0, 1.0, 1.0, 0.01)

        if st.session_state.lista_audios:
            opcoes = [os.path.basename(name) for name in st.session_state.lista_audios]
            audio_selecionado = st.selectbox(
                "Amostra de áudio para pré-visualização:",
                options=opcoes,
                index=opcoes.index(st.session_state.audio_selecionado)
                if st.session_state.audio_selecionado in opcoes
                else 0,
            )
            st.session_state.audio_selecionado = audio_selecionado

            if st.button("GERAR PRÉ-VISUALIZAÇÃO DE ÁUDIO"):
                nome_audio = next(
                    name for name in st.session_state.lista_audios
                    if os.path.basename(name) == audio_selecionado
                )
                try:
                    with zipfile.ZipFile(io.BytesIO(st.session_state.zip_bytes), "r") as z:
                        raw_audio = z.read(nome_audio)
                    preview_processed = processar_audio(
                        raw_audio,
                        st.session_state.dados_oto,
                        audio_selecionado,
                        p_pitch,
                        p_air,
                        p_formant,
                        p_gender,
                        p_growl,
                        p_naturalness,
                    )
                    st.audio(preview_processed, format="audio/wav")
                    st.success(f"Visualização gerada usando a amostra base: {audio_selecionado}")
                except (KeyError, ValueError, zipfile.BadZipFile) as exc:
                    st.error(f"Não foi possível processar a amostra: {exc}")

            if st.button("COMPILAR E COMPACTAR PACOTE DE DISTRIBUIÇÃO (.ZIP)"):
                if not st.session_state.lista_audios:
                    st.warning("Nenhum áudio foi encontrado no pacote.")
                elif not nome_ch.strip() or not autor_ch.strip():
                    st.warning("Informe o nome e o desenvolvedor do character.")
                else:
                    with st.spinner("Injetando módulos de configuração no formato MaxiVloid..."):
                        out_zip_io = io.BytesIO()
                        conteudo_mxai = [
                            "[MxAi_Configuration]",
                            "EngineCompatible=MaximizeVoiceSynthesizerMobile",
                            "GlobalType=MaxiVloid",
                            f"GlobalPitchShift={p_pitch:.2f}",
                            f"GlobalAirRatio={p_air:.2f}",
                            f"GlobalFormantShift={p_formant:.2f}",
                            f"GlobalGenderFactor={p_gender:.2f}",
                            f"GlobalGrowlAmount={p_growl:.2f}",
                            f"GlobalNaturalness={p_naturalness:.2f}",
                            "",
                            "[Amostras_Calibradas]",
                        ]
                        for caminho_audio in st.session_state.lista_audios:
                            nome_arquivo = os.path.basename(caminho_audio)
                            parametro_consonante = st.session_state.dados_oto.get(
                                nome_arquivo,
                                {},
                            ).get("consonant", "0")
                            conteudo_mxai.append(
                                f"{nome_arquivo}=Pitch={p_pitch:.2f},Air={p_air:.2f},"
                                f"Formant={p_formant:.2f},Gender={p_gender:.2f},"
                                f"Growl={p_growl:.2f},Naturalness={p_naturalness:.2f},"
                                f"ProtectConsonant={parametro_consonante}"
                            )

                        conteudo_txt = [
                            "type=MaxiVloid",
                            f"name={nome_ch}",
                            f"image={ilusr_nome}",
                            f"author={autor_ch}",
                            f"icon={icone_nome}",
                            "version=Maximize_Voice_2.0",
                        ]
                        conteudo_yaml = [
                            "type: MaxiVloid",
                            f"name: {nome_ch}",
                            f"author: {autor_ch}",
                            f"image: {ilusr_nome}",
                            f"icon: {icone_nome}",
                            "voice_engine: MaximizeVoiceSynthesizerMobile",
                        ]

                        with zipfile.ZipFile(out_zip_io, "w", zipfile.ZIP_DEFLATED) as z_out:
                            z_out.writestr("MxAi.ini", "\n".join(conteudo_mxai) + "\n")
                            z_out.writestr("character.txt", "\n".join(conteudo_txt) + "\n")
                            z_out.writestr("character.yaml", "\n".join(conteudo_yaml) + "\n")
                            if readme_text.strip():
                                z_out.writestr("README.txt", readme_text)
                            if icone_bytes:
                                z_out.writestr(icone_nome, icone_bytes)
                            if ilusr_bytes:
                                z_out.writestr(ilusr_nome, ilusr_bytes)

                            with zipfile.ZipFile(io.BytesIO(st.session_state.zip_bytes), "r") as z_in:
                                for item in z_in.infolist():
                                    if item.is_dir():
                                        continue
                                    nome_base = os.path.basename(item.filename)
                                    if nome_base.lower() == "oto.ini" or nome_base.lower().endswith(".wav"):
                                        z_out.writestr(nome_base, z_in.read(item.filename))

                        st.download_button(
                            label="BAIXAR BANCO DE VOZ COMPILADO MAXIVLOID 🎉",
                            data=out_zip_io.getvalue(),
                            file_name="banco_maxivloid_final.zip",
                            mime="application/zip",
                        )
        else:
            st.info("Aguardando a conclusão das etapas de importação do arquivo no painel esquerdo.")


col_esq, col_dir = st.columns(2)
with col_esq:
    st.title("MVX VoiceBank Creator")

    if st.session_state.tela == 1:
        _mostrar_tela_importacao()
    elif st.session_state.tela == 2:
        _mostrar_tela_validacao()
    elif st.session_state.tela == 3:
        st.subheader("Banco de Voz Carregado")
        st.write(f"Arquivo ZIP: {st.session_state.nome_arquivo_zip}")
        st.write(f"Áudios encontrados: {len(st.session_state.lista_audios)}")
        if st.button("VOLTAR PARA IMPORTAÇÃO"):
            st.session_state.tela = 1
            st.rerun()
    else:
        st.session_state.tela = 1

if st.session_state.tela == 3:
    _mostrar_tela_processamento()
