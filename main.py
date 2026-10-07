import os
import sys
import json
import base64
import wave
import math
import struct
import io
import zipfile
from scipy.ndimage import zoom
from scipy.signal import lfilter, butter
try:
import numpy as np
import streamlit as st
from PIL import Image
except ImportError:
import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "numpy", "scipy", "Pillow", "--quiet"])
import numpy as np
import streamlit as st
from PIL import Image
st.set_page_config(
page_title="MVX VoiceBank Creator",
page_icon="🟢",
layout="wide",
initial_sidebar_state="expanded"
)
st.markdown("""

.stApp {
background-color: #121212;
}
h1, h2, h3, p, label, .stSlider {
color: #ffffff !important;
}
.stButton>button {
background-color: #39FF14 !important;
color: #000000 !important;
font-weight: bold !important;
border: none !important;
width: 100% !important;
height: 45px !important;
}
.stButton>button:hover {
background-color: #2EE610 !important;
color: #000000 !important;
}
.import-btn>div>button {
background-color: #FFF000 !important;
color: #000000 !important;
}
.import-btn>div>button:hover {
background-color: #CCFF00 !important;
}
.stTextArea>div>div>textarea {
background-color: #1A1A1A !important;
color: #39FF14 !important;
font-family: 'Consolas', monospace !important;
}
function checkDevice() {
var isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
if (isMobile && window.innerWidth < 1024) {
alert("Aviso: Este sistema requer a ativação do 'Modo Computador' / 'Para Computador' nas configurações do seu navegador móvel para o correto funcionamento da área de trabalho.");
}
}
setTimeout(checkDevice, 1000);

""", unsafe_allow_html=True)
if 'tela' not in st.session_state:
st.session_state.tela = 1
if 'zip_bytes' not in st.session_state:
st.session_state.zip_bytes = None
if 'linhas_oto_cruas' not in st.session_state:
st.session_state.linhas_oto_cruas = None
if 'nome_arquivo_zip' not in st.session_state:
st.session_state.nome_arquivo_zip = ""
if 'dados_oto' not in st.session_state:
st.session_state.dados_oto = {}
if 'lista_audios' not in st.session_state:
st.session_state.lista_audios = []
if 'encoding' not in st.session_state:
st.session_state.encoding = "UTF-8"
def interpretar_oto_ini(conteudo):
dados = {}
for linha in conteudo.splitlines():
if '=' in linha:
nome_wav, dados_linha = linha.split('=', 1)
parametros = dados_linha.split(',')
if len(parametros) >= 1:
dados[nome_wav.strip()] = {
"alias": parametros[0].strip(),
"consonant": parametros[1].strip() if len(parametros) > 1 and parametros[1].strip() else "0"
}
return dados
def phase_vocoder_formante(dados_espectro, fator):
if fator == 1.0: return dados_espectro
tamanho = len(dados_espectro)
indices_origem = np.clip(np.arange(tamanho) * fator, 0, tamanho - 1).astype(np.int32)
return dados_espectro[indices_origem]
def processar_audio(audio_bytes_in, dados_oto, nome_wav_puro, p_pitch, p_air, p_formant, p_gender, p_growl, p_naturalness):
pitch_factor = 2 ** (p_pitch / 12.0)
fator_total = p_formant * p_gender
with wave.open(io.BytesIO(audio_bytes_in), 'rb') as w_in:
params = w_in.getparams()
audio_data = np.frombuffer(w_in.readframes(params.nframes), dtype=np.int16).astype(np.float32) if params.sampwidth == 2 else (np.frombuffer(w_in.readframes(params.nframes), dtype=np.uint8).astype(np.float32) - 128.0) * 256.0
try: consonante_ms = float(dados_oto.get(nome_wav_puro, {}).get("consonant", "0"))
except ValueError: consonante_ms = 0.0
ponto_corte = min(int((consonante_ms / 1000.0) * params.framerate), len(audio_data))
consonante_chunk = audio_data[:ponto_corte]
vogal_chunk = audio_data[ponto_corte:]
if len(vogal_chunk) > 0:
if p_naturalness > 0.05:
freq_vibrato = 5.6 + np.random.normal(0, 0.4)
amplitude_vibrato = 0.022 * p_naturalness
tempo = np.arange(len(vogal_chunk)) / params.framerate
pontos_vibrato = np.sin(2.0 * math.pi * freq_vibrato * tempo) * amplitude_vibrato
indices_organicos = np.clip(np.arange(len(vogal_chunk)) + pontos_vibrato * params.framerate * 0.01, 0, len(vogal_chunk) - 1).astype(np.int32)
vogal_chunk = vogal_chunk[indices_organicos]
if pitch_factor != 1.0:
vogal_chunk = zoom(vogal_chunk, 1.0 / pitch_factor)
if fator_total != 1.0:
t_janela, passo = 1024, 256
j_hann = np.hanning(t_janela)
stft_res = []
for i in range(0, len(vogal_chunk) - t_janela, passo):
espectro = np.fft.rfft(vogal_chunk[i:i+t_janela] * j_hann)
indices = np.clip(np.arange(len(espectro)) * fator_total, 0, len(espectro) - 1).astype(np.int32)
esp_mod = espectro[indices]
esp_mod = np.pad(esp_mod, (0, len(espectro) - len(esp_mod))) if len(esp_mod) < len(espectro) else esp_mod[:len(espectro)]
stft_res.append(np.fft.irfft(esp_mod))
if len(stft_res) > 0:
recon = np.zeros(len(stft_res) * passo + t_janela)
for idx, b in enumerate(stft_res): recon[idxpasso:idxpasso+t_janela] += b
vogal_chunk = recon[:len(vogal_chunk)]
if p_growl > 0.0:
tempo_g = np.arange(len(vogal_chunk)) / params.framerate
mod_ventricular = np.sin(2.0 * math.pi * 60.0 * tempo_g) * (p_growl * 0.28)
sinal_saturado = vogal_chunk / (np.max(np.abs(vogal_chunk)) + 1e-5)
sinal_saturado = np.arctan(sinal_saturado * (1.0 + p_growl * 2.5)) * np.max(np.abs(vogal_chunk))
vogal_chunk = (vogal_chunk * (1.0 - (p_growl * 0.4))) + (sinal_saturado * p_growl * 0.5) + (vogal_chunk * mod_ventricular)
if p_air > 0.0:
ruido_base = np.random.normal(0.0, 1000.0, len(vogal_chunk))
nyquist = 0.5 * params.framerate
b, a = butter(4, 3800.0 / nyquist, btype='high')
sopro_puro = lfilter(b, a, ruido_base)
envelope_vocal = np.abs(vogal_chunk)
sopro_modulado = sopro_puro * (envelope_vocal / (np.max(envelope_vocal) + 1e-5))
vogal_chunk = (vogal_chunk * (1.0 - (p_air * 0.3))) + (sopro_modulado * p_air * 1.6)
teto_maximo = 28000.0
picos_onda = np.max(np.abs(vogal_chunk))
if picos_onda > teto_maximo:
vogal_chunk = vogal_chunk * (teto_maximo / (picos_onda + 1e-5))
audio_final = np.clip(np.concatenate((consonante_chunk, vogal_chunk)), -32768.0, 32767.0).astype(np.int16)
out_mem = io.BytesIO()
with wave.open(out_mem, 'wb') as w_out:
w_out.setnchannels(params.nchannels)
w_out.setsampwidth(2)
w_out.setframerate(params.framerate)
w_out.writeframes(audio_final.tobytes())
return out_mem.getvalue()
col_esq, col_dir = st.columns(2)
with col_esq:
st.title("MVX VoiceBank Creator")
if st.session_state.tela == 1:
st.subheader("Painel de Importação")
st.markdown("", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Selecione o arquivo Zip contendo o banco de voz e o arquivo 'oto.ini'", type=["zip"])
st.markdown("", unsafe_allow_html=True)
if uploaded_file is not None:
zip_bytes = uploaded_file.read()
encontrou_oto = False
with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as z:
for name in z.namelist():
if os.path.basename(name).lower() == 'oto.ini':
encontrou_oto = True
with z.open(name) as f:
st.session_state.linhas_oto_cruas = f.read()
break
if not encontrou_oto:
st.error("Arquivo 'oto.ini' não foi encontrado dentro do arquivo ZIP fornecido.")
else:
st.session_state.zip_bytes = zip_bytes
st.session_state.nome_arquivo_zip = uploaded_file.name
st.session_state.tela = 2
st.rerun()
elif st.session_state.tela == 2:
st.subheader("Validação de Codificação do Pacote")
enc_opcao = st.selectbox("Selecione o formato de encoding do texto:", ["UTF-8", "Shift-JIS"])
st.session_state.encoding = enc_opcao
encoding_alvo = "shift_jis" if enc_opcao == "Shift-JIS" else "utf-8"
texto_decodificado = st.session_state.linhas_oto_cruas.decode(encoding_alvo, errors='ignore')
linhas_preview = "\n".join(texto_decodificado.splitlines()[:12])
st.text_area("Estrutura interna detectada no arquivo 'oto.ini':", value=linhas_preview, height=220, disabled=True)
if st.button("CONFIRMAR CODIFICAÇÃO E AVANÇAR ➔"):
st.session_state.dados_oto = interpretar_oto_ini(st.session_state.linhas_oto_cruas.decode(encoding_alvo, errors='ignore'))
with zipfile.ZipFile(io.BytesIO(st.session_state.zip_bytes), 'r') as z:
st.session_state.lista_audios = [n for n in z.namelist() if n.lower().endswith('.wav')]
st.session_state.tela = 3
st.rerun()
elif st.session_state.tela == 3:
st.subheader("Identidade Digital do Banco de Voz")
st.markdown("Formato de Metadados: MaxiVloid", unsafe_allow_html=True)
nome_ch = st.text_input("Nome do Character:", value="MaxiVloid_Character")
autor_ch = st.text_input("Autor / Desenvolvedor:", value="MVX_Developer")
uploaded_icon = st.file_uploader("Carregar Imagem de Ícone (Apenas quadratura limitada a 1280x1280):", type=["png", "jpg", "jpeg"])
uploaded_ilusr = st.file_uploader("Carregar Ilustração Completa do Avatar (.png/.jpg):", type=["png", "jpg", "jpeg"])
readme_text = st.text_area("Escreva o conteúdo para o arquivo complementar README.txt (Opcional):", height=150)
icone_bytes, icone_nome = None, ""
if uploaded_icon is not None:
img_ico = Image.open(uploaded_icon)
if img_ico.size[0] > 1280 or img_ico.size[1] > 1280:
st.error(f"Erro: O tamanho máximo permitido para o ícone é 1280x1280 pixels. A imagem enviada possui {img_ico.size[0]}x{img_ico.size[1]}.")
else:
icone_bytes = uploaded_icon.getvalue()
icone_nome = uploaded_icon.name
ilusr_bytes, ilusr_nome = None, ""
if uploaded_ilusr is not None:
ilusr_bytes = uploaded_ilusr.getvalue()
ilusr_nome = uploaded_ilusr.name
with col_dir:
st.subheader("Painel de Processamento")
p_pitch = st.slider("Ajuste de Nota Musical (Semitons):", min_value=-6.0, max_value=6.0, value=0.0, step=0.5)
p_air = st.slider("Ar na Voz Dinâmico (Sopro de Fonação):", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
p_formant = st.slider("Timbre do Corpo Vocal (Formantes Estáveis):", min_value=0.85, max_value=1.15, value=1.00, step=0.01)
p_gender = st.slider("Transição de Gênero Humano (Trato Vocal):", min_value=0.80, max_value=1.20, value=1.00, step=0.01)
p_growl = st.slider("Drive Vocal Ventricular (Growl Natural):", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
p_naturalness = st.slider("Naturalidade Emocional da Onda (Vibrato Orgânico):", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
st.markdown("
", unsafe_allow_html=True)
if st.session_state.tela == 3 and len(st.session_state.lista_audios) > 0:
if st.button("GERAR E CONFIGURAR PRÉ-VISUALIZAÇÃO DE ÁUDIO"):
with st.spinner("Processando sinais..."):
nome_amostra = st.session_state.lista_audios[0]
with zipfile.ZipFile(io.BytesIO(st.session_state.zip_bytes), 'r') as z:
raw_audio = z.read(nome_amostra)
preview_processed = processar_audio(
raw_audio, st.session_state.dados_oto, os.path.basename(nome_amostra),
p_pitch, p_air, p_formant, p_gender, p_growl, p_naturalness
)
st.audio(preview_processed, format="audio/wav")
st.success(f"Visualização gerada usando a amostra base: {os.path.basename(nome_amostra)}")
if st.button("COMPILAR E COMPACTAR PACOTE DE DISTRIBUIÇÃO (.ZIP)"):
with st.spinner("Injetando módulos de configuração no formato MaxiVloid..."):
out_zip_io = io.BytesIO()
conteudo_mxai = [
"[MxAi_Configuration]", "EngineCompatible=MaximizeVoiceSynthesizerMobile", "GlobalType=MaxiVloid",
f"GlobalPitchShift={p_pitch:.2f}", f"GlobalAirRatio={p_air:.2f}",
f"GlobalFormantShift={p_formant:.2f}", f"GlobalGenderFactor={p_gender:.2f}",
f"GlobalGrowlAmount={p_growl:.2f}", f"GlobalNaturalness={p_naturalness:.2f}\n",
"[Amostras_Calibradas]"
]
for c_wav in st.session_state.lista_audios:
w = os.path.basename(c_wav)
conteudo_mxai.append(f"{w}=Pitch={p_pitch:.2f},Air={p_air:.2f},Formant={p_formant:.2f},Gender={p_gender:.2f},Growl={p_growl:.2f},Naturalness={p_naturalness:.2f},ProtectConsonant={st.session_state.dados_oto.get(w, {}).get('consonant', '0')}")
conteudo_txt = [f"type=MaxiVloid", f"name={nome_ch}", f"image={ilusr_nome}", f"author={autor_ch}", f"icon={icone_nome}", "version=Maximize_Voice_2.0"]
conteudo_yaml = [f"type: MaxiVloid", f"name: {nome_ch}", f"author: {autor_ch}", f"image: {ilusr_nome}", f"icon: {icone_nome}", "voice_engine: MaximizeVoiceSynthesizerMobile"]
with zipfile.ZipFile(out_zip_io, 'w', zipfile.ZIP_DEFLATED) as z_out:
z_out.writestr("MxAi.ini", "\n".join(conteudo_mxai))
z_out.writestr("character.txt", "\n".join(conteudo_txt))
z_out.writestr("character.yaml", "\n".join(conteudo_yaml))
if readme_text.strip():
z_out.writestr("README.txt", readme_text)
if icone_bytes:
z_out.writestr(icone_nome, icone_bytes)
if ilusr_bytes:
z_out.writestr(ilusr_nome, ilusr_bytes)
with zipfile.ZipFile(io.BytesIO(st.session_state.zip_bytes), 'r') as z_in:
for item in z_in.infolist():
if not item.filename.endswith('/'):
base_n = os.path.basename(item.filename)
if base_n.lower() == 'oto.ini' or base_n.lower().endswith('.wav'):
z_out.writestr(base_n, z_in.read(item.filename))
st.download_button(
label="BAIXAR BANCO DE VOZ COMPILADO MAXIVLOID 🎉",
data=out_zip_io.getvalue(),
file_name="banco_maxivloid_final.zip",
mime="application/zip"
)
else:
st.info("Aguardando a conclusão das etapas de importação do arquivo no painel esquerdo.")
