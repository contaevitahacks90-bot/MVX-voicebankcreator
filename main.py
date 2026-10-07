import io
import json
import math
import os
import tempfile
import time
import tkinter as tk
import wave
import zipfile
from pathlib import PurePosixPath
from tkinter import filedialog, messagebox

try:
    import winsound
except ImportError:
    winsound = None

try:
    import customtkinter as ctk
    import numpy as np
    from PIL import Image
    from scipy.signal import butter, lfilter, resample
except ImportError as exc:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Dependência ausente",
        f"Não foi possível iniciar o editor: {exc}\n\n"
        "Instale as dependências com:\n"
        "pip install customtkinter Pillow numpy scipy",
    )
    root.destroy()
    raise SystemExit(1) from exc


ctk.set_appearance_mode("Dark")


class MvxCreatorApp(ctk.CTk):
    BG = "#121212"
    PANEL = "#1E1E1E"
    GREEN = "#39FF14"
    YELLOW = "#FFF000"

    def __init__(self):
        super().__init__()
        self.title("MVX VoiceBank Creator")
        self.geometry("960x700")
        self.minsize(900, 620)
        self.configure(fg_color=self.BG)

        self.source_zip = None
        self.oto_bytes = b""
        self.oto_data = {}
        self.audio_files = []
        self.icon_path = ""
        self.image_path = ""
        self.fullscreen = False
        self.preview_dir = tempfile.TemporaryDirectory(prefix="mvx-preview-")
        self.preview_path = None
        self._build_ui()
        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", self._leave_fullscreen)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=4, minsize=360)
        self.grid_columnconfigure(1, weight=6, minsize=500)
        self.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(self, fg_color=self.PANEL, corner_radius=0)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        ctk.CTkButton(
            left, text="⛶", width=35, command=self._toggle_fullscreen,
            fg_color="#1A1A1A", border_color="#333333", border_width=1,
        ).pack(anchor="w", padx=15, pady=15)
        ctk.CTkLabel(
            left, text="MVX VoiceBank Creator",
            font=ctk.CTkFont(size=20, weight="bold"), text_color=self.YELLOW,
        ).pack(anchor="w", padx=20, pady=(0, 10))
        status_frame = ctk.CTkFrame(left, fg_color=self.BG, border_color=self.GREEN, border_width=1)
        status_frame.pack(fill="x", padx=20, pady=10)
        self.status_label = ctk.CTkLabel(
            status_frame, text="Aguardando importação do pacote de voz... 🟢",
            text_color="white", wraplength=330, justify="left",
        )
        self.status_label.pack(padx=12, pady=12, fill="x")
        self.flow = ctk.CTkFrame(left, fg_color="transparent")
        self.flow.pack(expand=True, fill="both", padx=20, pady=10)
        self._show_import()

        right = ctk.CTkFrame(self, fg_color=self.BG, corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew", padx=25, pady=20)
        ctk.CTkLabel(
            right, text="PAINEL DE PROCESSAMENTO DIGITAL (VOICER ADJUSTER)",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=self.GREEN, wraplength=470,
        ).pack(anchor="w", pady=(5, 5))
        self.audio_combo = ctk.CTkComboBox(right, values=["Importe um pacote ZIP primeiro"])
        self.audio_combo.set("Importe um pacote ZIP primeiro")
        self.audio_combo.pack(fill="x", pady=(4, 12))
        self.audio_combo.configure(state="disabled")

        self.pitch_label, self.pitch = self._add_slider(right, "Nota musical: 0.0 semitons", -6, 6, 0, 24, self._pitch_changed)
        self.air_label, self.air = self._add_slider(right, "Ar na voz: 0%", 0, 0.6, 0, 60, self._air_changed)
        self.formant_label, self.formant = self._add_slider(right, "Timbre vocal: 1.00x", 0.85, 1.15, 1, 60, self._formant_changed)
        self.gender_label, self.gender = self._add_slider(right, "Fator de gênero: 1.00", 0.8, 1.2, 1, 80, self._gender_changed)
        self.growl_label, self.growl = self._add_slider(right, "Growl: 0%", 0, 1, 0, 100, self._growl_changed)
        self.natural_label, self.naturalness = self._add_slider(right, "Naturalidade: 100%", 0, 1, 1, 100, self._natural_changed)
        self.sliders = (self.pitch, self.air, self.formant, self.gender, self.growl, self.naturalness)
        self.preview_button = ctk.CTkButton(
            right, text="PROCESSAR E OUVIR PREVIEW", command=self._play_preview,
            state="disabled", fg_color="transparent", text_color=self.GREEN,
            border_color=self.GREEN, border_width=1,
        )
        self.preview_button.pack(fill="x", pady=4)
        self.export_button = ctk.CTkButton(
            right, text="COMPACTAR E EXPORTAR BANCO MAXIVLOID", command=self._export,
            state="disabled", fg_color=self.GREEN, text_color="black",
        )
        self.export_button.pack(fill="x", pady=(4, 0))

    @staticmethod
    def _add_slider(parent, label, minimum, maximum, initial, steps, callback):
        text = ctk.CTkLabel(parent, text=label, text_color="white")
        text.pack(anchor="w", pady=(5, 1))
        slider = ctk.CTkSlider(
            parent, from_=minimum, to=maximum, number_of_steps=steps,
            button_color="#39FF14", progress_color="#39FF14", command=callback,
        )
        slider.set(initial)
        slider.pack(fill="x", pady=(0, 9))
        slider.configure(state="disabled")
        return text, slider

    def _toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        self.attributes("-fullscreen", self.fullscreen)
        return "break"

    def _leave_fullscreen(self, event=None):
        if self.fullscreen:
            self.fullscreen = False
            self.attributes("-fullscreen", False)
        return "break"

    def _clear_flow(self):
        for widget in self.flow.winfo_children():
            widget.destroy()

    def _show_import(self):
        self._clear_flow()
        ctk.CTkButton(
            self.flow, text="Importar pacote ZIP [oto.ini requerido]", height=45,
            fg_color=self.YELLOW, text_color="black", command=self._import_zip,
        ).pack(fill="x", pady=80)

    def _show_encoding(self):
        self._clear_flow()
        self.status_label.configure(text="Confira a visualização do oto.ini e selecione o encoding.")
        ctk.CTkLabel(self.flow, text="Visualização do oto.ini:", text_color="white").pack(anchor="w")
        self.preview_text = tk.Text(
            self.flow, height=8, bg=self.BG, fg=self.GREEN, insertbackground="white",
            font=("Consolas", 10),
        )
        self.preview_text.pack(fill="x", pady=(0, 10))
        self.encoding_combo = ctk.CTkComboBox(
            self.flow, values=["UTF-8", "Shift-JIS"], command=self._update_oto_preview,
        )
        self.encoding_combo.set("UTF-8")
        self.encoding_combo.pack(fill="x", pady=(0, 15))
        ctk.CTkButton(
            self.flow, text="CONTINUAR ➔", fg_color=self.GREEN, text_color="black",
            command=self._confirm_encoding,
        ).pack(fill="x")
        self._update_oto_preview("UTF-8")

    def _show_identity(self):
        self._clear_flow()
        self.status_label.configure(text="Configure a identidade do personagem.")
        ctk.CTkLabel(
            self.flow, text="IDENTIDADE DO PERSONAGEM (MAXIVLOID)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=self.YELLOW,
        ).pack(anchor="w", pady=(0, 5))
        ctk.CTkLabel(self.flow, text="Nome:", text_color="white").pack(anchor="w")
        self.name_entry = ctk.CTkEntry(self.flow, fg_color=self.BG, text_color="white")
        self.name_entry.pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(self.flow, text="Autor / Desenvolvedor:", text_color="white").pack(anchor="w")
        self.author_entry = ctk.CTkEntry(self.flow, fg_color=self.BG, text_color="white")
        self.author_entry.pack(fill="x", pady=(0, 8))
        ctk.CTkButton(
            self.flow, text="Carregar ícone (máx. 1280x1280)",
            command=self._select_icon,
        ).pack(fill="x", pady=2)
        self.icon_label = ctk.CTkLabel(self.flow, text="", text_color="grey")
        self.icon_label.pack(anchor="w")
        ctk.CTkButton(
            self.flow, text="Carregar ilustração", command=self._select_image,
        ).pack(fill="x", pady=2)
        self.image_label = ctk.CTkLabel(self.flow, text="", text_color="grey")
        self.image_label.pack(anchor="w")
        ctk.CTkLabel(self.flow, text="README.txt (opcional):", text_color="white").pack(anchor="w")
        self.readme_text = tk.Text(
            self.flow, height=4, bg=self.BG, fg="white", insertbackground="white",
        )
        self.readme_text.pack(fill="x")

    @staticmethod
    def _zip_matches(archive, basename):
        return [
            item.filename for item in archive.infolist()
            if not item.is_dir()
            and PurePosixPath(item.filename.replace("\\", "/")).name.casefold() == basename.casefold()
        ]

    def _import_zip(self):
        filename = filedialog.askopenfilename(
            title="Selecionar pacote de voz ZIP",
            filetypes=[("Arquivos ZIP", "*.zip")],
            parent=self,
        )
        if not filename:
            return
        try:
            with zipfile.ZipFile(filename, "r") as archive:
                corrupt_file = archive.testzip()
                if corrupt_file:
                    raise ValueError(f"Arquivo corrompido no ZIP: {corrupt_file}")
                oto_matches = self._zip_matches(archive, "oto.ini")
                if len(oto_matches) != 1:
                    raise ValueError(f"Esperado um oto.ini; encontrados: {len(oto_matches)}.")
                audio_files = [
                    item.filename for item in archive.infolist()
                    if not item.is_dir()
                    and PurePosixPath(item.filename.replace("\\", "/")).suffix.casefold() == ".wav"
                ]
                if not audio_files:
                    raise ValueError("O pacote não contém arquivos WAV.")
                oto_bytes = archive.read(oto_matches[0])
        except (OSError, zipfile.BadZipFile, RuntimeError, ValueError) as exc:
            messagebox.showerror("Erro de leitura", f"Não foi possível validar o ZIP:\n{exc}", parent=self)
            return

        self.source_zip = filename
        self.audio_files = audio_files
        self.oto_bytes = oto_bytes
        self.oto_data = {}
        self._show_encoding()

    def _update_oto_preview(self, encoding):
        if not self.oto_bytes:
            return
        codec = "shift_jis" if encoding == "Shift-JIS" else "utf-8"
        preview = self.oto_bytes.decode(codec, errors="replace")
        self.preview_text.configure(state="normal")
        self.preview_text.delete("1.0", tk.END)
        self.preview_text.insert(tk.END, "\n".join(preview.splitlines()[:12]))
        self.preview_text.configure(state="disabled")

    @staticmethod
    def _parse_oto(text):
        parsed = {}
        for line_number, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith(";") or "=" not in line:
                continue
            wav_name, raw_parameters = line.split("=", 1)
            parameters = [part.strip() for part in raw_parameters.split(",")]
            if not wav_name.strip() or len(parameters) < 6:
                continue
            try:
                numbers = [float(value) for value in parameters[1:6]]
            except ValueError as exc:
                raise ValueError(f"Valor numérico inválido na linha {line_number}.") from exc
            parsed[PurePosixPath(wav_name.replace("\\", "/")).name.casefold()] = {
                "alias": parameters[0],
                "offset": numbers[0],
                "consonant": numbers[1],
                "cutoff": numbers[2],
                "preutterance": numbers[3],
                "overlap": numbers[4],
            }
        if not parsed:
            raise ValueError("oto.ini não contém nenhuma entrada válida.")
        return parsed

    def _confirm_encoding(self):
        codec = "shift_jis" if self.encoding_combo.get() == "Shift-JIS" else "utf-8"
        try:
            self.oto_data = self._parse_oto(self.oto_bytes.decode(codec, errors="replace"))
        except ValueError as exc:
            messagebox.showerror("Erro de formato", str(exc), parent=self)
            return
        self.audio_combo.configure(values=self.audio_files, state="readonly")
        self.audio_combo.set(self.audio_files[0])
        for slider in self.sliders:
            slider.configure(state="normal")
        self.preview_button.configure(state="normal")
        self.export_button.configure(state="normal")
        self.status_label.configure(
            text=f"Pronto: {len(self.audio_files)} WAVs e {len(self.oto_data)} entradas oto.ini."
        )
        self._show_identity()

    def _select_icon(self):
        self._select_image_file(icon=True)

    def _select_image(self):
        self._select_image_file(icon=False)

    def _select_image_file(self, icon):
        path = filedialog.askopenfilename(
            title="Selecionar imagem",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.webp")],
            parent=self,
        )
        if not path:
            return
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            if icon and (width > 1280 or height > 1280):
                raise ValueError(f"O ícone excede o limite 1280x1280 ({width}x{height}).")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Imagem inválida", str(exc), parent=self)
            return
        if icon:
            self.icon_path = path
            self.icon_label.configure(text=os.path.basename(path))
        else:
            self.image_path = path
            self.image_label.configure(text=os.path.basename(path))

    @staticmethod
    def _decode_pcm(data, width, channels):
        if width == 1:
            audio = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128) / 128
        elif width == 2:
            audio = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768
        elif width == 3:
            raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
            value = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
            value = np.where(value & 0x800000, value - 0x1000000, value)
            audio = value.astype(np.float32) / 8388608
        elif width == 4:
            audio = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648
        else:
            raise ValueError(f"PCM de {width} bytes por amostra não é suportado.")
        if channels > 1:
            if audio.size % channels:
                raise ValueError("Dados PCM incompletos para a quantidade de canais.")
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio

    def _render_audio(self, samples, rate, filename):
        pitch = 2 ** (self.pitch.get() / 12)
        key = PurePosixPath(filename.replace("\\", "/")).name.casefold()
        consonant_ms = max(0.0, self.oto_data.get(key, {}).get("consonant", 0.0))
        split = min(int(consonant_ms * rate / 1000), len(samples))
        consonant = samples[:split]
        vowel = samples[split:].copy()
        if len(vowel) > 1 and not math.isclose(pitch, 1.0, rel_tol=1e-5):
            vowel = resample(vowel, max(1, round(len(vowel) / pitch))).astype(np.float32)

        formant = self.formant.get() * self.gender.get()
        if len(vowel) > 1 and not math.isclose(formant, 1.0, rel_tol=1e-5):
            spectrum = np.fft.rfft(vowel)
            old_bins = np.arange(len(spectrum))
            new_bins = np.clip(old_bins * formant, 0, len(spectrum) - 1)
            real = np.interp(new_bins, old_bins, spectrum.real)
            imag = np.interp(new_bins, old_bins, spectrum.imag)
            vowel = np.fft.irfft(real + 1j * imag, n=len(vowel)).astype(np.float32)

        growl = self.growl.get()
        if len(vowel) and growl > 0:
            peak = max(float(np.max(np.abs(vowel))), 1e-8)
            time_axis = np.arange(len(vowel)) / rate
            modulation = np.sin(2 * math.pi * 60 * time_axis) * growl * 0.12
            saturated = np.arctan((vowel / peak) * (1 + growl * 2.5)) / (math.pi / 2)
            vowel = (vowel * (1 - growl * 0.25) + saturated * growl * 0.25) * (1 + modulation)

        air = self.air.get()
        cutoff = min(3800.0, rate * 0.45)
        if len(vowel) and air > 0 and cutoff > 0:
            b, a = butter(4, cutoff / (rate / 2), btype="high")
            noise = lfilter(b, a, np.random.normal(0, 0.02, len(vowel)))
            envelope = np.abs(vowel)
            noise *= envelope / max(float(np.max(envelope)), 1e-8)
            vowel = vowel * (1 - air * 0.2) + noise * air

        naturalness = self.naturalness.get()
        if len(vowel) and naturalness > 0:
            time_axis = np.arange(len(vowel)) / rate
            vibrato = 1 + np.sin(2 * math.pi * 5.6 * time_axis) * naturalness * 0.00015
            vowel = vowel * vibrato
        rendered = np.concatenate((consonant, vowel))
        peak = float(np.max(np.abs(rendered))) if rendered.size else 0.0
        if peak > 0.98:
            rendered *= 0.98 / peak
        return np.clip(rendered * 32767, -32768, 32767).astype("<i2")

    def _play_preview(self):
        if winsound is None:
            messagebox.showerror("Áudio", "A reprodução de preview requer Windows.", parent=self)
            return
        filename = self.audio_combo.get()
        if filename not in self.audio_files:
            messagebox.showerror("Áudio", "Selecione uma amostra WAV.", parent=self)
            return
        try:
            with zipfile.ZipFile(self.source_zip, "r") as archive:
                audio_bytes = archive.read(filename)
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
                if wav.getcomptype() != "NONE":
                    raise ValueError("O WAV precisa usar PCM sem compressão.")
                rate = wav.getframerate()
                samples = self._decode_pcm(
                    wav.readframes(wav.getnframes()), wav.getsampwidth(), wav.getnchannels()
                )
            if not samples.size:
                raise ValueError("O WAV está vazio.")
            rendered = self._render_audio(samples, rate, filename)
            output = io.BytesIO()
            with wave.open(output, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(rendered.tobytes())
            winsound.PlaySound(None, winsound.SND_PURGE)
            if self.preview_path and os.path.exists(self.preview_path):
                os.remove(self.preview_path)
            self.preview_path = os.path.join(self.preview_dir.name, f"preview_{time.time_ns()}.wav")
            with open(self.preview_path, "wb") as preview_file:
                preview_file.write(output.getvalue())
            winsound.PlaySound(self.preview_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, wave.Error) as exc:
            messagebox.showerror("Erro de áudio", f"Não foi possível processar a amostra:\n{exc}", parent=self)

    @staticmethod
    def _metadata_value(value):
        return str(value).replace("\r", " ").replace("\n", " ").strip()

    def _export(self):
        output_path = filedialog.asksaveasfilename(
            title="Exportar banco de voz MVX",
            defaultextension=".zip",
            filetypes=[("Arquivo ZIP", "*.zip")],
            initialfile="banco_maxivloid_final.zip",
            parent=self,
        )
        if not output_path:
            return
        if os.path.normcase(os.path.abspath(output_path)) == os.path.normcase(
            os.path.abspath(self.source_zip)
        ):
            messagebox.showerror("Exportação", "Escolha um destino diferente do ZIP de origem.", parent=self)
            return

        name = self._metadata_value(self.name_entry.get()) or "MaxiVloid_Character"
        author = self._metadata_value(self.author_entry.get()) or "MVX_Developer"
        image_name = os.path.basename(self.image_path) if self.image_path else ""
        icon_name = os.path.basename(self.icon_path) if self.icon_path else ""
        readme = self.readme_text.get("1.0", tk.END).strip()
        replaced = {"mxai.ini", "character.txt", "character.yaml"}
        if readme:
            replaced.add("readme.txt")
        replaced.update(filename.casefold() for filename in (image_name, icon_name) if filename)

        mxai = [
            "[MxAi_Configuration]",
            "EngineCompatible=MaximizeVoiceSynthesizerMobile",
            "GlobalType=MaxiVloid",
            f"GlobalPitchShift={self.pitch.get():.2f}",
            f"GlobalAirRatio={self.air.get():.2f}",
            f"GlobalFormantShift={self.formant.get():.2f}",
            f"GlobalGenderFactor={self.gender.get():.2f}",
            f"GlobalGrowlAmount={self.growl.get():.2f}",
            f"GlobalNaturalness={self.naturalness.get():.2f}",
            "",
            "[Amostras_Calibradas]",
        ]
        for filename in self.audio_files:
            base = PurePosixPath(filename.replace("\\", "/")).name
            oto = self.oto_data.get(base.casefold(), {})
            mxai.append(
                f"{base}=Pitch={self.pitch.get():.2f},Air={self.air.get():.2f},"
                f"Formant={self.formant.get():.2f},Gender={self.gender.get():.2f},"
                f"Growl={self.growl.get():.2f},Naturalness={self.naturalness.get():.2f},"
                f"ProtectConsonant={oto.get('consonant', 0)}"
            )
        character_txt = [
            "type=MaxiVloid",
            f"name={name}",
            f"image={image_name}",
            f"author={author}",
            f"icon={icon_name}",
            "version=Maximize_Voice_2.0",
        ]
        character_yaml = [
            "type: MaxiVloid",
            f"name: {json.dumps(name, ensure_ascii=False)}",
            f"author: {json.dumps(author, ensure_ascii=False)}",
            f"image: {json.dumps(image_name, ensure_ascii=False)}",
            f"icon: {json.dumps(icon_name, ensure_ascii=False)}",
            "voice_engine: MaximizeVoiceSynthesizerMobile",
        ]

        try:
            with zipfile.ZipFile(self.source_zip, "r") as source:
                with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as output:
                    for item in source.infolist():
                        if item.is_dir():
                            continue
                        basename = PurePosixPath(item.filename.replace("\\", "/")).name.casefold()
                        if basename not in replaced:
                            output.writestr(item, source.read(item.filename))
                    output.writestr("MxAi.ini", "\n".join(mxai))
                    output.writestr("character.txt", "\n".join(character_txt))
                    output.writestr("character.yaml", "\n".join(character_yaml))
                    if readme:
                        output.writestr("README.txt", readme)
                    if self.icon_path:
                        output.write(self.icon_path, icon_name)
                    if self.image_path:
                        output.write(self.image_path, image_name)
        except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
            messagebox.showerror("Erro de exportação", str(exc), parent=self)
            return

        self.status_label.configure(text="Pacote MaxiVloid exportado com sucesso.")
        messagebox.showinfo("Sucesso", f"Pacote exportado:\n{output_path}", parent=self)

    def _pitch_changed(self, value):
        self.pitch_label.configure(text=f"Nota musical: {value:.1f} semitons")

    def _air_changed(self, value):
        self.air_label.configure(text=f"Ar na voz: {value * 100:.0f}%")

    def _formant_changed(self, value):
        self.formant_label.configure(text=f"Timbre vocal: {value:.2f}x")

    def _gender_changed(self, value):
        self.gender_label.configure(text=f"Fator de gênero: {value:.2f}")

    def _growl_changed(self, value):
        self.growl_label.configure(text=f"Growl: {value * 100:.0f}%")

    def _natural_changed(self, value):
        self.natural_label.configure(text=f"Naturalidade: {value * 100:.0f}%")

    def _close(self):
        if winsound is not None:
            winsound.PlaySound(None, winsound.SND_PURGE)
        self.preview_dir.cleanup()
        self.destroy()


if __name__ == "__main__":
    app = MvxCreatorApp()
    app.mainloop()
