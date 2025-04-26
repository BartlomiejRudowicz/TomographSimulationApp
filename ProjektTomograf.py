import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
from PIL import Image, ImageTk, ImageOps
import math
import time
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import CTImageStorage, generate_uid, ExplicitVRLittleEndian
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity

# Oblicza pozycje emitera i detektorów na podstawie kąta, rozwarcia, liczby detektorów, promienia i środka
def compute_positions(alpha, span, num_det, radius, center):
    span_rad = np.radians(span)
    offset_det = np.radians(alpha - span/2)
    offset_emit = np.radians(alpha - span/2 + 180)
    angles = np.linspace(0, span_rad, num_det)
    detectors = np.column_stack((radius * np.cos(angles + offset_det) - center[0],
                                  radius * np.sin(angles + offset_det) - center[1])).astype(int)
    emitters = np.column_stack((radius * np.cos(angles + offset_emit) - center[0],
                                 radius * np.sin(angles + offset_emit) - center[1])).astype(int)[::-1]
    return emitters, detectors

# Implementacja integerowej wersji algorytmu Bresenhama do rysowania linii
def bresenham_line(x0, y0, x1, y1):
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    dx = x1 - x0
    dy = abs(y1 - y0)
    error = dx // 2
    ystep = 1 if y0 < y1 else -1
    y = y0
    points = []
    for x in range(x0, x1 + 1):
        if steep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= dy
        if error < 0:
            y += ystep
            error += dx
    return np.array(points).T

# Łączy obliczanie pozycji i rysowanie linii dla danej projekcji
def compute_line_segments(alpha, span, num_det, radius, center):
    emitters, detectors = compute_positions(alpha, span, num_det, radius, center)
    segments = []
    for e, d in zip(emitters, detectors):
        segments.append(bresenham_line(e[0], e[1], d[0], d[1]))
    return segments

# Powiększa obraz do kwadratowego rozmiaru o wymiarach równej przekątnej
def pad_image(array):
    w, h = array.shape
    side = int(np.ceil((w**2 + h**2)**0.5))
    shape = (side, side)
    pad = (np.array(shape) - np.array(array.shape)) / 2
    pad = np.array([np.floor(pad), np.ceil(pad)]).T.astype(int)
    return np.pad(array, pad)

# Normalizuje wartości tablicy do zakresu 0-255
def normalize_to_255(arr):
    arr = arr.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return arr * 255

# Przywraca oryginalny rozmiar obrazu z powiększonego obrazu
def unpad(img, height, width):
    y, x = img.shape
    startx = x // 2 - (width // 2)
    starty = y // 2 - (height // 2)
    return img[starty:starty+height, startx:startx+width]

# Oblicza pojedynczą projekcję (wiersz sinogramu) dla danego kąta
def compute_projection(image, alpha, span, num_det, radius, center):
    segments = compute_line_segments(alpha, span, num_det, radius, center)
    proj = np.array([np.sum(image[tuple(seg)]) for seg in segments])
    return normalize_to_255(proj)

# Oblicza sinogram dla obrazu przez generowanie projekcji dla wielu kątów
def radon_transform(image, num_scans, num_det, span):
    padded = pad_image(image)
    center = np.floor(np.array(padded.shape) / 2).astype(int)
    radius = padded.shape[0] // 2
    angles = np.linspace(0, 180, num_scans)
    projections = [compute_projection(padded, ang, span, num_det, radius, center) for ang in angles]
    sino = np.array(projections)
    return np.swapaxes(sino, 0, 1)

# Aktualizuje obraz akumulacyjny i macierz zliczającą na podstawie pojedynczej projekcji
def backproject_projection(accum, count, proj, alpha, span, num_det, radius, center):
    segments = compute_line_segments(alpha, span, num_det, radius, center)
    for idx, seg in enumerate(segments):
        accum[tuple(seg)] += proj[idx]
        count[tuple(seg)] += 1

# Rekonstruuje obraz z sinogramu przez back-projection i przycina do oryginalnego rozmiaru
def inverse_radon_transform(orig_shape, sinogram, span):
    num_det, num_scans = sinogram.shape
    sino = np.swapaxes(sinogram, 0, 1)
    recon = np.zeros(orig_shape, dtype=np.float32)
    recon = pad_image(recon)
    count = np.zeros_like(recon)
    center_coords = np.floor(np.array(recon.shape) / 2).astype(int)
    radius_val = recon.shape[0] // 2
    angles = np.linspace(0, 180, num_scans)
    for ang, proj in zip(angles, sino):
        backproject_projection(recon, count, proj, ang, span, num_det, radius_val, center_coords)
    count[count == 0] = 1
    recon /= count
    recon = normalize_to_255(recon)
    return unpad(recon, *orig_shape)

# Konwertuje obraz do formatu uint8 dla DICOM
def convert_image_to_ubyte_dicom(img):
    return img_as_ubyte(rescale_intensity(img, out_range=(0.0, 1.0)))

# Zapisuje obraz w formacie DICOM wraz z danymi pacjenta
def save_as_dicom(file_name, img, patient_data):
    img_converted = convert_image_to_ubyte_dicom(img)
    meta = Dataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, preamble=b"\0" * 128)
    ds.file_meta = meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientName = patient_data["PatientName"]
    ds.PatientID = patient_data["PatientID"]
    ds.ImageComments = patient_data["ImageComments"]
    ds.StudyDate = patient_data["StudyDate"]
    ds.Modality = "CT"
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = generate_uid()
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.SamplesPerPixel = 1
    ds.HighBit = 7
    ds.ImagesInAcquisition = 1
    ds.InstanceNumber = 1
    ds.Rows, ds.Columns = img_converted.shape
    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.PixelData = img_converted.tobytes()
    ds.save_as(file_name, write_like_original=False)

# Aplikacja GUI do symulacji tomografu
class TomographApp:
    def __init__(self, master):
        self.master = master
        master.title("Symulator Tomografu (Projekt)")
        self.original_image = None
        self.sinogram = None
        self.reconstructed = None
        self.angle_step = tk.DoubleVar(value=1.0)
        self.num_detectors = tk.IntVar(value=180)
        self.det_length = tk.DoubleVar(value=180.0)
        self.angles = []
        self.create_widgets()

    def create_widgets(self):
        file_frame = tk.Frame(self.master)
        file_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        btn_load = tk.Button(file_frame, text="Wczytaj obraz/DICOM", command=self.load_image)
        btn_load.pack(side=tk.LEFT, padx=5)
        params_frame = tk.LabelFrame(self.master, text="Parametry Tomografu")
        params_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Label(params_frame, text="Krok kątowy (Δα): ").grid(row=0, column=0, sticky="e")
        tk.Entry(params_frame, textvariable=self.angle_step, width=6).grid(row=0, column=1, sticky="w")
        tk.Label(params_frame, text="Liczba detektorów (n): ").grid(row=0, column=2, sticky="e")
        tk.Entry(params_frame, textvariable=self.num_detectors, width=6).grid(row=0, column=3, sticky="w")
        tk.Label(params_frame, text="Rozwarcie (l): ").grid(row=0, column=4, sticky="e")
        tk.Entry(params_frame, textvariable=self.det_length, width=6).grid(row=0, column=5, sticky="w")
        action_frame = tk.Frame(self.master)
        action_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        btn_sino = tk.Button(action_frame, text="Generuj sinogram", command=self.generate_sinogram)
        btn_sino.pack(side=tk.LEFT, padx=5)
        btn_recon = tk.Button(action_frame, text="Rekonstrukcja obrazu", command=self.reconstruct_image)
        btn_recon.pack(side=tk.LEFT, padx=5)
        btn_save_dicom = tk.Button(action_frame, text="Zapisz jako DICOM", command=self.save_dicom_dialog)
        btn_save_dicom.pack(side=tk.LEFT, padx=5)
        slider_frame = tk.LabelFrame(self.master, text="Podgląd kątów / sinogramu")
        slider_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.angle_slider = tk.Scale(slider_frame, from_=0, to=180, orient=tk.HORIZONTAL, command=self.update_partial_projection)
        self.angle_slider.pack(fill=tk.X, padx=5, pady=5)
        self.image_frame = tk.Frame(self.master)
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.label_original = tk.Label(self.image_frame, text="Oryginalny")
        self.label_original.pack(side=tk.LEFT, padx=5)
        self.label_sinogram = tk.Label(self.image_frame, text="Sinogram")
        self.label_sinogram.pack(side=tk.LEFT, padx=5)
        self.label_reconstruction = tk.Label(self.image_frame, text="Rekonstrukcja")
        self.label_reconstruction.pack(side=tk.LEFT, padx=5)
        self.status_label = tk.Label(self.master, text="Wczytaj obraz lub plik DICOM, ustaw parametry i rozpocznij.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Wybierz obraz lub DICOM",
            filetypes=[("Obrazy/DICOM", "*.png *.jpg *.bmp *.tif *.dcm"), ("Wszystkie pliki", "*.*")]
        )
        if not file_path:
            return
        if file_path.lower().endswith(".dcm"):
            try:
                ds = pydicom.dcmread(file_path)
                pixel_array = ds.pixel_array
                self.original_image = pixel_array.astype(np.float64)
                self.show_image_in_label(self.label_original, self.original_image)
                self.status_label.config(text=f"Wczytano plik DICOM: {file_path}")
                self.sinogram = None
                self.reconstructed = None
            except Exception as e:
                self.status_label.config(text=f"Nie udało się wczytać DICOM: {e}")
        else:
            pil_img = Image.open(file_path).convert("L")
            self.original_image = np.array(pil_img, dtype=np.float64)
            self.show_image_in_label(self.label_original, self.original_image)
            self.status_label.config(text=f"Wczytano: {file_path}")
            self.sinogram = None
            self.reconstructed = None

    def generate_sinogram(self):
        if self.original_image is None:
            self.status_label.config(text="Najpierw wczytaj obraz.")
            return
        step = self.angle_step.get()
        angles = list(np.arange(0, 180, step))
        self.angles = angles
        n_det = self.num_detectors.get()
        span = self.det_length.get()
        t0 = time.time()
        self.sinogram = radon_transform(self.original_image, len(angles), n_det, span)
        t1 = time.time()
        if self.sinogram is not None:
            sino_display = self.normalize_and_convert(self.sinogram)
            self.show_image_in_label(self.label_sinogram, sino_display)
            self.status_label.config(text=f"Sinogram wygenerowany. Czas: {t1-t0:.2f}s")

    def reconstruct_image(self):
        if self.sinogram is None:
            self.status_label.config(text="Brak sinogramu. Najpierw wygeneruj sinogram.")
            return
        h, w = self.original_image.shape
        n_det = self.num_detectors.get()
        span = self.det_length.get()
        t0 = time.time()
        self.reconstructed = inverse_radon_transform((h, w), self.sinogram, span)
        t1 = time.time()
        recon_display = self.normalize_and_convert(self.reconstructed)
        self.show_image_in_label(self.label_reconstruction, recon_display)
        self.status_label.config(text=f"Rekonstrukcja gotowa. Czas: {t1-t0:.2f}s")

    def save_dicom_dialog(self):
        if self.reconstructed is None:
            self.status_label.config(text="Brak obrazu wynikowego do zapisu.")
            return
        dicom_window = tk.Toplevel(self.master)
        dicom_window.title("Zapis do DICOM")
        tk.Label(dicom_window, text="Patient Name:").grid(row=0, column=0, sticky='e')
        entry_name = tk.Entry(dicom_window)
        entry_name.grid(row=0, column=1)
        tk.Label(dicom_window, text="Patient ID:").grid(row=1, column=0, sticky='e')
        entry_id = tk.Entry(dicom_window)
        entry_id.grid(row=1, column=1)
        tk.Label(dicom_window, text="Komentarz:").grid(row=2, column=0, sticky='e')
        entry_comment = tk.Entry(dicom_window)
        entry_comment.grid(row=2, column=1)
        tk.Label(dicom_window, text="Data badania (StudyDate):").grid(row=3, column=0, sticky='e')
        entry_date = tk.Entry(dicom_window)
        entry_date.grid(row=3, column=1)
        def on_save():
            file_path = filedialog.asksaveasfilename(defaultextension=".dcm", filetypes=[("DICOM", "*.dcm")])
            if file_path:
                patient_data = {
                    "PatientName": entry_name.get(),
                    "PatientID": entry_id.get(),
                    "ImageComments": entry_comment.get(),
                    "StudyDate": entry_date.get()
                }
                save_as_dicom(file_path, self.reconstructed, patient_data)
                self.status_label.config(text=f"Zapisano do pliku: {file_path}")
            dicom_window.destroy()
        btn_ok = tk.Button(dicom_window, text="Zapisz", command=on_save)
        btn_ok.grid(row=4, column=1, pady=5)

    def update_partial_projection(self, val):
        if self.sinogram is not None and self.original_image is not None:
            max_angle = float(val)
            angles_partial = [a for a in self.angles if a <= max_angle]
            if len(angles_partial) > 1:
                h, w = self.original_image.shape
                n_det = self.num_detectors.get()
                span = self.det_length.get()
                partial_sino = self.sinogram[:, :len(angles_partial)]
                partial_recon = inverse_radon_transform((h, w), partial_sino, span)
                partial_disp = self.normalize_and_convert(partial_recon)
                self.show_image_in_label(self.label_reconstruction, partial_disp)

    def normalize_and_convert(self, img):
        img_cpy = img.copy()
        mmin, mmax = img_cpy.min(), img_cpy.max()
        if mmax > mmin:
            img_cpy = (img_cpy - mmin) / (mmax - mmin)
        img_cpy = (img_cpy * 255.0).astype(np.uint8)
        return img_cpy

    def show_image_in_label(self, label_widget, img_array):
        pil_img = Image.fromarray(img_array)
        pil_img_small = pil_img.resize((400, 400), Image.Resampling.BILINEAR)
        imgtk = ImageTk.PhotoImage(image=pil_img_small)
        label_widget.configure(image=imgtk)
        label_widget.image = imgtk

if __name__ == "__main__":
    root = tk.Tk()
    app = TomographApp(root)
    root.mainloop()
