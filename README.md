# CT Scanner Simulator

A graphical simulator for computed tomography (CT) scans using the Radon transform and back-projection algorithms. This Python application allows you to load images or DICOM files, generate sinograms, and reconstruct images, mimicking real-world CT scanner operations.

## Features
- **Load Images/DICOM Files:** Supports PNG, JPG, BMP, TIFF, and DICOM (.dcm) formats.
- **Interactive Sinogram Generation:** Visualize the Radon transform with customizable parameters.
- **Image Reconstruction:** Perform back-projection to reconstruct images from generated sinograms.
- **Customizable Parameters:** Adjust angle increments, number of detectors, and detector span interactively.
- **Real-time Preview:** Use sliders to preview partial reconstructions at various angles.
- **Export to DICOM:** Save reconstructed images as DICOM files with customizable patient information.

## Technologies Used
- Python
- Tkinter (GUI)
- NumPy
- Pillow (Image processing)
- PyDICOM (DICOM file handling)
- Scikit-image

## Installation

### Requirements

Make sure you have Python installed along with these dependencies:

```bash
pip install numpy pillow pydicom scikit-image
```

### Running the Application

To start the simulator, run the main script:

```bash
python tomograph_simulator.py
```

## Usage

1. **Load an Image:**
   - Click **"Wczytaj obraz/DICOM"** and select a file.

2. **Set Parameters:**
   - Adjust the **angle step (Δα)**, **number of detectors (n)**, and **detector span (l)**.

3. **Generate Sinogram:**
   - Click **"Generuj sinogram"**.

4. **Reconstruct Image:**
   - Click **"Rekonstrukcja obrazu"** to see the reconstructed image.

5. **Save as DICOM:**
   - Click **"Zapisz jako DICOM"**, fill patient data, and save.


## Authors

- Paweł Kierkosz
- Bartłomiej Rudowicz

