#!/usr/bin/env python3

import os
import cv2
import torch
import yaml
import zipfile
import requests
from tqdm import tqdm
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Pfad-Konfiguration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

MODEL_DIR = os.path.join(ROOT_DIR, "Models")
DATA_ROOT = os.path.join(ROOT_DIR, "Data")
WIDERFACE_DIR = os.path.join(DATA_ROOT, "widerface")
DATASET_CONFIG_PATH = os.path.join(WIDERFACE_DIR, "data.yaml")

class YOLOv8FaceTrainer:
    """
    Trainiert ein YOLOv8-Modell zur Gesichtserkennung.
    
    Diese Klasse automatisiert den gesamten Prozess:
    1. Download des WIDER FACE Datasets.
    2. Konvertierung der Annotationen in das YOLO-Format.
    3. Starten des Trainingsprozesses.
    """
    def __init__(self,
                 base_model='yolov8n.pt',
                 epochs=100,
                 batch_size=16,
                 img_size=640):
        
        self.base_model = base_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.project_name = "YOLOv8_Face_Detection"

        os.makedirs(WIDERFACE_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

    def _download_file(self, url, dest_path):
        """Lädt eine Datei mit Fortschrittsbalken herunter."""
        if os.path.exists(dest_path):
            print(f"Datei {os.path.basename(dest_path)} existiert bereits. Download übersprungen.")
            return
        
        print(f"Lade {os.path.basename(dest_path)} herunter...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(dest_path, 'wb') as f, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc="Download"
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
            print("Download abgeschlossen.")
        except requests.exceptions.RequestException as e:
            print(f"Fehler beim Download von {url}: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path) # Lösche unvollständige Datei
            raise

    def _convert_annotations(self, image_dir, annotation_file, target_label_dir):
        """Konvertiert WIDER FACE Annotationen ins YOLO-Format."""
        os.makedirs(target_label_dir, exist_ok=True)
        
        print(f"Konvertiere Annotationen aus {os.path.basename(annotation_file)}...")
        
        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        file_count = 0
        i = 0
        with tqdm(total=len(lines), desc="Annotationen konvertieren") as pbar:
            while i < len(lines):
                image_path = lines[i].strip()
                i += 1
                pbar.update(1)
                
                if not image_path.endswith('.jpg'): continue
                
                num_boxes = int(lines[i].strip())
                i += 1
                pbar.update(1)

                image_full_path = os.path.join(image_dir, image_path)
                try:
                    img = cv2.imread(image_full_path)
                    if img is None:
                        # Skip boxes for this image
                        i += num_boxes
                        pbar.update(num_boxes)
                        continue
                    
                    img_h, img_w, _ = img.shape
                except Exception as e:
                    print(f"Konnte Bild nicht laden: {image_full_path}. Überspringe. Fehler: {e}")
                    i += num_boxes
                    pbar.update(num_boxes)
                    continue

                label_rel_path = image_path.replace(".jpg", ".txt")  # enthält z. B. 0--Parade/0_Parade_marchingband_1_5.txt
                label_path = os.path.join(target_label_dir, label_rel_path)
                os.makedirs(os.path.dirname(label_path), exist_ok=True)  # erstelle Unterordner wie 0--Parade/

                yolo_annotations = []
                
                for j in range(num_boxes):
                    box_line = lines[i+j].strip().split()
                    # x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
                    x1, y1, w, h = map(int, box_line[:4])
                    invalid = int(box_line[7])

                    if invalid or w <= 0 or h <= 0:
                        continue # Ignoriere ungültige oder zu kleine Boxen

                    # Konvertiere zu YOLO-Format (center_x, center_y, width, height) - normalisiert
                    x_center = (x1 + w / 2) / img_w
                    y_center = (y1 + h / 2) / img_h
                    norm_w = w / img_w
                    norm_h = h / img_h
                    
                    # Klasse 0 für "Gesicht"
                    yolo_annotations.append(f"0 {x_center} {y_center} {norm_w} {norm_h}")

                if yolo_annotations:
                    with open(label_path, 'w') as label_file:
                        label_file.write("\n".join(yolo_annotations))
                
                i += num_boxes
                pbar.update(num_boxes)
                file_count += 1
        
        print(f"Konvertierung abgeschlossen. {file_count} Label-Dateien erstellt.")

    def prepare_widerface_dataset(self):
        """Lädt und konvertiert das WIDER FACE Dataset."""
        if os.path.exists(DATASET_CONFIG_PATH):
            print("✅ Dataset bereits vorbereitet. Überspringe Vorbereitung.")
            return

        # 1. URLs und Pfade definieren
        urls = {
            "train_images": "https://drive.usercontent.google.com/download?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M&export=download&authuser=0&confirm=t&uuid=e7dbc579-2bca-4880-bc4a-9923d57234b9&at=AN8xHooh-HEhiwv8yTfR_ujz1xO7%3A1752011527170",
            "val_images": "https://drive.usercontent.google.com/download?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q&export=download&authuser=0&confirm=t&uuid=1e2bba55-3caf-43f5-aedf-141ed6392641&at=AN8xHoqPg0G2Cu6OAI0ajEUA6IFi%3A1752011574688",
            "annotations": "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip"
        }
        zip_paths = {name: os.path.join(WIDERFACE_DIR, f"{name}.zip") for name in urls}

        # 2. Download
        for name, url in urls.items():
            self._download_file(url, zip_paths[name])
        
        # 3. Entpacken
        print("Entpacke Dateien...")
        for name, path in zip_paths.items():
            target_dir = os.path.join(WIDERFACE_DIR, "WIDER_train" if name == "train_images" else "")
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(WIDERFACE_DIR)
        
        # 4. Konvertierung
        train_img_dir = os.path.join(WIDERFACE_DIR, "WIDER_train/images")
        val_img_dir = os.path.join(WIDERFACE_DIR, "WIDER_val/images")
        train_annot_file = os.path.join(WIDERFACE_DIR, "wider_face_split/wider_face_train_bbx_gt.txt")
        val_annot_file = os.path.join(WIDERFACE_DIR, "wider_face_split/wider_face_val_bbx_gt.txt")
        
        # Zielverzeichnisse für YOLO-Format
        yolo_train_labels = os.path.join(WIDERFACE_DIR, "labels/train")
        yolo_val_labels = os.path.join(WIDERFACE_DIR, "labels/val")
        yolo_train_images = os.path.join(WIDERFACE_DIR, "images/train")
        yolo_val_images = os.path.join(WIDERFACE_DIR, "images/val")

        # Symlinks oder Verschieben der Bilder
        if not os.path.exists(yolo_train_images): os.renames(train_img_dir, yolo_train_images)
        if not os.path.exists(yolo_val_images): os.renames(val_img_dir, yolo_val_images)

        self._convert_annotations(yolo_train_images, train_annot_file, yolo_train_labels)
        self._convert_annotations(yolo_val_images, val_annot_file, yolo_val_labels)

        # 5. data.yaml erstellen
        print("Erstelle data.yaml Konfigurationsdatei...")
        data_yaml = {
            'train': os.path.abspath(yolo_train_images),
            'val': os.path.abspath(yolo_val_images),
            'nc': 1,
            'names': ['face']
        }
        with open(DATASET_CONFIG_PATH, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"✅ Dataset erfolgreich vorbereitet und unter {DATASET_CONFIG_PATH} konfiguriert.")


    def train(self):
        """Startet den gesamten Prozess: Datenvorbereitung und Training."""
        # Schritt 1: Sicherstellen, dass die Daten bereit sind
        self.prepare_widerface_dataset()

        # Schritt 2: Training starten
        print("\n🚀 Starte YOLOv8 Training zur Gesichtserkennung...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Verwende Gerät: {device}")

        model = YOLO(self.base_model)
        model.train(
            data=DATASET_CONFIG_PATH,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.img_size,
            project=MODEL_DIR,
            name=self.project_name,
            exist_ok=True,
            patience=20,
            device=device
        )

        print("✅ Training abgeschlossen.")
        best_model_path = os.path.join(MODEL_DIR, self.project_name, 'weights', 'best.pt')
        print(f"Das beste Modell wurde gespeichert unter: {best_model_path}")
        return best_model_path

# Beispiel für die Ausführung
if __name__ == '__main__':
    trainer = YOLOv8FaceTrainer(epochs=50) # Reduzierte Epochen für ein schnelleres Beispiel
    trainer.train()