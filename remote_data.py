# remote_data.py
import io
import re
import requests
import tarfile
import numpy as np
from config import FIGSHARE_URLS, PHYSICOCHEMICAL_PROPERTIES
from zernike_descriptors import compute_3dzd, map_properties_to_surface, extract_patch
import os
from tqdm import tqdm
from time import sleep
import time

class FigShareLoader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate'
        })
    
    def _get_download_url(self, figshare_url):
        """Ottiene l'URL diretto di download da FigShare usando l'API"""
        try:
            # Estrai l'ID del file dall'URL
            file_id_match = re.search(r'file=(\d+)', figshare_url)
            if not file_id_match:
                raise ValueError("Impossibile estrarre l'ID del file dall'URL")
            
            file_id = file_id_match.group(1)
            # Costruisci l'URL diretto usando il servizio ndownloader di FigShare
            direct_url = f"https://ndownloader.figshare.com/files/{file_id}"
           
            
            # Verifica che l'URL sia valido facendo una richiesta HEAD
            try:
                head_response = self.session.head(direct_url, timeout=10)
                head_response.raise_for_status()
           
                return direct_url
            except Exception as e:
                print(f"Errore nella verifica dell'URL: {e}")
                # Se la verifica fallisce, prova comunque l'URL
                return direct_url
            
        except Exception as e:
            raise RuntimeError(f"Errore nell'ottenere l'URL di download: {str(e)}")

    def _get_tar_stream(self, url, max_retries=3):
     """Ottiene uno stream diretto al file tar.gz con meccanismo di riprova"""
     for attempt in range(max_retries):
        try:
            download_url = self._get_download_url(url)
            # Aumenta il timeout a 10 minuti (600 secondi)
            response = self.session.get(download_url, stream=True, timeout=600)
            response.raise_for_status()
            
            # Download con progress bar
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            
            content = bytearray()
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                content.extend(data)
            progress_bar.close()
            
            if len(content) != total_size:
                raise RuntimeError(f"Download incompleto: {len(content)}/{total_size} bytes")
            
            # Verifica che non sia una pagina di errore HTML
            if len(content) < 1000 and b'<html' in content[:100].lower():
                raise ValueError("Il contenuto sembra essere una pagina HTML di errore")
            
            return io.BytesIO(content)
            
        except (requests.exceptions.RequestException, IOError) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Errore nel download dopo {max_retries} tentativi: {str(e)}")
            print(f"Tentativo {attempt + 1} fallito, riprovo...")
            time.sleep(5 * (attempt + 1))  # Backoff esponenziale

    def _extract_file_from_tar(self, tar_stream, filename_pattern):
        """Estrae un file specifico dall'archivio tar.gz"""
        try:
            tar_stream.seek(0)
            
            # Prova prima con compressione gzip
            try:
                with tarfile.open(fileobj=tar_stream, mode='r:gz') as tar:
                    return self._find_and_extract_file(tar, filename_pattern)
            except tarfile.TarError:
                # Se fallisce, prova senza compressione
                tar_stream.seek(0)
                with tarfile.open(fileobj=tar_stream, mode='r') as tar:
                    return self._find_and_extract_file(tar, filename_pattern)
                    
        except tarfile.TarError as e:
            raise RuntimeError(f"Errore nell'estrazione del tar: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Errore generico nell'estrazione: {str(e)}")
    
    def _find_and_extract_file(self, tar, filename_pattern):
        """Trova ed estrae un file dal tar"""
        
        file_found = None
        
        for member in tar.getmembers():
            
            if member.isfile() and filename_pattern in member.name:
                file_found = member
                break
        
        if file_found is None:
            raise FileNotFoundError(f"Nessun file corrisponde a '{filename_pattern}'")
        
        
        with tar.extractfile(file_found) as file_obj:
            content = file_obj.read()
            
            # Prova a decodificare come UTF-8
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
                # Se fallisce, prova con latin-1
                try:
                    return content.decode('latin-1')
                except UnicodeDecodeError:
                    # Come ultimo tentativo, ignora i caratteri problematici
                    return content.decode('utf-8', errors='ignore')

   # remote_data.py (modifiche alla funzione _parse_features)
    def _parse_features(self, content, pdb_id=None):
     """Converte il contenuto del file in array numpy"""
  
     data = []
     lines_processed = 0
     lines_skipped = 0
    
     for line_num, line in enumerate(content.split('\n'), 1):
        line = line.strip()
        if not line:
            continue
            
        try:
            parts = line.split('\t')
            if len(parts) < 2:
                continue
                
            # Estrai etichetta (converti in 0/1)
            label = float(parts[0])
            binary_label = 1 if label == 1 else 0
            
            # Estrai features (assicurati che siano 276 come nel paper)
            features = np.zeros(276)  # 276 features come specificato nel paper
            
            for part in parts[1:]:
                if ':' in part:
                    idx_str, val_str = part.split(':', 1)
                    idx = int(idx_str)
                    val = float(val_str)
                    if 1 <= idx <= 276:
                        features[idx-1] = val
                    else:
                        raise ValueError(f"Indice feature fuori range: {idx}")
            
            # Aggiungi alla lista dei dati
            data.append(np.concatenate([[binary_label], features]))
            lines_processed += 1
            
        except Exception as e:
            lines_skipped += 1
            if lines_skipped <= 5:
                print(f"Warning: Saltata linea {line_num} - {str(e)}")
            continue
    
     if not data:
        raise ValueError("Nessun dato valido trovato nel file")
    
     data_array = np.array(data)
    
     # Verifica che ci siano entrambe le classi
     unique_labels = np.unique(data_array[:, 0])
     if len(unique_labels) < 2:
        raise ValueError(f"Solo una classe presente nel dataset: {unique_labels}")
    
    # Ritorna solo features e labels (come nel codice originale)
     return data_array[:, 1:], data_array[:, 0]  # X, y

    def load_features(self, dataset_type, pdb_id=None, return_coords=False):
     """Carica i dati direttamente da FigShare"""
     if dataset_type not in FIGSHARE_URLS:
        raise ValueError(f"Tipo dataset non valido. Scegli tra: {list(FIGSHARE_URLS.keys())}")
    
     
    
     try:
        # 1. Ottieni lo stream del file tar.gz
        tar_stream = self._get_tar_stream(FIGSHARE_URLS[dataset_type])
        
        # 2. Estrai il file specifico
        file_content = self._extract_file_from_tar(tar_stream, "_descriptors_N5.txt")
        
        # 3. Processa i dati
        X, y = self._parse_features(file_content)
        
        if return_coords:
            # Generate dummy coordinates if needed
            coords = np.random.rand(len(y), 3) * 100  # Random coordinates in 100Å cube
            return X, y, coords
        else:
            return X, y
            
     except Exception as e:
        raise RuntimeError(f"Errore nel processamento: {str(e)}")
# Istanza globale
data_loader = FigShareLoader()